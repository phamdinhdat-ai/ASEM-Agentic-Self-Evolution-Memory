"""
Run the LoCoMo baseline benchmark against all ASEM systems.

Converts data/training/val.jsonl into the format expected by eval/evaluate.py,
builds all systems (NoMemory, FullContext, SimRetrieval, AtomicLinking,
RLManagerOnly, ValueRetrievalOnly, ASEM), runs evaluation, and saves results.

Usage
-----
    python scripts/run_locomo_benchmark.py \
        --val     data/training/val.jsonl \
        --config  configs/locomo_0.5b.yaml \
        --results data/benchmarks/results/locomo_baseline.json \
        --db-dir  data/benchmarks/eval_banks

Then generate the Markdown table:
    python scripts/make_results_table.py \
        --results data/benchmarks/results/locomo_baseline.json \
        --output  data/benchmarks/results/locomo_baseline_table.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import evaluate as hf_evaluate  # for per-pair BERTScore
from dotenv import load_dotenv

file_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(file_dir, "..", ".env"))


# Ensure project root is on sys.path when running as a script
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env from project root (OPENAI_API_KEY, OPENAI_BASE_URL, etc.)
_dotenv_path = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path, override=False)
    except ImportError:
        # Fallback: parse .env manually if python-dotenv is not installed
        with open(_dotenv_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    _v = _v.strip().strip('"').strip("'")
                    os.environ.setdefault(_k.strip(), _v)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def locomo_to_eval_format(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert generated training examples to the format eval/evaluate.py expects.

    eval/evaluate.py:run_baseline() needs each item to have:
      - "query"   : str
      - "answer"  : str   (gold answer)
      - "history" : List[str]  (context strings passed to baseline.answer())

    Our val examples have:
      - "query"        : str  (already enriched with speaker names)
      - "gold_answer"  : str
      - "candidates"   : List[dict]  each with "content" key
      - "category"     : int
    """
    converted = []
    for ex in examples:
        converted.append({
            "query":    ex["query"],
            "answer":   ex["gold_answer"],
            "history":  [c["content"] for c in ex.get("candidates", [])],
            "category": ex.get("category", 0),
            "category_name": ex.get("category_name", ""),
            "session_id":    ex.get("session_id", ""),
        })
    return converted


# ---------------------------------------------------------------------------
# Per-category breakdown
# ---------------------------------------------------------------------------

def split_by_category(examples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Return a dict of category_name -> examples for per-category reporting."""
    from collections import defaultdict
    by_cat: Dict[str, List] = defaultdict(list)
    for ex in examples:
        key = ex.get("category_name") or f"cat{ex.get('category', 0)}"
        by_cat[key].append(ex)
    return dict(by_cat)


def _get_backend(system: Any) -> Any:
    """Extract the InferenceBackend from any system (baseline or ASEM)."""
    # Baselines (NoMemory, FullContext, SimRetrieval, etc.) have .backend
    if hasattr(system, "backend"):
        return system.backend
    # ASEMSystem has .pipeline, which has components with .backend
    if hasattr(system, "pipeline"):
        pipeline = system.pipeline
        for attr in ["note_constructor", "answer_agent", "memory_manager",
                     "link_evolver", "retriever", "utility_updater"]:
            comp = getattr(pipeline, attr, None)
            if comp is not None and hasattr(comp, "backend"):
                return comp.backend
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LoCoMo baseline benchmark for ASEM systems"
    )
    parser.add_argument(
        "--val",
        default="data/training/val.jsonl",
        help="Path to val JSONL (generated by generate_training_data.py)",
    )
    parser.add_argument(
        "--config",
        default="configs/locomo_0.5b.yaml",
        help="Path to YAML config (inference + hyperparameters)",
    )
    parser.add_argument(
        "--results",
        default="data/benchmarks/results/locomo_baseline.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/eval_banks",
        help="Directory for SQLite memory banks used during eval",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=(
            "Subset of systems to run. Choices: NoMemory FullContext SimRetrieval "
            "AtomicLinking RLManagerOnly ValueRetrievalOnly ASEM. "
            "Default: all systems."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["em", "rougeL"],
        help="Metrics to compute: em rougeL bertscore_f1 (default: em rougeL)",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Also report metrics broken down by QA category",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate on the first N examples (useful for smoke-testing the pipeline)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"],
        help="Loguru log level for console output (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path for persistent debug log file (rotated, compressed)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup loguru logging early so all modules benefit
    # ------------------------------------------------------------------
    from asem.logging_utils import setup_logging
    setup_logging(level=args.log_level, log_file=args.log_file)

    # ------------------------------------------------------------------
    # Imports — give clear errors if deps are missing
    # ------------------------------------------------------------------
    try:
        from eval.evaluate import EvalConfig, DatasetPaths, compute_metrics
        from eval.systems import get_systems
    except ImportError as e:
        print(f"ERROR: Could not import eval modules: {e}")
        print("Make sure you are running from the project root:")
        print("  cd <project-root> && python scripts/run_locomo_benchmark.py ...")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load and convert val data
    # ------------------------------------------------------------------
    print(f"Loading val data from {args.val} ...")
    raw_val = load_jsonl(args.val)
    print(f"  {len(raw_val)} examples loaded")

    val_data = locomo_to_eval_format(raw_val)
    print(f"  Converted to eval format")

    if args.limit is not None:
        val_data = val_data[: args.limit]
        print(f"  Limited to first {len(val_data)} examples (--limit {args.limit})")

    # ------------------------------------------------------------------
    # Build systems
    # ------------------------------------------------------------------
    print(f"\nBuilding systems from config {args.config} ...")
    all_systems = get_systems(config_path=args.config, db_dir=args.db_dir)

    if args.systems:
        unknown = set(args.systems) - set(all_systems.keys())
        if unknown:
            print(f"WARNING: Unknown systems requested: {unknown}")
            print(f"  Available: {list(all_systems.keys())}")
        systems = {k: v for k, v in all_systems.items() if k in args.systems}
    else:
        systems = all_systems

    print(f"  Running systems: {list(systems.keys())}")

    # ------------------------------------------------------------------
    # Run evaluation — incremental saving so a crash doesn't lose work
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.results), exist_ok=True)

    # EvalConfig used only for compute_metrics; DatasetPaths not used in the loop
    eval_config = EvalConfig(
        datasets=DatasetPaths(
            longmemeval=args.val,
            locomo=args.val,
            personalmembench=args.val,
        ),
        results_path=args.results,
        metrics=args.metrics,
    )

    preds_dir = os.path.join(os.path.dirname(args.results), "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Load any previously completed results so we can resume
    results: Dict[str, Any] = {}
    if os.path.exists(args.results):
        with open(args.results, "r", encoding="utf-8") as fh:
            try:
                results = json.load(fh)
            except json.JSONDecodeError:
                results = {}

    def _flush_results(res: Dict[str, Any]) -> None:
        """Atomically write results JSON so a crash never leaves a corrupt file."""
        tmp = args.results + ".tmp"
        with open(tmp, "w", encoding="utf-8") as _fh:
            json.dump(res, _fh, indent=2)
        os.replace(tmp, args.results)

    print(f"\nRunning evaluation ({args.metrics}) ...")

    # Lazy-load BERTScore for per-pair scoring (only if requested)
    _bertscore = None
    if "bertscore_f1" in args.metrics:
        try:
            _bertscore = hf_evaluate.load("bertscore")
        except Exception as exc:
            print(f"WARNING: Could not load bertscore: {exc}")

    for sys_name, system in systems.items():
        key = f"locomo/{sys_name}"
        partial_key = f"locomo/{sys_name}/__partial__"

        if key in results:
            print(f"  [{sys_name}] already completed — skipping (delete {args.results} to re-run)")
            continue

        # Reset token counter before starting this system
        backend = _get_backend(system)
        if backend is not None and hasattr(backend, "reset_token_count"):
            backend.reset_token_count()

        preds_path = os.path.join(preds_dir, f"locomo_{sys_name}.jsonl")
        # Load any partial predictions from a previous interrupted run
        done_ids: set = set()
        preds_so_far: List[str] = []
        refs_so_far: List[str] = []
        pair_bertscores: List[float] = []  # per-pair BERTScore F1
        if os.path.exists(preds_path):
            with open(preds_path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        done_ids.add(rec["idx"])
                        preds_so_far.append(rec["pred"])
                        refs_so_far.append(rec["ref"])
                        if "bertscore_f1" in rec:
                            pair_bertscores.append(rec["bertscore_f1"])
            if done_ids:
                print(f"  [{sys_name}] resuming from {len(done_ids)} saved predictions")
                # Immediately publish partial metrics so the JSON is never stale
                partial_metrics = compute_metrics(preds_so_far, refs_so_far, eval_config)
                partial_metrics["__n__"] = len(done_ids)
                partial_metrics["__total__"] = len(val_data)
                results[partial_key] = partial_metrics
                _flush_results(results)
                print(f"  [{sys_name}] partial metrics on {len(done_ids)} examples: {partial_metrics}")

        with open(preds_path, "a", encoding="utf-8") as fp:
            for idx, item in enumerate(val_data):
                if idx in done_ids:
                    continue
                query = str(item.get("query", ""))
                ref = str(item.get("answer", ""))
                history = [str(h) for h in item.get("history", [])]

                t0 = time.perf_counter()
                try:
                    pred = system.answer(query, history)
                except Exception as exc:
                    print(f"\n  [{sys_name}] ERROR on example {idx}: {exc}")
                    traceback.print_exc()
                    pred = ""
                elapsed = time.perf_counter() - t0

                preds_so_far.append(pred)
                refs_so_far.append(ref)

                # Per-pair BERTScore (batch of 1, cached for efficiency)
                pair_bs_f1 = None
                if _bertscore is not None and pred and ref:
                    try:
                        bs_out = _bertscore.compute(
                            predictions=[pred], references=[ref], lang=eval_config.bertscore_lang
                        )
                        pair_bs_f1 = float(bs_out["f1"][0])
                        pair_bertscores.append(pair_bs_f1)
                    except Exception:
                        pass

                # Write prediction immediately so progress survives a crash
                rec = {
                    "idx": idx,
                    "session_id": item.get("session_id", ""),
                    "query": query,
                    "pred": pred,
                    "ref": ref,
                }
                if pair_bs_f1 is not None:
                    rec["bertscore_f1"] = pair_bs_f1
                fp.write(json.dumps(rec) + "\n")
                fp.flush()

                # --- Pretty-print gold vs pred ---
                print(f"\n  [{sys_name}] #{idx + 1}/{len(val_data)}  ({elapsed:.1f}s)")
                print(f"    GOLD : {ref[:120]}{'…' if len(ref) > 120 else ''}")
                print(f"    PRED : {pred[:120]}{'…' if len(pred) > 120 else ''}")
                if pair_bs_f1 is not None:
                    print(f"    BERTScore F1: {pair_bs_f1:.4f}")

                # Update partial metrics in the JSON every 10 predictions
                if (idx + 1) % 10 == 0 or (idx + 1) == len(val_data):
                    partial_metrics = compute_metrics(preds_so_far, refs_so_far, eval_config)
                    partial_metrics["__n__"] = len(preds_so_far)
                    partial_metrics["__total__"] = len(val_data)
                    if pair_bertscores:
                        partial_metrics["bertscore_f1_per_pair"] = (
                            sum(pair_bertscores) / len(pair_bertscores)
                        )
                    # Token count so far
                    if backend is not None and hasattr(backend, "token_count"):
                        partial_metrics["__tokens__"] = backend.token_count
                    results[partial_key] = partial_metrics
                    _flush_results(results)

        # Final metrics — promote from partial to permanent and clean up partial key
        final_metrics = compute_metrics(preds_so_far, refs_so_far, eval_config)
        if pair_bertscores:
            final_metrics["bertscore_f1_per_pair"] = (
                sum(pair_bertscores) / len(pair_bertscores)
            )
        if backend is not None and hasattr(backend, "token_count"):
            final_metrics["total_tokens"] = backend.token_count
        results[key] = final_metrics
        results.pop(partial_key, None)
        _flush_results(results)

        # Print final summary for this system
        token_str = ""
        if backend is not None and hasattr(backend, "token_count"):
            token_str = f"  tokens={backend.token_count:,}"
        print(f"\n  [{sys_name}] FINAL metrics: {final_metrics}{token_str}  (saved to {args.results})")

    # ------------------------------------------------------------------
    # Per-category breakdown (optional)
    # ------------------------------------------------------------------
    if args.per_category:
        print("\nComputing per-category breakdown ...")
        by_cat = split_by_category(val_data)
        cat_results: Dict[str, Any] = {}

        for cat_name, cat_examples in by_cat.items():
            cat_results[cat_name] = {}
            for sys_name, system in systems.items():
                preds = [system.answer(ex["query"], ex["history"]) for ex in cat_examples]
                refs  = [ex["answer"] for ex in cat_examples]
                cat_results[cat_name][sys_name] = compute_metrics(preds, refs, eval_config)

        # Merge into results
        for cat_name, sys_metrics in cat_results.items():
            for sys_name, metrics in sys_metrics.items():
                key = f"locomo_cat_{cat_name}/{sys_name}"
                results[key] = metrics

        # Re-save with category breakdown included
        _flush_results(results)

    # ------------------------------------------------------------------
    # Print summary table to stdout
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"RESULTS  (saved to {args.results})")
    print(f"{'='*60}")

    header_metrics = args.metrics
    col_w = 14
    # Build dynamic header — add extra columns if any system has them
    has_bertscore_pair = any(
        "bertscore_f1_per_pair" in metrics
        for key, metrics in results.items()
        if "/" in key and key.startswith("locomo/")
    )
    has_tokens = any(
        "total_tokens" in metrics
        for key, metrics in results.items()
        if "/" in key and key.startswith("locomo/")
    )

    header = f"{'System':<30}" + "".join(f"{m:>{col_w}}" for m in header_metrics)
    if has_bertscore_pair:
        header += f"{'bs_pair':>{col_w}}"
    if has_tokens:
        header += f"{'tokens':>12}"
    print(header)
    print("-" * len(header))

    for key, metrics in sorted(results.items()):
        if "/" not in key:
            continue
        dataset, sys_name = key.split("/", 1)
        if dataset != "locomo":
            continue
        if "__partial__" in sys_name:
            continue
        row = f"{sys_name:<30}" + "".join(
            f"{metrics.get(m, 0.0):>{col_w}.4f}" for m in header_metrics
        )
        if has_bertscore_pair:
            bs_pair = metrics.get("bertscore_f1_per_pair", 0.0)
            row += f"{bs_pair:>{col_w}.4f}"
        if has_tokens:
            tok = metrics.get("total_tokens", 0)
            row += f"{tok:>12,}"
        print(row)

    print(f"\nDone. Results written to: {args.results}")
    print(f"Next step: python scripts/make_results_table.py "
          f"--results {args.results} "
          f"--output data/benchmarks/results/locomo_baseline_table.md")


if __name__ == "__main__":
    main()
