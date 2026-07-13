#!/usr/bin/env python
"""Run ASEM v2 two-phase benchmark on locomo10.json.

Two-phase architecture:
  Phase 1 (offline): Batch ingest ALL dialogue turns once per conversation.
  Phase 2 (online):  Answer ALL QA pairs from the pre-built knowledge graph.

This eliminates the dedup bug in v1 where the same turns were re-ingested
for every QA pair (~108,000 redundant LLM calls per conversation).

Usage::

    # Smoke test (10 QA pairs)
    python scripts/run_asem_v2.py --limit 10

    # Full benchmark
    python scripts/run_asem_v2.py --metrics em rougeL bertscore_f1

    # Compare with v1 baselines
    python scripts/run_asem_v2.py --systems ASEM ASEMv2 SimRetrieval --limit 50

    # Per-category breakdown
    python scripts/run_asem_v2.py --per-category
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure project root on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env
_dotenv_path = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_dotenv_path):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv_path, override=False)
    except ImportError:
        with open(_dotenv_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# Import data conversion from existing script
from scripts.run_locomo10_experiments import (
    convert_locomo10_to_eval,
    group_by_conversation,
    split_by_category,
    compute_metrics,
    CATEGORY_NAMES,
)


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _build_dialogue_turns(conversation: Dict[str, Any]) -> List[str]:
    """Collect ALL unique dialogue turns from a conversation, ordered by session+turn."""
    import re

    turns: List[Tuple[int, int, str]] = []  # (session, turn_num, text)
    seen: Set[str] = set()

    for key, value in conversation.items():
        if not key.startswith("session_") or key.endswith("date_time"):
            continue
        if not isinstance(value, list):
            continue
        for turn in value:
            dia_id = turn.get("dia_id", "")
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            blip = turn.get("blip_caption", "")

            content = f"[{speaker}] {text}"
            if blip:
                content += f" (image: {blip})"

            content_hash = hash(content)
            if content_hash in seen:
                continue
            seen.add(content_hash)

            m = re.match(r"D(\d+):(\d+)", dia_id)
            sess = int(m.group(1)) if m else 0
            turn_num = int(m.group(2)) if m else 0
            turns.append((sess, turn_num, content))

    turns.sort(key=lambda x: (x[0], x[1]))
    return [t[2] for t in turns]


def _build_system_runner(name: str, config_path: str, db_dir: str):
    """Build a single system runner (v1 baseline, ASEM, or ASEMv2)."""
    from eval.systems import (
        build_asem_system,
        build_asem_v2_system,
        build_baselines,
    )

    os.makedirs(db_dir, exist_ok=True)

    if name == "ASEM":
        return build_asem_system(config_path, db_dir)
    elif name == "ASEMv2":
        return build_asem_v2_system(config_path, db_dir)
    else:
        baselines = build_baselines(config_path, db_dir)
        if name in baselines:
            return baselines[name]
        raise ValueError(f"Unknown system: {name}")


def evaluate_v2(
    runner,
    conversation_groups: List[List[Dict[str, Any]]],
    raw_dataset: List[Dict[str, Any]],
    metric_names: List[str],
    preds_dir: str,
    sys_name: str,
    results: Dict[str, Any],
    results_path: str,
) -> Tuple[List[str], List[str]]:
    """Evaluate using the two-phase architecture.

    Phase 1: Pre-ingest all turns ONCE per conversation.
    Phase 2: Answer all QA pairs without re-ingestion.
    """
    preds_path = os.path.join(preds_dir, f"locomo10_v2_{sys_name}.jsonl")
    done_ids: Set[int] = set()
    preds_so_far: List[str] = []
    refs_so_far: List[str] = []

    # Resume from partial predictions
    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_ids.add(rec["idx"])
                    preds_so_far.append(rec["pred"])
                    refs_so_far.append(rec["ref"])
        if done_ids:
            print(f"    Resuming from {len(done_ids)} saved predictions")

    def _flush(res):
        tmp = results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        os.replace(tmp, results_path)

    # Evaluate conversation by conversation
    with open(preds_path, "a", encoding="utf-8") as fp:
        for group_idx, group in enumerate(conversation_groups):
            # Determine which raw conversation this group belongs to
            conv_idx = group_idx  # They map 1:1 in order

            # Phase 1: Pre-ingest all dialogue turns for this conversation
            t0 = time.perf_counter()
            if hasattr(runner, 'ingest_conversation') and conv_idx < len(raw_dataset):
                conv = raw_dataset[conv_idx].get("conversation", {})
                turns = _build_dialogue_turns(conv)
                if hasattr(runner, '_ingested') and not runner._ingested:
                    n_notes = runner.ingest_conversation(turns)
                    ingest_time = time.perf_counter() - t0
                    print(f"  [Phase 1] conv={conv_idx}  turns={len(turns)}  "
                          f"notes={n_notes}  time={ingest_time:.1f}s")
            elif hasattr(runner, 'reset'):
                # v1 systems: reset at conversation boundary
                runner.reset()

            # Phase 2: Answer all QA pairs (no re-ingestion)
            for item in group:
                idx = item.get("_idx", -1)
                if idx in done_ids:
                    continue

                query = str(item.get("query", ""))
                ref = str(item.get("answer", ""))
                history = [str(h) for h in item.get("history", [])]

                try:
                    if hasattr(runner, '_ingested') and runner._ingested:
                        pred = runner.answer(query)  # v2: no history needed
                    else:
                        pred = runner.answer(query, history)  # v1: pass history
                except Exception as exc:
                    print(f"\n    ERROR on example {idx}: {exc}")
                    traceback.print_exc()
                    pred = ""

                preds_so_far.append(pred)
                refs_so_far.append(ref)

                fp.write(json.dumps({
                    "idx": idx,
                    "session_id": item.get("session_id", ""),
                    "category": item.get("category", 0),
                    "category_name": item.get("category_name", ""),
                    "query": query,
                    "pred": pred,
                    "ref": ref,
                }) + "\n")
                fp.flush()

                n_done = len(done_ids) + len(preds_so_far)
                if n_done % 10 == 0:
                    pct = n_done / sum(len(g) for g in conversation_groups) * 100
                    print(f"    [{sys_name}] {n_done} ({pct:.0f}%)  "
                          f"latest pred: {pred[:80]!r}")

                # Partial metrics every 50
                if n_done % 50 == 0:
                    partial_key = f"locomo10/{sys_name}/__partial__"
                    partial_metrics = compute_metrics(preds_so_far, refs_so_far, metric_names)
                    partial_metrics["__n__"] = n_done
                    results[partial_key] = partial_metrics
                    _flush(results)

    return preds_so_far, refs_so_far


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASEM v2 two-phase benchmark on locomo10.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="datasets/locomo/locomo10.json")
    parser.add_argument("--config", default="configs/locomo_openai.yaml")
    parser.add_argument(
        "--results",
        default="data/benchmarks/results/locomo10_v2.json",
    )
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/eval_banks_v2",
    )
    parser.add_argument(
        "--systems", nargs="+", default=["ASEMv2"],
        help="Systems to run: NoMemory FullContext SimRetrieval AtomicLinking "
             "RLManagerOnly ValueRetrievalOnly ASEM ASEMv2",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=["em", "rougeL"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-category", action="store_true")
    parser.add_argument(
        "--max-history-turns", type=int, default=0,
    )
    parser.add_argument(
        "--clean", action="store_true",
    )
    args = parser.parse_args()

    # ---- Timestamped DB dir ----
    db_dir = os.path.join(args.db_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(db_dir, exist_ok=True)
    print(f"DB dir: {db_dir}")

    # ---- Step 1: Load & convert data ----
    print("=" * 60)
    print("STEP 1: Load locomo10.json")
    print("=" * 60)
    raw_dataset = json.load(open(args.input, "r", encoding="utf-8"))
    eval_data = convert_locomo10_to_eval(args.input, limit=args.limit)
    conversation_groups = group_by_conversation(eval_data)
    total_items = sum(len(g) for g in conversation_groups)
    print(f"  {len(conversation_groups)} conversations, {total_items} QA pairs")

    # Category distribution
    by_cat = split_by_category(eval_data)
    print("Category distribution:")
    for cat_name, items in sorted(by_cat.items()):
        print(f"  {cat_name}: {len(items)}")

    # ---- Step 2: Evaluate ----
    print(f"\n{'='*60}")
    print(f"STEP 2: Evaluate ({args.metrics})")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    preds_dir = os.path.join(os.path.dirname(args.results), "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Clean mode
    if args.clean:
        import shutil
        for path in [args.results, preds_dir, db_dir]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        os.makedirs(preds_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)
        print("  Cleaned all previous results.")

    # Load previous results for resume
    results: Dict[str, Any] = {}
    if os.path.exists(args.results):
        with open(args.results, "r", encoding="utf-8") as fh:
            try:
                results = json.load(fh)
            except json.JSONDecodeError:
                results = {}

    def _flush(res):
        tmp = args.results + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        os.replace(tmp, args.results)

    for sys_name in args.systems:
        key = f"locomo10/{sys_name}"
        if key in results:
            print(f"\n  [{sys_name}] already completed -- skipping")
            continue

        print(f"\n  [{sys_name}] building system ...")
        runner = _build_system_runner(sys_name, args.config, db_dir)
        print(f"  [{sys_name}] running on {total_items} examples ...")

        preds, refs = evaluate_v2(
            runner=runner,
            conversation_groups=conversation_groups,
            raw_dataset=raw_dataset,
            metric_names=args.metrics,
            preds_dir=preds_dir,
            sys_name=sys_name,
            results=results,
            results_path=args.results,
        )

        final_metrics = compute_metrics(preds, refs, args.metrics)
        results[key] = final_metrics
        results.pop(f"locomo10/{sys_name}/__partial__", None)
        _flush(results)
        print(f"    [{sys_name}] FINAL: {final_metrics}")

    # ---- Step 3: Print summary ----
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    col_w = 16
    header = f"{'System':<30}" + "".join(f"{m:>{col_w}}" for m in args.metrics)
    print(header)
    print("-" * len(header))
    for sys_name in args.systems:
        key = f"locomo10/{sys_name}"
        metrics = results.get(key, {})
        row = f"{sys_name:<30}" + "".join(f"{metrics.get(m, 0.0):>{col_w}.4f}" for m in args.metrics)
        print(row)

    print(f"\nResults saved to: {args.results}")


if __name__ == "__main__":
    main()
