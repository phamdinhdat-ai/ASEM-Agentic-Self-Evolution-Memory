#!/usr/bin/env python
"""Evaluate systems against frozen (ingest-once) memory banks — full LoCoMo QA.

Phase B of the static-bank workflow. Reads the banks built by
``scripts/build_static_banks.py``, answers every QA pair, and scores each system
with EM / token-F1 / ROUGE-L / BERTScore-F1 / LLM-as-a-judge — always including
the ``FullContext`` and ``NoMemory`` baselines so the comparison is explicit.

Frozen banks are copied into a per-run working directory before being opened, so
systems that write while answering cannot contaminate the ingest-once artifact.

Every answered QA pair is flushed to ``preds/*.jsonl`` immediately and the
aggregate JSON is rewritten after every conversation, so Ctrl-C / a crash / an
API outage never loses completed work: re-running the identical command resumes.

Examples
--------
::

    # Ingest once, then evaluate everything (memory systems + baselines)
    python scripts/build_static_banks.py --tag gpt54 \
        --ingest-config configs/models/gpt54_api.yaml
    python scripts/run_static_eval.py --tag gpt54 \
        --config configs/models/gpt54_api.yaml \
        --judge-config configs/models/judge_api.yaml

    # Retrieval-only: swap the backbone, reuse the same frozen banks
    python scripts/run_static_eval.py --tag gpt54 --config configs/models/qwen3_4b_api.yaml \
        --systems FastASEM SimRetrieval FullContext

    # Quick, cheap sanity check
    python scripts/run_static_eval.py --tag gpt54 --limit 10 \
        --metrics em rougeL f1 --systems SimRetrieval FullContext NoMemory
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env (OPENAI_API_KEY / OPENAI_BASE_URL / LLM_MODEL) if present.
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


DEFAULT_CONFIG = "configs/models/deepseek_api.yaml"
DEFAULT_REGISTRY = "configs/models/registry.yaml"


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_retrieval_configs(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Return ``[{"tag": ..., "config": ...}]`` retrieval backbones to evaluate."""
    registry = _load_yaml(args.registry) if os.path.exists(args.registry) else {}
    models = registry.get("models") or {}
    groups = registry.get("groups") or {}

    tags: List[str] = []
    if args.models_group:
        if args.models_group not in groups:
            raise SystemExit(
                f"Unknown --models-group {args.models_group!r}. Available: {list(groups)}"
            )
        tags = list(groups[args.models_group])
    if args.models:
        tags = list(args.models)

    if tags:
        out: List[Dict[str, str]] = []
        for tag in tags:
            if tag not in models:
                raise SystemExit(f"Unknown model tag {tag!r}. Available: {list(models)}")
            out.append({"tag": tag, "config": str(models[tag]["config"])})
        return out

    if args.config:
        return [{"tag": "", "config": args.config}]
    return [{"tag": "", "config": DEFAULT_CONFIG}]


def _preflight(
    systems: List[str],
    groups: List[List[Dict[str, Any]]],
    bank_root: str,
    tag: str,
    allow_missing: bool,
) -> bool:
    """Print a bank-coverage table. Returns False when required banks are missing."""
    from eval.phase_runner import bank_exists, conversation_index

    banked = [s for s in systems if s not in ("NoMemory", "FullContext")]
    conv_ids = [str(g[0].get("session_id", "")) for g in groups]

    print("\nBank coverage:")
    ok = True
    for name in banked:
        present = [c for c in conv_ids if bank_exists(bank_root, tag, name, c)]
        missing = [c for c in conv_ids if c not in present]
        status = "OK" if not missing else "MISSING"
        print(f"  {name:<20} {len(present)}/{len(conv_ids)}  [{status}]"
              + (f"  missing: {', '.join(missing[:6])}" if missing else ""))
        if missing:
            ok = False
            if allow_missing and len(missing) == len(conv_ids):
                print(f"    WARNING: {name} has no banks at all under tag {tag!r} — "
                      "it will be skipped.")
    for name in [s for s in systems if s in ("NoMemory", "FullContext")]:
        print(f"  {name:<20} bankless (answered from item history)")

    if not ok and not allow_missing:
        print(
            f"\nERROR: missing banks under {os.path.join(bank_root, tag)}. "
            "Run scripts/build_static_banks.py first, or pass --allow-missing-banks."
        )
    return ok or allow_missing


def _resolve_conversations(requested: List[str], available: List[str]) -> List[str]:
    """Map user-supplied conversation selectors to canonical dataset ids.

    ``locomo_0003``, ``locomo_3`` and ``3`` all resolve to ``locomo_0003``.
    Unknown selectors abort the run: silently evaluating a different
    conversation than the one asked for would be worse than failing.
    """
    by_index: Dict[int, str] = {}
    for cid in available:
        m = re.search(r"(\d+)$", str(cid))
        if m:
            by_index[int(m.group(1))] = str(cid)

    out: List[str] = []
    unknown: List[str] = []
    for raw in requested:
        token = str(raw).strip()
        m = re.search(r"(\d+)$", token)
        resolved = token if token in available else by_index.get(
            int(m.group(1)) if m else -1, ""
        )
        if not resolved:
            unknown.append(token)
        elif resolved not in out:
            out.append(resolved)

    if unknown:
        raise SystemExit(
            f"Unknown --conversations {unknown}. "
            f"Available: {', '.join(available) or '(none)'}"
        )
    return out


def _fresh(paths: List[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"  removed {path}")
            except OSError as exc:
                print(f"  could not remove {path}: {exc}")


def main() -> int:
    from asem.logging_utils import get_logger, setup_logging
    from eval.phase_runner import (
        ALL_SYSTEMS,
        BANK_FILE_NAMES,
        build_backend_from_config,
        model_tag_from_config,
        render_sweep_table,
        write_results,
    )
    from eval.static_eval import (
        render_per_conversation_matrix,
        render_report_table,
        run_static_eval,
    )
    from scripts.run_locomo10_experiments import (
        convert_locomo10_to_eval,
        group_by_conversation,
    )

    parser = argparse.ArgumentParser(
        description="Evaluate systems against frozen static memory banks (full QA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="datasets/locomo/locomo10.json")
    parser.add_argument("--dataset-name", default=None,
                        help="Dataset folder name under static/memory_banks "
                             "(default: input file stem)")
    parser.add_argument("--tag", default="shared",
                        help="Bank tag written by build_static_banks.py")

    # Backbones
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Answer/retrieval backbone config")
    parser.add_argument("--judge-config", default=None,
                        help="Backbone used for LLM-as-a-judge (default: --config)")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Registry tags to sweep (reuses the same frozen banks)")
    parser.add_argument("--models-group", default=None,
                        help="Registry group to sweep (e.g. api_proxy, hf_small)")

    # Scope
    parser.add_argument("--systems", nargs="+", default=None,
                        help="Systems to evaluate (default: every banked system + "
                             "FullContext NoMemory)")
    parser.add_argument("--metrics", nargs="+",
                        default=["em", "f1", "rougeL", "bertscore_f1", "judge"],
                        help="Any of: em em_loose f1 rougeL bertscore_f1 judge")
    parser.add_argument("--limit", type=int, default=None,
                        help="Answer only the first N QA pairs (smoke tests)")
    parser.add_argument("--conversations", nargs="+", default=None,
                        help="Evaluate only these conversations (ids like locomo_0003, "
                             "or 0-based indices like 3). Applied before --limit, so "
                             "one conversation can be evaluated per invocation.")
    parser.add_argument("--per-category", dest="per_category", action="store_true",
                        default=True, help="Per-category breakdown (default: on)")
    parser.add_argument("--no-per-category", dest="per_category", action="store_false",
                        help="Skip the per-category breakdown")
    parser.add_argument("--per-conversation", dest="per_conversation", action="store_true",
                        default=True,
                        help="Per-conversation breakdown + matrices (default: on)")
    parser.add_argument("--no-per-conversation", dest="per_conversation",
                        action="store_false",
                        help="Skip the per-conversation breakdown")
    parser.add_argument("--full-context-dates", action="store_true",
                        help="Use the date-leveled FullContext prompt (fair vs FastASEM)")
    parser.add_argument("--full-context-max-turns", type=int, default=0,
                        help="Truncate the FullContext history to N turns (0 = no truncation)")
    parser.add_argument("--abort-after-consecutive-errors", type=int, default=8,
                        help="Abort a system after N consecutive answer failures (0 = never)")

    # Storage
    parser.add_argument("--static-root", default="static/memory_banks")
    parser.add_argument("--results-dir", default="data/benchmarks/results/static")
    parser.add_argument("--work-root", default="data/benchmarks/retrieval_work_static",
                        help="Per-run bank copies (keeps the frozen banks pristine)")
    parser.add_argument("--no-work-copy", action="store_true",
                        help="Answer against the static banks in place "
                             "(mutating systems WILL modify them)")
    parser.add_argument("--allow-missing-banks", action="store_true",
                        help="Skip (instead of failing) systems whose bank is absent")
    parser.add_argument("--fresh", action="store_true",
                        help="Delete saved predictions/scores for this tag+model first")
    parser.add_argument("--no-isolate-systems", action="store_true",
                        help="Evaluate all systems in one pass instead of one system "
                             "per run (isolation gives better fault tolerance)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    logger = get_logger(__name__)

    if not os.path.exists(args.input):
        raise SystemExit(f"Dataset not found: {args.input}")

    dataset = args.dataset_name or os.path.splitext(os.path.basename(args.input))[0]
    bank_root = os.path.join(args.static_root, dataset)

    all_banked = [s for s in ALL_SYSTEMS if s in BANK_FILE_NAMES]
    default_systems = all_banked + ["FullContext", "NoMemory"]
    systems: List[str] = list(args.systems or default_systems)
    unknown = [s for s in systems if s not in ALL_SYSTEMS]
    if unknown:
        raise SystemExit(f"Unknown systems {unknown}. Known: {ALL_SYSTEMS}")

    # ---- Data ----------------------------------------------------------
    print(f"Loading {args.input} ...")
    eval_items = convert_locomo10_to_eval(args.input, limit=args.limit)
    groups = group_by_conversation(eval_items)
    if args.conversations:
        args.conversations = _resolve_conversations(
            args.conversations, [str(g[0].get("session_id", "")) for g in groups]
        )
        groups = [
            g for g in groups if str(g[0].get("session_id", "")) in set(args.conversations)
        ]
    total_qa = sum(len(g) for g in groups)
    print(f"  {len(groups)} conversations | {total_qa} QA pairs (--limit={args.limit})")
    print(f"  conversations: {', '.join(str(g[0].get('session_id', '')) for g in groups)}")

    retrieval_cfgs = _resolve_retrieval_configs(args)
    out_root = os.path.join(args.results_dir, dataset)
    os.makedirs(out_root, exist_ok=True)

    print(f"  dataset     : {dataset}")
    print(f"  bank root   : {os.path.join(bank_root, args.tag)}")
    print(f"  systems     : {', '.join(systems)}")
    print(f"  metrics     : {', '.join(args.metrics)}")
    print(f"  results     : {out_root}")
    print(f"  work copy   : {not args.no_work_copy}")
    print(f"  isolated    : {not args.no_isolate_systems}")

    if not _preflight(systems, groups, bank_root, args.tag, args.allow_missing_banks):
        return 1

    # Build the judge backend once and share it across every model/system.
    judge_backend = None
    judge_config = args.judge_config
    if "judge" in [m.lower() for m in args.metrics]:
        jc = judge_config or retrieval_cfgs[0]["config"]
        print(f"\nBuilding judge backend from {jc} ...")
        judge_backend = build_backend_from_config(jc)
        judge_config = jc

    sweep: Dict[str, Dict[str, Any]] = {}
    exit_code = 0

    for entry in retrieval_cfgs:
        config_path = entry["config"]
        model_tag = model_tag_from_config(config_path)
        display_tag = entry["tag"] or model_tag

        print(f"\n{'=' * 72}")
        print(f"Model: {display_tag}   ({config_path})")
        print("=" * 72)

        preds_dir = os.path.join(out_root, "preds")
        scores_dir = os.path.join(out_root, "scores")
        logs_dir = os.path.join(out_root, "logs")
        out_path = os.path.join(out_root, f"{args.tag}__{display_tag}.json")
        md_path = os.path.join(out_root, f"{args.tag}__{display_tag}.md")
        log_path = os.path.join(logs_dir, f"{args.tag}__{display_tag}.log")

        if args.fresh:
            print("Removing saved predictions/scores (--fresh):")
            _fresh([
                os.path.join(preds_dir, f"{args.tag}__{display_tag}__{s}.jsonl")
                for s in systems
            ] + [
                os.path.join(scores_dir, f"{args.tag}__{display_tag}__{s}.jsonl")
                for s in systems
            ])

        print(f"Building answer backend from {config_path} ...")
        backend = build_backend_from_config(config_path)

        def _run(subset: List[str]) -> Optional[Dict[str, Any]]:
            return run_static_eval(
                groups=groups,
                systems=subset,
                config_path=config_path,
                bank_root=bank_root,
                tag=args.tag,
                dataset=dataset,
                backend=backend,
                metric_names=args.metrics,
                judge_config=judge_config,
                judge_backend=judge_backend,
                preds_dir=preds_dir,
                scores_dir=scores_dir,
                out_path=out_path,
                log_path=log_path,
                per_category=args.per_category,
                per_conversation=args.per_conversation,
                conversations=args.conversations,
                require_banks=not args.allow_missing_banks,
                work_root=None if args.no_work_copy else args.work_root,
                model_tag=display_tag,
                limit=args.limit,
                full_context_dates=args.full_context_dates,
                max_full_context_turns=args.full_context_max_turns,
                abort_after_consecutive_errors=args.abort_after_consecutive_errors,
            )

        t0 = time.perf_counter()

        if args.no_isolate_systems:
            try:
                results = _run(systems)
            except KeyboardInterrupt:
                print(f"\n[interrupt] Partial results are saved at {out_path}. "
                      f"Re-run the same command to resume.")
                return 130
            except Exception as exc:  # noqa: BLE001
                logger.opt(exception=exc).error("Evaluation run failed")
                print(f"\nERROR: {type(exc).__name__}: {exc}")
                return 1
        else:
            # One run per system: a fatal error in one system cannot discard the
            # others' already-saved predictions. Each system's aggregate is merged
            # into a single result file so the report covers the whole set.
            results = None
            merged_systems: Dict[str, Any] = {}
            for name in systems:
                print(f"\n--- System: {name} ---")
                try:
                    partial = _run([name])
                except KeyboardInterrupt:
                    print(f"\n[interrupt] Partial results are saved at {out_path}. "
                          f"Re-run the same command to resume.")
                    return 130
                except Exception as exc:  # noqa: BLE001
                    logger.opt(exception=exc).error("[{}] evaluation failed", name)
                    print(f"  ERROR [{name}]: {type(exc).__name__}: {exc} — continuing")
                    exit_code = 1
                    continue
                merged_systems.update(partial.get("systems") or {})
                results = dict(partial)
                results["systems"] = dict(merged_systems)
                entry_res = merged_systems.get(name, {})
                print(f"  [{name}] n={entry_res.get('n', 0)} "
                      f"{entry_res.get('overall', {})}")
                # Checkpoint the union after every system so an interrupt here
                # still leaves a report covering everything scored so far.
                write_results(out_path, results)
            if results is None:
                # Every system failed before producing anything.
                continue

        elapsed = time.perf_counter() - t0
        results["elapsed_sec"] = round(elapsed, 2)
        write_results(out_path, results)
        table = render_report_table(results, args.metrics)
        matrix = render_per_conversation_matrix(results, args.metrics)
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Static-bank evaluation — tag `{args.tag}`, model `{display_tag}`\n\n")
            fh.write(f"- dataset: `{dataset}`\n- bank root: `{bank_root}/{args.tag}`\n")
            fh.write(f"- config: `{config_path}`\n- judge: `{judge_config or config_path}`\n")
            fh.write(f"- metrics: `{', '.join(args.metrics)}`\n")
            fh.write(
                "- conversations: "
                + (f"{len(results.get('conversations') or [])} "
                   f"({', '.join(results.get('conversations') or [])})\n"
                   if results.get("conversations") else "all\n")
            )
            fh.write(f"- QA pairs: {results.get('n_qa', 0)} | elapsed: {elapsed:.1f}s\n\n")
            fh.write(table + "\n")
            if matrix:
                fh.write("\n## Per-conversation matrices\n")
                fh.write(matrix + "\n")

        print(f"\nResults -> {out_path}")
        print(f"Report  -> {md_path}")
        print("\n" + table)
        sweep[display_tag] = results

    # ---- Multi-model sweep summary ------------------------------------
    if len(sweep) > 1:
        sweep_path = os.path.join(out_root, f"{args.tag}__sweep.json")
        table_path = os.path.join(out_root, f"{args.tag}__sweep_table.md")
        write_results(sweep_path, sweep)
        table = render_sweep_table(sweep, args.metrics)
        with open(table_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Static-bank sweep — tag `{args.tag}`\n\n")
            fh.write(f"Bank root: `{bank_root}/{args.tag}`\n\n")
            fh.write(table + "\n")
        print(f"\nSweep JSON  -> {sweep_path}")
        print(f"Sweep table -> {table_path}")
        print("\n" + table)

    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[interrupt] Interrupted — re-run the same command to resume.")
        raise SystemExit(130)
