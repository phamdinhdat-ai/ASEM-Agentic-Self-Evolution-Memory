#!/usr/bin/env python
"""Build the static (ingest-once) memory banks for a LoCoMo-style dataset.

Phase A of the static-bank workflow: every ingestion-based system gets **one
frozen memory bank per conversation**, written under a repo-root ``static/``
tree that any evaluation run can then reuse:

    static/memory_banks/<dataset>/<tag>/<System>/<locomo_000N>/<bank>.sqlite
    static/memory_banks/<dataset>/<tag>/manifest.json
    static/memory_banks/<dataset>/<tag>/banks.json

Once a bank exists here it is never rebuilt unless ``--force`` is given, and the
evaluation phase copies (never writes) it — so one ingestion result can back any
number of retrieval backbones and model-size sweeps.

Safety / resumability
---------------------
The build loop lives in :mod:`eval.static_banks`; this file is only the CLI.
Its contract:

* **Idempotent**: a ``(system, conversation)`` pair whose bank is non-empty is
  skipped. Re-running the identical command resumes instead of restarting.
* **Error-isolated**: a failure while building one bank is logged with its
  traceback, recorded in the manifest as ``status: "error"``, and the run
  continues with the next system/conversation (opt out with ``--fail-fast``).
* **Crash-safe**: ``manifest.json`` and ``banks.json`` are rewritten atomically
  after *every* system, so Ctrl-C (or a crash) always leaves a valid, resumable
  artifact. SIGINT finishes the in-flight system before exiting.

Exit codes: ``0`` ok, ``2`` finished but some banks recorded an error,
``130`` interrupted (partial state saved), ``1`` fatal error.

Examples
--------
::

    # Build every banked system for all 10 LoCoMo conversations
    python scripts/build_static_banks.py --tag gpt54 \
        --ingest-config configs/models/gpt54_api.yaml

    # Fast subset first (SimRetrieval + FastASEM), then add the rest later
    python scripts/build_static_banks.py --tag gpt54 --systems SimRetrieval FastASEM

    # Rebuild one system from scratch after changing its config
    python scripts/build_static_banks.py --tag gpt54 --systems ASEMv2 --force
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
from typing import Any, Dict, List

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


DEFAULT_INGEST_CONFIG = "configs/models/deepseek_api.yaml"

#: Set by the SIGINT handler; checked between systems for a graceful stop.
_STOP_REQUESTED = False


def _install_sigint_handler() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        global _STOP_REQUESTED
        if _STOP_REQUESTED:
            print("\nSecond interrupt — exiting immediately.", flush=True)
            raise SystemExit(130)
        _STOP_REQUESTED = True
        print(
            "\n[interrupt] Finishing the in-flight system, then saving and exiting. "
            "Press Ctrl-C again to abort now.",
            flush=True,
        )

    try:
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):  # not on the main thread
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    from asem.logging_utils import setup_logging
    from eval.phase_runner import (
        ALL_SYSTEMS,
        BANK_FILE_NAMES,
        build_backend_from_config,
        conversation_index,
        load_raw_dataset,
    )
    from eval.static_banks import (
        build_static_banks,
        embedder_name_from_config,
        model_name_from_config,
    )
    from scripts.run_locomo10_experiments import (
        convert_locomo10_to_eval,
        group_by_conversation,
    )

    parser = argparse.ArgumentParser(
        description="Build static (ingest-once) memory banks for a LoCoMo dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="datasets/locomo/locomo10.json",
                        help="Path to locomo10.json")
    parser.add_argument("--dataset-name", default=None,
                        help="Dataset folder name under static/memory_banks "
                             "(default: input file stem, e.g. 'locomo10')")
    parser.add_argument("--tag", default="shared",
                        help="Experiment/bank tag — the directory reused by evaluation")
    parser.add_argument("--ingest-config", default=DEFAULT_INGEST_CONFIG,
                        help="Backbone config used to build the banks")
    parser.add_argument("--systems", nargs="+", default=None,
                        help="Systems to build (default: every banked system: "
                             f"{[s for s in ALL_SYSTEMS if s in BANK_FILE_NAMES]})")
    parser.add_argument("--static-root", default="static/memory_banks",
                        help="Root of the frozen bank tree")
    parser.add_argument("--conversations", nargs="+", type=int, default=None,
                        help="Only build these conversation indices (0-based)")
    parser.add_argument("--limit-conversations", type=int, default=None,
                        help="Only build the first N conversations")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild banks that already exist and are non-empty")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Abort on the first bank that fails (default: record and continue)")
    parser.add_argument("--allow-empty", action="store_true",
                        help="Accept a bank with 0 notes as success (default: record it as an error)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    _install_sigint_handler()

    if not os.path.exists(args.input):
        raise SystemExit(f"Dataset not found: {args.input}")
    if not os.path.exists(args.ingest_config):
        raise SystemExit(f"Ingest config not found: {args.ingest_config}")

    dataset = args.dataset_name or os.path.splitext(os.path.basename(args.input))[0]
    bank_root = os.path.join(args.static_root, dataset)

    all_banked = [s for s in ALL_SYSTEMS if s in BANK_FILE_NAMES]
    systems: List[str] = list(args.systems or all_banked)
    unknown = [s for s in systems if s not in all_banked]
    if unknown:
        raise SystemExit(
            f"Unknown/non-banked systems {unknown}. Available: {all_banked}"
        )

    # ---- Resolve the conversation list -------------------------------
    print(f"Loading {args.input} ...")
    raw_dataset = load_raw_dataset(args.input)
    # limit=None keeps the canonical locomo_000N ids identical to what the
    # retrieval phase will look for; ingestion always covers whole conversations.
    eval_items = convert_locomo10_to_eval(args.input, limit=None)
    groups = group_by_conversation(eval_items)

    selected: List[List[Dict[str, Any]]] = []
    for group in groups:
        idx = conversation_index(str(group[0].get("session_id", "")))
        if args.conversations is not None and idx not in set(args.conversations):
            continue
        selected.append(group)
    if args.limit_conversations is not None:
        selected = selected[: args.limit_conversations]

    if not selected:
        raise SystemExit("No conversations selected — check --conversations/--limit-conversations")

    total_qa = sum(len(g) for g in selected)
    print(f"  {len(groups)} conversations in the dataset | {len(selected)} selected | {total_qa} QA pairs")
    print(f"  dataset    : {dataset}")
    print(f"  bank root  : {os.path.join(bank_root, args.tag)}")
    print(f"  systems    : {', '.join(systems)}")
    print(f"  backend   : {model_name_from_config(args.ingest_config)}  ({args.ingest_config})")
    print(f"  embedder  : {embedder_name_from_config(args.ingest_config)}")
    print(f"  force     : {args.force}")

    print("\nLoading inference backend ...")
    backend = build_backend_from_config(args.ingest_config)

    manifest = build_static_banks(
        raw_dataset=raw_dataset,
        groups=selected,
        systems=systems,
        ingest_config=args.ingest_config,
        bank_root=bank_root,
        tag=args.tag,
        dataset=dataset,
        backend=backend,
        force=args.force,
        allow_empty=args.allow_empty,
        fail_fast=args.fail_fast,
        should_stop=lambda: _STOP_REQUESTED,
        input_path=args.input,
    )

    run = manifest.get("run", {})
    totals = manifest.get("totals", {})

    print(f"\nManifest   -> {os.path.join(bank_root, args.tag, 'manifest.json')}")
    print(f"Bank index -> {os.path.join(bank_root, args.tag, 'banks.json')}")
    print("\n" + "=" * 72)
    print(f"Static bank build — tag={args.tag}  dataset={dataset}")
    print("=" * 72)
    print(f"  built this run   : {run.get('built', 0)}")
    print(f"  skipped existing : {run.get('skipped', 0)}")
    print(f"  re-ingested      : {run.get('discarded_partial', 0)}  (partial banks from an interrupted run)")
    print(f"  errors           : {run.get('errors', 0)}")
    print(f"  elapsed          : {manifest.get('elapsed_sec', 0)}s")
    print(f"  banks on disk    : ok={totals.get('banks_ok', 0)} "
          f"skipped={totals.get('banks_skipped', 0)} error={totals.get('banks_error', 0)}")
    for name, n in sorted((totals.get("notes") or {}).items()):
        print(f"    {name:<20} {n} notes")
    if run.get("errors"):
        print(f"\n  {run['errors']} bank(s) failed — see the manifest for tracebacks, "
              "then re-run the same command to retry only those.")

    if _STOP_REQUESTED:
        return 130
    return 2 if run.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
