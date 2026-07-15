#!/usr/bin/env python
"""Run ASEM v2 two-phase benchmark on locomo10.json.

Two independent phases that can run separately:

  Phase 1 (ingest):  Build knowledge graph session-by-session → save SQLite.
  Phase 2 (evaluate): Load pre-built graph → answer QA → compute metrics.

Each session is ~20-30 turns (~2,000-4,000 chars) — small enough for one
LLM call to extract facts reliably.

Usage::

    # ---- Combined mode (default) ----
    python scripts/run_asem_v2.py --limit 10

    # ---- Phase 1 only: ingest sessions, save graph ----
    python scripts/run_asem_v2.py --phase ingest --db data/banks/locomo10.sqlite

    # ---- Phase 2 only: load graph, evaluate ----
    python scripts/run_asem_v2.py --phase evaluate --db data/banks/locomo10.sqlite

    # ---- Compare v1 vs v2 ----
    python scripts/run_asem_v2.py --systems ASEM ASEMv2 --metrics em rougeL
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple

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

from scripts.run_locomo10_experiments import (
    convert_locomo10_to_eval,
    group_by_conversation,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Per-session dialogue extraction
# ---------------------------------------------------------------------------

def _parse_sessions(conversation: Dict[str, Any]) -> List[Tuple[int, str, List[str]]]:
    """Extract dialogue turns grouped by session number.

    Returns:
        List of ``(session_num, date_str, [turn_texts])`` sorted by session.
    """
    import re

    # Discover session numbers
    session_nums: Set[int] = set()
    for key in conversation:
        m = re.match(r"session_(\d+)$", key)
        if m:
            session_nums.add(int(m.group(1)))

    sessions: List[Tuple[int, str, List[str]]] = []
    for sess_num in sorted(session_nums):
        turns_key = f"session_{sess_num}"
        date_key = f"session_{sess_num}_date_time"
        date_str = conversation.get(date_key, "")

        turns_data = conversation.get(turns_key, [])
        if not isinstance(turns_data, list):
            continue

        turns: List[str] = []
        for turn in turns_data:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            blip = turn.get("blip_caption", "")
            content = f"[{speaker}] {text}"
            if blip:
                content += f" (image: {blip})"
            turns.append(content)

        if turns:
            sessions.append((sess_num, date_str, turns))

    return sessions


# ---------------------------------------------------------------------------
# Phase 1: Ingest sessions → build knowledge graph
# ---------------------------------------------------------------------------

def run_ingest_phase(
    raw_dataset: List[Dict[str, Any]],
    db_path: str,
    config_path: str,
) -> Dict[int, int]:
    """Ingest all conversations session-by-session into a persistent SQLite bank.

    Returns:
        Dict mapping conversation index → number of notes created.
    """
    from eval.systems import build_asem_v2_system

    # Build one ASEMv2 system — its memory bank will be saved to db_path
    db_dir = os.path.dirname(db_path)
    os.makedirs(db_dir, exist_ok=True)

    # The builder creates a bank at {db_dir}/asem_v2.sqlite — we'll save to db_path
    runner = build_asem_v2_system(config_path, db_dir)

    stats: Dict[int, int] = {}
    total_sessions = 0
    total_turns = 0
    total_notes = 0
    t0 = time.perf_counter()

    for conv_idx, record in enumerate(raw_dataset):
        conversation = record.get("conversation", {})
        sessions = _parse_sessions(conversation)
        speaker_a = conversation.get("speaker_a", "A")
        speaker_b = conversation.get("speaker_b", "B")

        if not sessions:
            continue

        print(f"\n{'='*60}")
        print(f"Conversation {conv_idx}: {speaker_a} & {speaker_b}")
        print(f"  {len(sessions)} sessions, "
              f"{sum(len(t[2]) for t in sessions)} total turns")
        print(f"{'='*60}")

        conv_notes = 0
        for sess_num, date_str, turns in sessions:
            # Add session header for context
            header = f"[Session {sess_num}"
            if date_str:
                header += f" — {date_str}"
            header += "]"
            dialogue = [header] + turns

            bank_before = runner.pipeline.memory_bank.size()
            n = runner.ingest_conversation(dialogue)
            conv_notes += n
            total_notes += n
            total_turns += len(turns)
            total_sessions += 1

            bank_after = runner.pipeline.memory_bank.size()
            new_links = bank_after - bank_before  # rough estimate
            print(f"  session_{sess_num:02d} | {len(turns):>3d} turns, "
                  f"{len(dialogue[0]) + sum(len(t) for t in turns):,} chars "
                  f"-> {n:>2d} notes, bank: {bank_before} -> {bank_after}"
                  f"{'  [' + date_str + ']' if date_str else ''}")

        stats[conv_idx] = conv_notes
        print(f"  TOTAL: {conv_notes} notes across {len(sessions)} sessions")

    elapsed = time.perf_counter() - t0
    print(f"\nPhase 1 complete | convs={len(stats)}  sessions={total_sessions}  "
          f"turns={total_turns}  notes={total_notes}  time={elapsed:.1f}s")

    # Save the bank to the requested path
    runner.pipeline.memory_bank.save(db_path)
    print(f"Knowledge graph saved → {db_path}")

    return stats


# ---------------------------------------------------------------------------
# Phase 2: Evaluate from pre-built graph
# ---------------------------------------------------------------------------

def run_evaluate_phase(
    db_path: str,
    config_path: str,
    conversation_groups: List[List[Dict[str, Any]]],
    metric_names: List[str],
    preds_dir: str,
    sys_name: str,
    results: Dict[str, Any],
    results_path: str,
) -> Tuple[List[str], List[str]]:
    """Evaluate QA pairs using a pre-built knowledge graph (NO re-ingestion).

    Loads the SQLite bank from ``db_path`` and answers all QA pairs from
    the pre-built knowledge without re-processing any dialogue turns.
    """
    from asem.backends import build_backend
    from asem.enhanced_retriever import EnhancedHybridRetriever
    from asem.answer_agent import AnswerAgent
    from asem.memory_bank import MemoryBank
    import yaml

    preds_path = os.path.join(preds_dir, f"locomo10_v2_{sys_name}.jsonl")
    done_ids: Set[int] = set()
    preds_so_far: List[str] = []
    refs_so_far: List[str] = []

    # Resume
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
            print(f"  Resuming from {len(done_ids)} saved predictions")

    def _flush(res):
        tmp = results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        os.replace(tmp, results_path)

    # Load config + build retrieval-only pipeline
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    def _load_txt(p):
        with open(p, "r", encoding="utf-8") as fh:
            return fh.read()

    retriever = EnhancedHybridRetriever(
        backend=backend,
        k1=hp["k1"], k2=hp["k2"],
        delta=hp["delta"], lambda_weight=hp["lambda"],
        max_hops=2, hop_decay=0.7, multi_hop_topn=5,
        alpha=0.35, beta=0.25, gamma=0.40,
        enable_global_semantics=True,
        enable_intent_q=True,
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template=_load_txt("data/prompts/P3_memory_evolution.txt"),  # reused as distil
        baseline_prompt_template=(
            "Answer using the memory notes below. Reply with ONLY the answer.\n"
            "Memory:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        ),
    )

    # Load the pre-built bank
    memory_bank = MemoryBank.load(db_path)
    print(f"  Loaded knowledge graph | {memory_bank.size()} notes  "
          f"from {db_path}")

    total_items = sum(len(g) for g in conversation_groups)

    with open(preds_path, "a", encoding="utf-8") as fp:
        for group in conversation_groups:
            for item in group:
                idx = item.get("_idx", -1)
                if idx in done_ids:
                    continue

                query = str(item.get("query", ""))
                ref = str(item.get("answer", ""))

                try:
                    candidates = retriever.retrieve(query, memory_bank)
                    _, answer = answer_agent.distil_and_answer(query, candidates)
                except Exception as exc:
                    print(f"\n  ERROR on example {idx}: {exc}")
                    traceback.print_exc()
                    answer = ""

                preds_so_far.append(answer)
                refs_so_far.append(ref)

                fp.write(json.dumps({
                    "idx": idx,
                    "session_id": item.get("session_id", ""),
                    "category": item.get("category", 0),
                    "category_name": item.get("category_name", ""),
                    "query": query,
                    "pred": answer,
                    "ref": ref,
                }) + "\n")
                fp.flush()

                n_done = len(done_ids) + len(preds_so_far)
                if n_done % 20 == 0:
                    pct = n_done / total_items * 100
                    print(f"  [{sys_name}] {n_done}/{total_items} ({pct:.0f}%)  "
                          f"latest: {answer[:80]!r}")

                if n_done % 50 == 0:
                    partial_key = f"locomo10/{sys_name}/__partial__"
                    partial_metrics = compute_metrics(preds_so_far, refs_so_far, metric_names)
                    partial_metrics["__n__"] = n_done
                    results[partial_key] = partial_metrics
                    _flush(results)

    return preds_so_far, refs_so_far


# ---------------------------------------------------------------------------
# Combined mode (ingest + evaluate in one run)
# ---------------------------------------------------------------------------

def run_combined(
    args: argparse.Namespace,
    raw_dataset: List[Dict[str, Any]],
    eval_data: List[Dict[str, Any]],
    conversation_groups: List[List[Dict[str, Any]]],
) -> None:
    """Ingest once, then evaluate all systems."""
    db_dir = os.path.join(args.db_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(db_dir, exist_ok=True)
    preds_dir = os.path.join(os.path.dirname(args.results), "preds")
    os.makedirs(preds_dir, exist_ok=True)

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

    total_items = sum(len(g) for g in conversation_groups)

    for sys_name in args.systems:
        key = f"locomo10/{sys_name}"
        if key in results:
            print(f"\n  [{sys_name}] already completed -- skipping")
            continue

        print(f"\n  [{sys_name}] building system ...")

        if sys_name == "ASEMv2":
            # ---- ASEMv2 two-phase path ----
            bank_path = os.path.join(db_dir, "asem_v2.sqlite")
            print(f"  [Phase 1] Ingesting sessions → {bank_path}")
            run_ingest_phase(raw_dataset, bank_path, args.config)

            print(f"  [Phase 2] Evaluating {total_items} QA pairs ...")
            preds, refs = run_evaluate_phase(
                db_path=bank_path,
                config_path=args.config,
                conversation_groups=conversation_groups,
                metric_names=args.metrics,
                preds_dir=preds_dir,
                sys_name=sys_name,
                results=results,
                results_path=args.results,
            )
        else:
            # ---- v1 / baseline path ----
            from eval.systems import build_asem_system, build_baselines

            if sys_name == "ASEM":
                runner = build_asem_system(args.config, db_dir)
            else:
                baselines = build_baselines(args.config, db_dir)
                runner = baselines[sys_name]

            print(f"  [{sys_name}] running on {total_items} examples ...")
            preds_path = os.path.join(preds_dir, f"locomo10_v2_{sys_name}.jsonl")
            preds, refs = [], []

            with open(preds_path, "w", encoding="utf-8") as fp:
                for group in conversation_groups:
                    if hasattr(runner, 'reset'):
                        runner.reset()
                    for item in group:
                        query = str(item.get("query", ""))
                        ref = str(item.get("answer", ""))
                        history = [str(h) for h in item.get("history", [])]
                        try:
                            pred = runner.answer(query, history)
                        except Exception:
                            pred = ""
                        preds.append(pred)
                        refs.append(ref)
                        fp.write(json.dumps({
                            "idx": item.get("_idx", -1),
                            "query": query, "pred": pred, "ref": ref,
                        }) + "\n")
                        fp.flush()
                        if len(preds) % 20 == 0:
                            print(f"  [{sys_name}] {len(preds)}/{total_items} "
                                  f"({len(preds)/total_items*100:.0f}%)")

        final_metrics = compute_metrics(preds, refs, args.metrics)
        results[key] = final_metrics
        results.pop(f"locomo10/{sys_name}/__partial__", None)
        _flush(results)
        print(f"  [{sys_name}] FINAL: {final_metrics}")

    # ---- Summary table ----
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASEM v2 — Two-phase benchmark (per-session ingestion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default="datasets/locomo/locomo10.json")

    # Phase selection
    parser.add_argument(
        "--phase",
        choices=["ingest", "evaluate", "combined"],
        default="combined",
        help="ingest: build graph only. evaluate: answer QA from saved graph. "
             "combined: both (default).",
    )

    # Shared
    parser.add_argument("--config", default="configs/locomo_openai.yaml")
    parser.add_argument("--db", default=None, help="Path to SQLite bank (for --phase ingest/evaluate)")

    # Combined / evaluate
    parser.add_argument("--results", default="data/benchmarks/results/locomo10_v2.json")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks_v2")
    parser.add_argument(
        "--systems", nargs="+", default=["ASEMv2"],
        help="Systems: NoMemory FullContext SimRetrieval AtomicLinking "
             "RLManagerOnly ValueRetrievalOnly ASEM ASEMv2",
    )
    parser.add_argument("--metrics", nargs="+", default=["em", "rougeL"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-category", action="store_true")
    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()

    # ---- Load data ----
    print("Loading locomo10.json ...")
    raw_dataset = json.load(open(args.input, "r", encoding="utf-8"))
    eval_data = convert_locomo10_to_eval(args.input, limit=args.limit)
    conversation_groups = group_by_conversation(eval_data)
    total_items = sum(len(g) for g in conversation_groups)
    print(f"  {len(conversation_groups)} conversations, {total_items} QA pairs")

    # ---- Dispatch by phase ----
    if args.phase == "ingest":
        db_path = args.db or "data/banks/locomo10_v2.sqlite"
        run_ingest_phase(raw_dataset, db_path, args.config)

    elif args.phase == "evaluate":
        db_path = args.db
        if not db_path or not os.path.exists(db_path):
            print(f"Error: bank not found: {db_path}. Run --phase ingest first.")
            sys.exit(1)

        os.makedirs(os.path.dirname(args.results), exist_ok=True)
        preds_dir = os.path.join(os.path.dirname(args.results), "preds")
        os.makedirs(preds_dir, exist_ok=True)

        results: Dict[str, Any] = {}
        if os.path.exists(args.results):
            with open(args.results, "r", encoding="utf-8") as fh:
                try:
                    results = json.load(fh)
                except json.JSONDecodeError:
                    results = {}

        for sys_name in args.systems:
            key = f"locomo10/{sys_name}"
            if key in results:
                print(f"  [{sys_name}] already completed -- skipping")
                continue
            print(f"  [{sys_name}] evaluating from {db_path} ...")
            preds, refs = run_evaluate_phase(
                db_path=db_path,
                config_path=args.config,
                conversation_groups=conversation_groups,
                metric_names=args.metrics,
                preds_dir=preds_dir,
                sys_name=sys_name,
                results=results,
                results_path=args.results,
            )
            final_metrics = compute_metrics(preds, refs, args.metrics)
            results[key] = final_metrics
            results.pop(f"locomo10/{sys_name}/__partial__", None)
            print(f"  [{sys_name}] FINAL: {final_metrics}")

        # Save results
        def _flush(res):
            tmp = args.results + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2)
            os.replace(tmp, args.results)
        _flush(results)
        print(f"Results saved to: {args.results}")

    else:
        # combined mode
        run_combined(args, raw_dataset, eval_data, conversation_groups)


if __name__ == "__main__":
    main()
