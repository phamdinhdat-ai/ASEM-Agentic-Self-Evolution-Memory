"""Run memory baselines (FullContext / SimRetrieval / ASEMv2) on the SAME
question set that FastASEM was evaluated on, so all methods are directly
comparable on identical questions + gold answers.

The canonical test set is read from FastASEM's prediction JSONL
(``data/benchmarks/results/preds/locomo10_FastASEM_conv0_qa.jsonl``), which
contains 117 (query, ref, category) pairs for conv-26.

Usage (activate memory-r1 first):
    python scripts/run_baseline_comparison.py --systems FullContext SimRetrieval
    python scripts/run_baseline_comparison.py --systems ASEMv2
    python scripts/run_baseline_comparison.py --systems FullContext SimRetrieval ASEMv2 --judge

Each system's predictions are streamed to its own JSONL under --preds-dir so
partial results survive a crash. A final comparison table (including FastASEM)
is printed at the end.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_dotenv = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_dotenv):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv, override=False)
    except ImportError:
        with open(_dotenv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from eval.benchmark_runner import (
    CATEGORY_NAMES,
    compute_em,
    compute_rouge_l,
    evaluate_with_judge,
    extract_sessions_from_conv,
)
from eval.systems import build_asem_v2_system, build_baselines
from asem.backends import build_backend
from asem.config import ASEMConfig


def load_test_set(preds_path: str) -> List[Dict[str, Any]]:
    """Load the canonical (query, ref, category) test set from FastASEM preds."""
    items = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            items.append({
                "query": r["query"],
                "ref": r["ref"],
                "category": r.get("category", 1),
                "category_name": r.get("category_name", ""),
            })
    return items


def build_system(name: str, config_path: str, db_dir: str):
    if name == "ASEMv2":
        return build_asem_v2_system(config_path, db_dir)
    baselines = build_baselines(config_path, db_dir, max_history_turns=0)
    if name not in baselines:
        raise ValueError(f"Unknown system: {name} (choose FullContext, SimRetrieval, ASEMv2)")
    return baselines[name]


_TRANSIENT_MARKERS = (
    "getaddrinfo", "ConnectError", "Connection error", "APIConnectionError",
    "timeout", "Timeout", "timed out", "503", "502", "504", "ECONNRESET",
    "RemoteDisconnected", "Connection aborted",
)


def answer_with_retry(sys_instance, name: str, question: str, all_turns,
                      max_retries: int = 5):
    """Call a system's answer() with retry on transient network errors.

    The LLM backend's generate() has no built-in retry, so a DNS blip or
    transient 5xx would otherwise abort the whole run.
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            if name in ("NoMemory", "FullContext", "SimRetrieval", "AtomicLinking",
                       "RLManagerOnly", "ValueRetrievalOnly"):
                return sys_instance.answer(question, all_turns)
            return sys_instance.answer(question)
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_exc = e
            msg = str(e)
            transient = any(m in msg for m in _TRANSIENT_MARKERS)
            if not transient or attempt == max_retries:
                raise
            wait = 2 ** attempt
            print(f"    [retry {attempt}/{max_retries - 1}] transient "
                  f"{type(e).__name__} — retrying in {wait}s")
            time.sleep(wait)
    raise last_exc  # pragma: no cover - unreachable


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline comparison on FastASEM's test set")
    parser.add_argument("--data", default="datasets/locomo/locomo10.json")
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml")
    parser.add_argument("--conv-index", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=["FullContext", "SimRetrieval"])
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks")
    parser.add_argument("--preds-dir", default="data/benchmarks/results/preds")
    parser.add_argument("--fastasem-preds",
                        default="data/benchmarks/results/preds/locomo10_FastASEM_conv0_qa.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA pairs")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--reuse-bank", action="store_true",
                        help="Skip reset+ingestion for bank-based systems (SimRetrieval/ASEMv2) "
                             "when the bank is already built. Saves LLM ingestion cost.")
    args = parser.parse_args()

    # Load conversation (for history / ingestion)
    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conversations = raw if isinstance(raw, list) else [raw]
    conv = conversations[args.conv_index]
    conv_id = conv.get("conversation_id") or conv.get("sample_id") or f"conv_{args.conv_index}"
    conv_data = conv.get("conversation", conv)
    sessions = extract_sessions_from_conv(conv_data)
    all_turns = [t for s in sessions for t in s["turns"]]
    session_batches = [(s["session_id"] + " " + s["date"], s["turns"]) for s in sessions]

    # Canonical test set (identical questions + gold for every method)
    test_set = load_test_set(args.fastasem_preds)
    if args.limit:
        test_set = test_set[: args.limit]

    print("=" * 70)
    print(f"BASELINE COMPARISON | conv={conv_id} | {len(sessions)} sessions | "
          f"{len(all_turns)} turns | {len(test_set)} QA pairs")
    print(f"Systems: {', '.join(args.systems)} | Judge: {args.judge}")
    print("=" * 70)

    backend = build_backend(ASEMConfig.load(args.config).inference)
    os.makedirs(args.preds_dir, exist_ok=True)

    # Build + ingest each system ONCE
    systems: Dict[str, Any] = {}
    for name in args.systems:
        print(f"\n--- Building + ingesting: {name} ---")
        sys_instance = build_system(name, args.config, args.db_dir)
        if args.reuse_bank and name in ("SimRetrieval", "ASEMv2"):
            print(f"  (reusing existing bank — skipping reset+ingestion)")
            if name == "SimRetrieval" and hasattr(sys_instance, "_seen_hashes"):
                # Repopulate the in-memory dedup set so answer() does NOT fall
                # back to per-question re-ingestion. Must mirror ingest_batch's
                # exact enrichment: f"[{session_label}] {turn}".
                for label, turns in session_batches:
                    for t in turns:
                        sys_instance._seen_hashes.add(hash(f"[{label}] {t}"))
                print(f"    primed {len(sys_instance._seen_hashes)} seen-hashes "
                      f"from {len(session_batches)} sessions")
            systems[name] = sys_instance
            continue
        if hasattr(sys_instance, "reset"):
            sys_instance.reset()  # clean bank so re-runs don't accumulate duplicates
        t0 = time.time()
        if name == "ASEMv2":
            n_notes = sys_instance.ingest_conversation(all_turns)
            print(f"  Ingested {n_notes} notes in {time.time() - t0:.1f}s")
        elif name == "SimRetrieval":
            sys_instance.ingest_conversation(session_batches)
            print(f"  Ingested in {time.time() - t0:.1f}s")
        else:  # FullContext needs no ingestion
            print("  (no ingestion — full context per query)")
        systems[name] = sys_instance

    # Answer all questions for each system
    results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in args.systems}
    preds_files = {
        name: open(os.path.join(args.preds_dir, f"locomo10_{name}_conv26.jsonl"), "w",
                   encoding="utf-8")
        for name in args.systems
    }

    for idx, item in enumerate(test_set):
        question, gold = item["query"], item["ref"]
        cat_id = item["category"]
        cat_name = item["category_name"] or CATEGORY_NAMES.get(cat_id, f"Category {cat_id}")

        for name in args.systems:
            sys_instance = systems[name]
            pred = answer_with_retry(sys_instance, name, question, all_turns)

            em = compute_em(pred, gold)
            rouge = compute_rouge_l(pred, gold)
            judge_ok = None
            if args.judge:
                judge_ok, _ = evaluate_with_judge(backend, question, gold, pred)

            entry = {
                "idx": idx,
                "session_id": conv_id,
                "category": cat_id,
                "category_name": cat_name,
                "query": question,
                "pred": pred,
                "ref": gold,
                "em": em,
                "rouge_l": rouge,
                "judge_correct": judge_ok,
            }
            results[name].append(entry)
            preds_files[name].write(json.dumps(entry, ensure_ascii=False) + "\n")
            preds_files[name].flush()

        print(f"[{idx + 1}/{len(test_set)}] {cat_name} | "
              + " | ".join(f"{n}: EM={results[n][-1]['em']:.0f}" for n in args.systems)
              + f" | Q: {question[:55]}")

    for fh in preds_files.values():
        fh.close()

    # ---- Comparison table (including FastASEM from its existing preds) ----
    print("\n" + "=" * 70)
    print("COMPARISON (identical %d questions, same metric functions)" % len(test_set))
    print("=" * 70)

    def summarize(entries):
        n = len(entries)
        em = sum(e["em"] for e in entries) / n
        rl = sum(e["rouge_l"] for e in entries) / n
        jd = (sum(1 for e in entries if e["judge_correct"]) / n) if args.judge else None
        return em, rl, jd

    rows = []
    # FastASEM (existing)
    fa_entries = []
    with open(args.fastasem_preds, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                fa_entries.append(json.loads(line))
    rows.append(("FastASEM", *summarize(fa_entries)))
    for name in args.systems:
        rows.append((name, *summarize(results[name])))

    hdr = f"{'Method':<14} {'EM':>7} {'ROUGE-L':>9}" + (" {'Judge':>7}" if args.judge else "")
    print(hdr)
    print("-" * len(hdr))
    for name, em, rl, jd in rows:
        line = f"{name:<14} {em:>7.3f} {rl:>9.3f}"
        if args.judge:
            line += f" {jd:>7.3f}"
        print(line)

    # Per-category for the newly-run systems
    print("\nPer-category (newly-run systems):")
    for name in args.systems:
        cats = defaultdict(list)
        for e in results[name]:
            cats[e["category_name"]].append(e)
        print(f"  {name}:")
        for c in sorted(cats):
            v = cats[c]
            print(f"    {c:<28} n={len(v):>3} EM={sum(x['em'] for x in v)/len(v):.3f} "
                  f"ROUGE={sum(x['rouge_l'] for x in v)/len(v):.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
