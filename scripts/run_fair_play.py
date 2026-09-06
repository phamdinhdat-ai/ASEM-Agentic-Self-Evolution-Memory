"""Fair-play benchmark: ASEMv2, SimRetrieval, FullContext on the SAME
conversation + SAME 117 questions, with all four metrics (EM, ROUGE-L,
BERTScore-F1, LLM-as-a-Judge).

FastASEM is NOT re-run — its existing predictions
(``data/benchmarks/results/preds/locomo10_FastASEM_conv0_qa.jsonl``) are
loaded and its metrics recomputed with the identical metric functions so all
four methods are directly comparable.

Fairness guarantees
-------------------
* Every method answers the exact same 117 (query, ref, category) tuples,
  loaded from FastASEM's canonical test set.
* Every method sees the exact same conversation (conv-26, all sessions/turns).
* Bank-based systems (SimRetrieval, ASEMv2) are reset + re-ingested from
  scratch on every run so no stale notes leak in.
* EM / ROUGE-L use the same functions as ``eval.benchmark_runner``.
* BERTScore-F1 (roberta-base, CPU) and the LLM judge are computed for every
  method with the same code path.

Usage (activate memory-r1 first):
    # Smoke test (3 questions, no judge) to verify the pipeline works:
    python scripts/run_fair_play.py --limit 3

    # Full fair-play run (all 117 questions, judge on):
    python scripts/run_fair_play.py --judge

    # Re-score BERTScore + write summary without re-running the LLM:
    python scripts/run_fair_play.py --score-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

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
    compute_bertscore_batch,
    compute_em,
    compute_rouge_l,
    evaluate_with_judge,
    extract_sessions_from_conv,
)
from eval.systems import build_asem_v2_system, build_baselines
from asem.backends import build_backend
from asem.config import ASEMConfig

SYSTEMS = ["ASEMv2", "SimRetrieval", "FullContext"]
FASTASEM_NAME = "FastASEM"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def load_fastasem_entries(preds_path: str) -> List[Dict[str, Any]]:
    """Load FastASEM's existing per-question entries (pred/ref/category)."""
    entries = []
    with open(preds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def build_system(name: str, config_path: str, db_dir: str):
    if name == "ASEMv2":
        return build_asem_v2_system(config_path, db_dir)
    baselines = build_baselines(config_path, db_dir, max_history_turns=0)
    if name not in baselines:
        raise ValueError(f"Unknown system: {name} (choose {', '.join(SYSTEMS)})")
    return baselines[name]


_TRANSIENT_MARKERS = (
    "getaddrinfo", "ConnectError", "Connection error", "APIConnectionError",
    "timeout", "Timeout", "timed out", "503", "502", "504", "ECONNRESET",
    "RemoteDisconnected", "Connection aborted",
)


def answer_with_retry(sys_instance, name: str, question: str, all_turns,
                      max_retries: int = 5) -> Optional[str]:
    """Call a system's answer() with retry on transient network errors.

    Returns the raw prediction (may be None/empty if the system produced
    nothing — that is recorded, not silently dropped).
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


def _norm_pred(pred: Optional[str]) -> str:
    """Coerce a prediction to a non-null string for metric safety."""
    if pred is None:
        return ""
    if not isinstance(pred, str):
        return str(pred)
    return pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_entries(entries: List[Dict[str, Any]], use_judge: bool) -> Dict[str, float]:
    """Compute EM / ROUGE-L / BERTScore-F1 / Judge over a list of entries."""
    n = len(entries)
    if n == 0:
        return {"n": 0}
    preds = [_norm_pred(e.get("pred")) for e in entries]
    refs = [str(e.get("ref")) for e in entries]

    em = sum(compute_em(p, r) for p, r in zip(preds, refs)) / n
    rl = sum(compute_rouge_l(p, r) for p, r in zip(preds, refs)) / n

    bs_scores = compute_bertscore_batch(preds, refs)
    bs = sum(bs_scores) / n if bs_scores else 0.0

    out: Dict[str, float] = {
        "n": n,
        "em": em,
        "rouge_l": rl,
        "bertscore_f1": bs,
        "null_preds": sum(1 for p in preds if not p.strip()),
    }
    if use_judge:
        judged = [e for e in entries if e.get("judge_correct") is not None]
        if judged:
            out["judge"] = sum(1 for e in judged if e["judge_correct"]) / len(judged)
        else:
            out["judge"] = None
    return out


def score_by_category(entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    cats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entries:
        cats[e.get("category_name") or f"cat{e.get('category', 0)}"].append(e)
    return {c: score_entries(v, use_judge=False) for c, v in sorted(cats.items())}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fair-play 4-method LoCoMo comparison")
    parser.add_argument("--data", default="datasets/locomo/locomo10.json")
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml")
    parser.add_argument("--conv-index", type=int, default=0)
    parser.add_argument("--systems", nargs="+", default=SYSTEMS)
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks")
    parser.add_argument("--preds-dir", default="data/benchmarks/results/preds")
    parser.add_argument("--results-dir", default="data/benchmarks/results")
    parser.add_argument("--fastasem-preds",
                        default="data/benchmarks/results/preds/locomo10_FastASEM_conv0_qa.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA pairs")
    parser.add_argument("--judge", action="store_true",
                        help="Run LLM-as-a-Judge for every prediction (costs 1 call each)")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip LLM runs; re-score existing fairplay_*.jsonl + FastASEM")
    args = parser.parse_args()

    os.makedirs(args.preds_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    # ---- Load conversation ------------------------------------------------
    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conversations = raw if isinstance(raw, list) else [raw]
    conv = conversations[args.conv_index]
    conv_id = conv.get("conversation_id") or conv.get("sample_id") or f"conv_{args.conv_index}"
    conv_data = conv.get("conversation", conv)
    sessions = extract_sessions_from_conv(conv_data)
    all_turns = [t for s in sessions for t in s["turns"]]
    session_batches = [(s["session_id"] + " " + s["date"], s["turns"]) for s in sessions]

    # ---- Canonical test set (identical for every method) ------------------
    test_set = load_test_set(args.fastasem_preds)
    if args.limit:
        test_set = test_set[: args.limit]

    print("=" * 72)
    print(f"FAIR-PLAY BENCHMARK | conv={conv_id} | {len(sessions)} sessions | "
          f"{len(all_turns)} turns | {len(test_set)} QA pairs")
    print(f"Systems to run: {', '.join(args.systems)}  (+ FastASEM from existing preds)")
    print(f"Judge: {args.judge} | Score-only: {args.score_only}")
    print("=" * 72)

    results: Dict[str, List[Dict[str, Any]]] = {}

    if not args.score_only:
        backend = build_backend(ASEMConfig.load(args.config).inference)

        # ---- Build + ingest each system ONCE (fresh bank) -----------------
        systems: Dict[str, Any] = {}
        for name in args.systems:
            print(f"\n--- Building + ingesting: {name} ---")
            sys_instance = build_system(name, args.config, args.db_dir)
            if hasattr(sys_instance, "reset"):
                sys_instance.reset()  # clean bank so re-runs don't accumulate duplicates
            t0 = time.time()
            if name == "ASEMv2":
                n_notes = sys_instance.ingest_conversation(all_turns)
                print(f"  Ingested {n_notes} notes in {time.time() - t0:.1f}s")
                if n_notes == 0:
                    print("  !! WARNING: ASEMv2 ingested 0 notes — answers will be empty.")
            elif name == "SimRetrieval":
                sys_instance.ingest_conversation(session_batches)
                print(f"  Ingested in {time.time() - t0:.1f}s")
            else:  # FullContext needs no ingestion
                print("  (no ingestion — full context per query)")
            systems[name] = sys_instance

        # ---- Answer all questions for each system -------------------------
        results = {name: [] for name in args.systems}
        preds_files = {
            name: open(os.path.join(args.preds_dir, f"fairplay_{name}_conv26.jsonl"), "w",
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
                pred_str = _norm_pred(pred)

                em = compute_em(pred_str, gold)
                rouge = compute_rouge_l(pred_str, gold)
                judge_ok = None
                if args.judge:
                    judge_ok, _ = evaluate_with_judge(backend, question, gold, pred_str)

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
    else:
        # ---- Score-only: load existing fairplay_*.jsonl -------------------
        for name in args.systems:
            path = os.path.join(args.preds_dir, f"fairplay_{name}_conv26.jsonl")
            if not os.path.exists(path):
                print(f"  [score-only] missing {path} — skipping {name}")
                continue
            entries = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
            results[name] = entries

    # ---- FastASEM (existing predictions, re-scored identically) -----------
    fa_entries = load_fastasem_entries(args.fastasem_preds)
    if args.limit:
        fa_entries = fa_entries[: args.limit]

    # ---- Summary ----------------------------------------------------------
    summary: Dict[str, Any] = {
        "metadata": {
            "data_path": args.data,
            "config_path": args.config,
            "conv_id": conv_id,
            "conv_index": args.conv_index,
            "n_sessions": len(sessions),
            "n_turns": len(all_turns),
            "total_questions": len(test_set),
            "judge": args.judge,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "overall": {},
        "by_category": {},
    }

    all_entries: Dict[str, List[Dict[str, Any]]] = {FASTASEM_NAME: fa_entries}
    all_entries.update(results)

    for name in [FASTASEM_NAME] + args.systems:
        entries = all_entries.get(name, [])
        summary["overall"][name] = score_entries(entries, use_judge=args.judge)
        summary["by_category"].setdefault(name, {})
        summary["by_category"][name] = score_by_category(entries)

    summary_path = os.path.join(args.results_dir, "fairplay_locomo10_conv26.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- Print table ------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"FAIR-PLAY COMPARISON (identical {len(test_set)} questions, same metric fns)")
    print("=" * 72)
    hdr = f"{'Method':<14}{'n':>4}{'EM%':>8}{'ROUGE-L%':>10}{'BERT-F1%':>10}"
    if args.judge:
        hdr += f"{'Judge%':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name in [FASTASEM_NAME] + args.systems:
        s = summary["overall"].get(name, {})
        if not s:
            continue
        line = (f"{name:<14}{s.get('n', 0):>4}"
                f"{100 * s.get('em', 0):>8.1f}"
                f"{100 * s.get('rouge_l', 0):>10.1f}"
                f"{100 * s.get('bertscore_f1', 0):>10.1f}")
        if args.judge:
            jd = s.get("judge")
            line += f"{(100 * jd if jd is not None else float('nan')):>8.1f}"
        print(line)

    print("\nPer-category (EM% / ROUGE-L% / BERT-F1%):")
    for name in [FASTASEM_NAME] + args.systems:
        cats = summary["by_category"].get(name, {})
        if not cats:
            continue
        print(f"  {name}:")
        for c, s in cats.items():
            print(f"    {c:<28} n={s.get('n', 0):>3} "
                  f"EM={100 * s.get('em', 0):5.1f}  "
                  f"RL={100 * s.get('rouge_l', 0):5.1f}  "
                  f"BS={100 * s.get('bertscore_f1', 0):5.1f}")

    print(f"\nSummary written to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
