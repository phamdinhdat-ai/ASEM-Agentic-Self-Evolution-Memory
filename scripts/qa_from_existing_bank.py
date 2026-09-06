"""Run QA against an already-ingested Fast-ASEM memory bank (no re-ingestion).

Usage:
    $env:OPENAI_API_KEY="..."; $env:OPENAI_BASE_URL="https://api.xah.io/v1"
    python scripts/qa_from_existing_bank.py --data datasets/locomo/locomo10.json --conv-index 0 --judge --bertscore
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
    compute_bertscore_batch,
    compute_em,
    compute_rouge_l,
    evaluate_with_judge,
    extract_sessions_from_conv,
)
from eval.systems import build_fast_asem_system
from asem.backends import build_backend
from asem.config import ASEMConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="QA from an existing Fast-ASEM memory bank")
    parser.add_argument("--data", default="datasets/locomo/locomo10.json")
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml")
    parser.add_argument("--conv-index", type=int, default=0, help="Index of the conversation in the dataset")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks")
    parser.add_argument("--out", default="data/benchmarks/results/fastasem_locomo10_conv0_qa.json")
    parser.add_argument("--preds", default="data/benchmarks/results/preds/locomo10_FastASEM_conv0_qa.jsonl")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of QA pairs")
    args = parser.parse_args()

    # Load dataset + pick conversation
    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conversations = raw if isinstance(raw, list) else [raw]
    conv = conversations[args.conv_index]
    conv_id = conv.get("conversation_id") or conv.get("sample_id") or f"conv_{args.conv_index}"
    conv_data = conv.get("conversation", conv)
    sessions = extract_sessions_from_conv(conv_data)
    qa_items = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]
    if args.limit:
        qa_items = qa_items[: args.limit]

    print("=" * 70)
    print(f"QA FROM EXISTING BANK | conv={conv_id} | {len(sessions)} sessions | {len(qa_items)} QA pairs")
    print(f"Judge: {args.judge} | BERTScore: {args.bertscore}")
    print("=" * 70)

    # Build system on the EXISTING bank (no reset, no re-ingest)
    system = build_fast_asem_system(args.config, args.db_dir)
    bank = system.pipeline.memory_bank
    print(f"Memory bank: {bank.size()} notes (reusing existing, no re-ingestion)\n")

    backend = build_backend(ASEMConfig.load(args.config).inference)

    # Stream predictions
    os.makedirs(os.path.dirname(os.path.abspath(args.preds)), exist_ok=True)
    preds_fh = open(args.preds, "w", encoding="utf-8")

    results: Dict[str, Any] = {
        "metadata": {
            "data_path": args.data,
            "config_path": args.config,
            "conv_id": conv_id,
            "conv_index": args.conv_index,
            "bank_size": bank.size(),
            "timestamp": datetime.now().isoformat(),
        },
        "total_questions": 0,
        "em_score": 0.0,
        "rouge_l": 0.0,
        "judge_score": 0.0,
        "qa_time_sec": 0.0,
        "by_category": defaultdict(lambda: {"total": 0, "em": 0.0, "rouge_l": 0.0, "judge": 0.0}),
        "qa_log": [],
    }

    t0 = time.time()
    for idx, qa in enumerate(qa_items):
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        cat_id = qa.get("category", 1)
        cat_name = CATEGORY_NAMES.get(cat_id, f"Category {cat_id}")
        if not question or not gold:
            continue

        pred = system.answer(question)
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
        results["qa_log"].append(entry)
        results["total_questions"] += 1
        results["em_score"] += em
        results["rouge_l"] += rouge
        if args.judge:
            results["judge_score"] += 1.0 if judge_ok else 0.0
        cstat = results["by_category"][cat_name]
        cstat["total"] += 1
        cstat["em"] += em
        cstat["rouge_l"] += rouge
        if args.judge:
            cstat["judge"] += 1.0 if judge_ok else 0.0

        # Stream prediction immediately
        preds_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        preds_fh.flush()

        # Save partial results
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"[{idx + 1}/{len(qa_items)}] {cat_name} | EM={em:.0f} | "
              f"Q: {question[:60]} | A: {str(pred)[:60]} | gold: {str(gold)[:40]}")

    results["qa_time_sec"] = time.time() - t0

    if args.bertscore:
        bs = compute_bertscore_batch([e["pred"] for e in results["qa_log"]],
                                     [e["ref"] for e in results["qa_log"]])
        for e, s in zip(results["qa_log"], bs):
            e["bertscore_f1"] = s
        results["bertscore_f1"] = sum(bs) / max(1, len(bs))

    n = max(1, results["total_questions"])
    results["avg_em"] = results["em_score"] / n
    results["avg_rouge_l"] = results["rouge_l"] / n
    if args.judge:
        results["avg_judge"] = results["judge_score"] / n
    for cname, cstat in results["by_category"].items():
        cn = max(1, cstat["total"])
        cstat["avg_em"] = cstat["em"] / cn
        cstat["avg_rouge_l"] = cstat["rouge_l"] / cn
        if args.judge:
            cstat["avg_judge"] = cstat["judge"] / cn

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    preds_fh.close()

    print("\n" + "=" * 70)
    summary = f"RESULT | {conv_id} | {n} QA | EM={results['avg_em']:.3f} | ROUGE-L={results['avg_rouge_l']:.3f}"
    if args.judge:
        summary += f" | Judge={results.get('avg_judge', 0.0):.3f}"
    if args.bertscore:
        summary += f" | BERTScore-F1={results.get('bertscore_f1', 0.0):.3f}"
    summary += f" | QA time={results['qa_time_sec']:.1f}s"
    print(summary)
    print(f"Results: {args.out}")
    print(f"Predictions: {args.preds}")
    print("=" * 70)


if __name__ == "__main__":
    main()
