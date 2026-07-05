#!/usr/bin/env python3
"""
Ingest and evaluate ONE conversation from locomo10.json.

Usage:
    python scripts/ingest_eval_one_conversation.py --conv 0
    python scripts/ingest_eval_one_conversation.py --conv 5 --system ASEM
    python scripts/ingest_eval_one_conversation.py --conv 0 --system SimRetrieval --judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

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
                    _v = _v.strip().strip('"').strip("'")
                    os.environ.setdefault(_k.strip(), _v)

from asem.logging_utils import setup_logging
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

from eval.systems import build_asem_system, build_baselines
from eval.llm_as_a_judge import LLMJudge, compute_judge_metrics, verdicts_to_list
from asem.backends import build_backend
import yaml

# Import session extraction helpers
from scripts.run_locomo10_experiments import (
    _extract_session_batches,
    _turn_to_text,
    convert_locomo10_to_eval,
)


CATEGORY_NAMES = {
    1: "single_hop", 2: "temporal", 3: "commonsense",
    4: "conversational", 5: "adversarial",
}


def load_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match(preds: List[str], refs: List[str]) -> float:
    matches = [1.0 if normalize(p) == normalize(r) else 0.0 for p, r in zip(preds, refs)]
    return sum(matches) / len(matches) if matches else 0.0


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    results = {"em": exact_match(preds, refs)}
    try:
        import evaluate as hf_eval
        rouge = hf_eval.load("rouge")
        scores = rouge.compute(predictions=preds, references=refs)
        results["rougeL"] = float(scores.get("rougeL", 0.0))
    except Exception:
        results["rougeL"] = 0.0
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest & evaluate one conversation")
    parser.add_argument("--conv", type=int, required=True,
                       help="Conversation index (0-9)")
    parser.add_argument("--config", default="configs/locomo_openai.yaml",
                       help="YAML config path")
    parser.add_argument("--system", default="ASEM",
                       choices=["ASEM", "NoMemory", "FullContext", "SimRetrieval",
                                "AtomicLinking", "RLManagerOnly", "ValueRetrievalOnly"],
                       help="System to evaluate")
    parser.add_argument("--judge", action="store_true",
                       help="Run LLM-as-a-Judge after evaluation")
    parser.add_argument("--limit", type=int, default=None,
                       help="Evaluate only first N QA pairs (default: all)")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks_single",
                       help="SQLite DB directory")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────
    dataset_path = "datasets/locomo/locomo10.json"
    dataset = load_dataset(dataset_path)

    if args.conv < 0 or args.conv >= len(dataset):
        print(f"ERROR: --conv must be 0-{len(dataset)-1}")
        sys.exit(1)

    record = dataset[args.conv]
    conversation = record.get("conversation", {})
    qa_list = record.get("qa", [])
    conv_id = f"locomo_{args.conv:04d}"
    speaker_a = conversation.get("speaker_a", "A")
    speaker_b = conversation.get("speaker_b", "B")

    print_header(f"Conversation {args.conv}: {speaker_a} & {speaker_b}")
    print(f"  QA pairs: {len(qa_list)}")

    # ── Limit QA pairs ─────────────────────────────────────────
    if args.limit is not None and args.limit < len(qa_list):
        qa_list = qa_list[: args.limit]
        print(f"  Limited to first {args.limit} QA pairs")

    # Category distribution
    from collections import Counter
    cat_counts = Counter(qa.get("category", 0) for qa in qa_list)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {CATEGORY_NAMES.get(cat, f'cat{cat}')}: {count}")

    # ── Session batches ────────────────────────────────────────
    batches = _extract_session_batches(conversation)

    # If QA pairs are limited, only ingest sessions up to max evidence
    if args.limit is not None:
        from scripts.run_locomo10_experiments import (
            _max_evidence_session, _filter_batches_by_session,
        )
        max_sess = _max_evidence_session(qa_list)
        if max_sess > 0:
            old_count = len(batches)
            batches = _filter_batches_by_session(batches, max_sess)
            print(f"  Sessions limited to first {max_sess} (from {old_count}) due to --limit")

    total_turns = sum(len(t) for _, t in batches)
    print(f"\n  Sessions: {len(batches)}")
    print(f"  Total turns: {total_turns}")
    for label, turns in batches:
        print(f"    {label}: {len(turns)} turns")

    # ── Build system ───────────────────────────────────────────
    print_header("Building System")
    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.db_dir, exist_ok=True)
    db_path = os.path.join(args.db_dir, f"{conv_id}_{args.system}.sqlite")

    # Clean stale DB
    for suffix in ["", "-wal", "-shm", "-journal"]:
        p = db_path + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    if args.system == "ASEM":
        runner = build_asem_system(args.config, args.db_dir)
    else:
        baselines = build_baselines(args.config, args.db_dir)
        runner = baselines[args.system]

    # ── Ingest ─────────────────────────────────────────────────
    print_header("Ingesting Conversation")
    t0 = time.time()

    if hasattr(runner, 'ingest_conversation'):
        print(f"  Processing {len(batches)} sessions ({total_turns} turns total):")
        print(f"  Each session: 1 batched S1 call + 1 batched S2 call + 1 S3 link pass = 3 LLM calls")
        runner.ingest_conversation(batches)
    elif hasattr(runner, 'ingest_session'):
        for label, turns in batches:
            print(f"  Ingesting {label} ({len(turns)} turns)...", end=" ", flush=True)
            t_start = time.time()
            runner.ingest_session(turns, label)
            print(f"{time.time() - t_start:.1f}s")
    else:
        print("  No batch ingestion — will use per-question history replay")

    if hasattr(runner, 'finalize_conversation'):
        new_edges = runner.finalize_conversation()
        print(f"  Cross-chunk linking: {new_edges} new edges")

    bank_size = runner.bank_size if hasattr(runner, 'bank_size') else 0
    ingest_time = time.time() - t0
    print(f"\n  Ingest time: {ingest_time:.1f}s")
    print(f"  Bank size: {bank_size} notes")

    # ── Answer questions ───────────────────────────────────────
    print_header("Answering Questions")
    questions: List[str] = []
    gold_answers: List[str] = []
    predictions: List[str] = []
    results_detail: List[Dict[str, Any]] = []

    for i, qa in enumerate(qa_list):
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        category = qa.get("category", 1)
        category_name = CATEGORY_NAMES.get(category, f"cat{category}")

        if category == 5:
            gold = qa.get("adversarial_answer", gold)

        enriched_query = (
            f"Conversation between {speaker_a} and {speaker_b}. "
            f"Question: {question}"
        )

        t0_q = time.time()
        pred = runner.answer(enriched_query, [])
        q_time = time.time() - t0_q

        questions.append(question)
        gold_answers.append(str(gold))
        predictions.append(pred)

        status = "✓" if normalize(pred) == normalize(str(gold)) else "✗"
        print(f"  [{i+1}/{len(qa_list)}] {status} {category_name} | {q_time:.1f}s")
        print(f"    Q: {question[:100]}")
        print(f"    Gold: {str(gold)[:80]}")
        print(f"    Pred: {pred[:120]}")

        results_detail.append({
            "conversation_id": conv_id,
            "question_type": category_name,
            "category": category,
            "question": question,
            "expected_answer": str(gold),
            "ai_response": pred,
            "evidence": qa.get("evidence", []),
        })

    # ── Metrics ────────────────────────────────────────────────
    print_header("Results")
    metrics = compute_metrics(predictions, gold_answers)
    print(f"  Exact Match: {metrics['em']:.4f}")
    print(f"  ROUGE-L:     {metrics['rougeL']:.4f}")

    # Per-category
    print("\n  By category:")
    by_cat: Dict[str, Dict[str, List[str]]] = {}
    for q, g, p, qa in zip(questions, gold_answers, predictions, qa_list):
        cat = CATEGORY_NAMES.get(qa.get("category", 1), "other")
        by_cat.setdefault(cat, {"preds": [], "refs": []})
        by_cat[cat]["preds"].append(p)
        by_cat[cat]["refs"].append(g)

    for cat, data in sorted(by_cat.items()):
        em = exact_match(data["preds"], data["refs"])
        print(f"    {cat}: EM={em:.4f} ({len(data['preds'])} questions)")

    # ── Judge (optional) ───────────────────────────────────────
    if args.judge:
        print_header("LLM-as-a-Judge")
        judge_backend = build_backend(cfg["inference"])
        judge = LLMJudge(backend=judge_backend)

        verdicts = judge.judge_batch(
            questions=questions,
            expected_answers=gold_answers,
            ai_responses=predictions,
            conversation_ids=[conv_id] * len(questions),
            question_types=[
                CATEGORY_NAMES.get(qa.get("category", 1), "other")
                for qa in qa_list
            ],
            categories=[qa.get("category", 0) for qa in qa_list],
        )

        jm = compute_judge_metrics(verdicts, per_category=True)
        print(f"  Judge accuracy: {jm.judge_mean:.4f} ({jm.correct}/{jm.total})")
        print(f"  Perfect (no errors): {jm.judge_pct_perfect:.1f}%")

        # Add judge results to details
        for detail, verdict in zip(results_detail, verdicts):
            detail["evaluation"] = verdict.to_dict()

    # ── Save results ───────────────────────────────────────────
    output_path = f"data/benchmarks/results/single_{conv_id}_{args.system}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "conversation_id": conv_id,
        "system": args.system,
        "speaker_a": speaker_a,
        "speaker_b": speaker_b,
        "num_sessions": len(batches),
        "total_turns": total_turns,
        "bank_size": bank_size,
        "ingest_time_s": round(ingest_time, 1),
        "num_questions": len(qa_list),
        "metrics": metrics,
        "results": results_detail,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
