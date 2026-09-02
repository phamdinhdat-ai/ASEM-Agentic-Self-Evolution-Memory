"""Diagnostic: run Stage-4 retrieval + answer on the already-ingested
conversation 1 bank (data/benchmarks/eval_banks/fast_asem.sqlite).

Does NOT re-ingest. Attaches to the existing bank via build_fast_asem_system
(which reuses the same sqlite path) and calls pipeline.read_path() for a
handful of LoCoMo conversation-1 questions, including the integer-answer
ones that previously crashed compute_em.

Usage:
    python scripts/retrieval_check_conv1.py [--limit 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.benchmark_runner import compute_em, compute_rouge_l, extract_sessions_from_conv  # noqa: E402
from eval.systems import build_fast_asem_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval check on ingested conversation 1")
    parser.add_argument("--data", default="datasets/locomo/locomo10.json")
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks")
    parser.add_argument("--limit", type=int, default=8, help="Number of QA pairs to run")
    args = parser.parse_args()

    # Load conversation 1 QA pairs
    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conv = raw[0]
    qa_items = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]
    qa_items = qa_items[: args.limit]

    # Attach to the existing ingested bank.
    system = build_fast_asem_system(args.config, args.db_dir)
    bank = system.pipeline.memory_bank

    # If the bank is empty, ingest conversation 1 first (fast session-level path).
    if bank.size() == 0:
        conv_data = conv.get("conversation", conv)
        sessions = extract_sessions_from_conv(conv_data)
        print(f"Bank empty -> ingesting conversation 1 ({len(sessions)} sessions)...\n")
        system.ingest_conversation(sessions)
        print(f"Ingested. Bank size: {bank.size()} notes\n")
    else:
        print(f"Bank size (already ingested): {bank.size()} notes\n")

    print(f"{'#':<3}{'CAT':<6}{'GOLD':<28}{'PRED':<28}{'EM':<5}{'R-L':<6}{'N':<3}")
    print("-" * 80)
    for i, qa in enumerate(qa_items, start=1):
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        cat = qa.get("category", 1)

        used_notes, answer = system.pipeline.read_path(question)
        em = compute_em(answer, gold)
        rouge = compute_rouge_l(answer, gold)

        g = str(gold)[:26]
        p = str(answer)[:26]
        print(f"{i:<3}{cat:<6}{g:<28}{p:<28}{em:<5.0f}{rouge:<6.2f}{len(used_notes):<3}")

    print("\nDone. (Retrieval + answer ran without AttributeError on int gold answers.)")


if __name__ == "__main__":
    main()
