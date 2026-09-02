"""Diagnostic: dump the retrieved candidate notes for specific conversation-1
questions, to determine whether the gold answer is PRESENT in the top-k
candidates (=> AnswerAgent synthesis problem) or ABSENT (=> retrieval /
ingestion problem).

Attaches to the existing ingested bank (does NOT re-ingest).

Usage:
    python scripts/dump_candidates_conv1.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.systems import build_fast_asem_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump retrieved candidates for conv-1 questions")
    parser.add_argument("--data", default="datasets/locomo/locomo10.json")
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks")
    parser.add_argument("--limit", type=int, default=8, help="Number of QA pairs to inspect")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conv = raw[0]
    qa_items = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]
    qa_items = qa_items[: args.limit]

    system = build_fast_asem_system(args.config, args.db_dir)
    bank = system.pipeline.memory_bank
    print(f"Bank size: {bank.size()} notes\n")

    for i, qa in enumerate(qa_items, start=1):
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        candidates = system.pipeline.retriever.retrieve(question, bank)
        print("=" * 90)
        print(f"[{i}] Q: {question}")
        print(f"    GOLD: {gold!r}")
        print(f"    Retrieved {len(candidates)} candidates:")
        for rank, n in enumerate(candidates, start=1):
            date = n.session_date or (n.t.strftime("%d %B %Y") if n.t else "?")
            ents = f" (entities: {', '.join(n.entities)})" if n.entities else ""
            print(f"      {rank:>2}. [{date}] {n.c}{ents}")
        print()


if __name__ == "__main__":
    main()
