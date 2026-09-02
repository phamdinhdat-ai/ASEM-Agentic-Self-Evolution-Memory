"""Reproduce the EXACT context string the AnswerAgent LLM receives for the
failing conv-1 questions (same logic as AnswerAgent.direct_answer), so we can
judge whether the retrieved context is sufficient to answer.

Attaches to the existing bank (does NOT re-ingest).

Usage:
    python scripts/dump_llm_context_conv1.py --limit 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.systems import build_fast_asem_system  # noqa: E402


def build_context(candidates) -> str:
    """Mirror AnswerAgent.direct_answer context construction exactly."""
    sorted_notes = sorted(candidates, key=lambda n: n.t if n.t else datetime.min)
    items = []
    for n in sorted_notes:
        date_prefix = f"[{n.session_date}] " if n.session_date else f"[{n.t.strftime('%d %B %Y')}] "
        entities_str = f" (Entities: {', '.join(n.entities)})" if n.entities else ""
        keywords_str = f" (Keywords: {', '.join(n.K[:12])})" if n.K else ""
        desc_str = f" (Description: {n.X})" if (n.X and n.X != n.c) else ""
        items.append(f"- {date_prefix}{n.c}{entities_str}{keywords_str}{desc_str}")
    return "\n".join(items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/locomo/locomo10.json")
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--only", default="", help="Comma-separated question indices to show (1-based)")
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conv = raw[0]
    qa_items = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]
    qa_items = qa_items[: args.limit]

    only = set(int(x) for x in args.only.split(",") if x.strip()) if args.only else None

    system = build_fast_asem_system(args.config, args.db_dir)
    bank = system.pipeline.memory_bank
    print(f"Bank size: {bank.size()} notes\n")

    for i, qa in enumerate(qa_items, start=1):
        if only and i not in only:
            continue
        question = qa.get("question", "")
        gold = qa.get("answer", "")
        candidates = system.pipeline.retriever.retrieve(question, bank)
        context = build_context(candidates)
        print("=" * 90)
        print(f"[{i}] Q: {question}")
        print(f"    GOLD: {gold!r}")
        print(f"    Retrieved {len(candidates)} candidates. EXACT LLM context below:")
        print("-" * 90)
        print(context)
        print("-" * 90)
        print()


if __name__ == "__main__":
    main()
