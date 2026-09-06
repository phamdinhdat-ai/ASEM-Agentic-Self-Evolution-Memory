"""Dump the EXACT full context FastASEM's QA LLM receives for a given question.

Attaches to the existing fast_asem bank (no re-ingest). Prints:
  - the query + gold answer
  - the number of retrieved candidates
  - the exact context string (what goes into P_temporal_qa.txt {context})
  - the fully-rendered prompt (template + context + query)

Usage:
    python scripts/dump_fastasem_context.py --only 1 2 3
    python scripts/dump_fastasem_context.py --q "When did Caroline go to the therapy?"
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

# Load .env (OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL) — same as benchmark runner.
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
                    os.environ.setdefault(_k.strip(), _v)

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
    parser.add_argument("--only", default="", help="Comma-separated 1-based question indices")
    parser.add_argument("--q", default="", help="Exact question text to retrieve for")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    conv = raw[0]
    qa_items = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]

    system = build_fast_asem_system(args.config, args.db_dir)
    bank = system.pipeline.memory_bank
    print(f"Bank size: {bank.size()} notes\n")

    # Build the target list
    if args.q:
        targets = [(0, args.q, "")]
    else:
        only = set(int(x) for x in args.only.split(",") if x.strip()) if args.only else None
        targets = []
        for i, qa in enumerate(qa_items, start=1):
            if only and i not in only:
                continue
            targets.append((i, qa.get("question", ""), qa.get("answer", "")))
            if len(targets) >= args.limit:
                break

    # Load the exact prompt template the system uses
    prompt_path = os.path.join(_PROJECT_ROOT, "data", "prompts", "P_temporal_qa.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    for idx, question, gold in targets:
        candidates = system.pipeline.retriever.retrieve(question, bank)
        context = build_context(candidates)
        print("=" * 96)
        print(f"[{idx}] Q: {question}")
        if gold:
            print(f"    GOLD: {gold!r}")
        print(f"    Retrieved {len(candidates)} candidates.")
        print("-" * 96)
        print(">>> EXACT CONTEXT (goes into {context}) <<<")
        print(context)
        print("-" * 96)
        print(">>> FULLY-RENDERED PROMPT <<<")
        print(template.format(context=context, query=question))
        print("=" * 96)
        print()


if __name__ == "__main__":
    main()
