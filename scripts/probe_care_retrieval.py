"""Compare CARE candidate coverage with and without session date in BM25."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from asem.care.sourcefirst import CareSourceFirst, _words
from scripts.run_care_locomo import select_questions, turns_for


def main() -> None:
    dataset = json.loads(Path("build/care/locomo10.json").read_text(encoding="utf-8"))
    conversations = {conversation["sample_id"]: conversation for conversation in dataset}
    questions = select_questions(dataset)
    summary = {variant: {"top8": [], "pool": []} for variant in ("text", "text_date")}
    for question in questions:
        turns = turns_for(conversations[question["sample_id"]]["conversation"])
        for variant in summary:
            memory = CareSourceFirst(turns)
            if variant == "text_date":
                memory._tokens = [Counter(_words(turn.speaker + " " + turn.date + " " + turn.text))
                                  for turn in turns]
                memory._lengths = [sum(tokens.values()) for tokens in memory._tokens]
                memory._avg_length = sum(memory._lengths) / len(memory._lengths) or 1.0
                memory._df = Counter(word for tokens in memory._tokens for word in tokens)
            evidence = set(question["evidence"])
            if not evidence:
                continue
            top8 = {turn.source_id for turn in memory.rank(question["question"], limit=8)}
            pool = {turn.source_id for turn in memory.candidate_pool(question["question"])}
            summary[variant]["top8"].append(len(evidence & top8) / len(evidence))
            summary[variant]["pool"].append(len(evidence & pool) / len(evidence))
            if variant == "text_date":
                print(json.dumps({"id": question["id"], "category": question["category"],
                    "top8": len(evidence & top8) / len(evidence),
                    "pool": len(evidence & pool) / len(evidence)}))
    print(json.dumps({variant: {name: sum(values) / len(values)
        for name, values in metrics.items()} for variant, metrics in summary.items()}))


if __name__ == "__main__":
    main()
