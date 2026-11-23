"""Read a completed CARE pilot and print paired diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("build/care/locomo10.json"))
    parser.add_argument("--report", type=Path, default=Path("build/care/kaggle_report.json"))
    args = parser.parse_args()
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    gold = {f"{conversation['sample_id']}:{i}": qa
            for conversation in dataset for i, qa in enumerate(conversation["qa"])}
    report = json.loads(args.report.read_text(encoding="utf-8"))
    by_category = defaultdict(lambda: {"bm25": [], "care": []})
    by_conversation = defaultdict(list)
    answerable = {"bm25": [], "care": []}
    answerable_recall = {"bm25": [], "care": []}
    calls = {"bm25": 0, "care": 0}
    model_seconds = {"bm25": 0.0, "care": 0.0}
    improved = worsened = tied = 0
    for row in report["rows"]:
        left, right = row["systems"]["bm25"], row["systems"]["care"]
        for arm, result in (("bm25", left), ("care", right)):
            by_category[row["category"]][arm].append(result["f1"])
            calls[arm] += result["new_calls"]
            model_seconds[arm] += result["seconds"]
            if row["category"] != 5:
                answerable[arm].append(result["f1"])
                if result["evidence_recall"] is not None:
                    answerable_recall[arm].append(result["evidence_recall"])
        improved += right["f1"] > left["f1"]
        worsened += right["f1"] < left["f1"]
        tied += right["f1"] == left["f1"]
        by_conversation[row["sample_id"]].append(right["f1"] - left["f1"])
        qa = gold[row["id"]]
        print(json.dumps({"id": row["id"], "category": row["category"],
            "question": row["question"], "gold": qa.get("answer", ""),
            "gold_evidence": qa.get("evidence"),
            "bm25_answer": left["answer"], "care_answer": right["answer"],
            "bm25_f1": left["f1"], "care_f1": right["f1"],
            "bm25_recall": left["evidence_recall"], "care_recall": right["evidence_recall"],
            "care_error": right["error"], "care_risk": right["risk_reasons"],
            "care_search_queries": right.get("search_queries", [])}, ensure_ascii=False))
    conversations = list(by_conversation)
    randomizer = random.Random(0)
    draws = []
    for _ in range(10000):
        sampled = [by_conversation[randomizer.choice(conversations)] for _ in conversations]
        values = [delta for group in sampled for delta in group]
        draws.append(sum(values) / len(values))
    draws.sort()
    all_deltas = [delta for values in by_conversation.values() for delta in values]
    print(json.dumps({"summary": {"n": len(report["rows"]), "improved": improved,
        "worsened": worsened, "tied": tied, "calls": calls,
        "answer_seconds": model_seconds,
        "mean_delta": sum(all_deltas) / len(all_deltas),
        "n_conversations": len(conversations),
        "conversation_bootstrap_95": [draws[250], draws[9750]],
        "answerable_1_to_4": {arm: {"n": len(values),
            "mean_f1": sum(values) / len(values),
            "mean_evidence_recall": sum(answerable_recall[arm]) / len(answerable_recall[arm])}
            for arm, values in answerable.items()},
        "category_f1": {category: {arm: sum(values) / len(values)
            for arm, values in systems.items()} for category, systems in by_category.items()}}},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
