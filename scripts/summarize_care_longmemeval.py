"""Merge five CARE LongMemEval cleaned-oracle diagnostic shards."""

from __future__ import annotations

from collections import Counter
import glob
import json
from pathlib import Path
import random


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    paths = sorted(glob.glob("build/care/longmemeval/shard-*/**/report.json", recursive=True))
    if len(paths) != 5:
        raise ValueError(f"Expected five shard reports; found {len(paths)}")
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if {report["manifest"]["shard_index"] for report in reports} != set(range(5)):
        raise ValueError("LongMemEval shard index mismatch")
    hashes = {report["manifest"]["dataset_sha256"] for report in reports}
    if len(hashes) != 1:
        raise ValueError("LongMemEval dataset checksums differ across shards")
    if any(report["manifest"]["question_limit"] for report in reports):
        raise ValueError("A pilot report was included in the full merge")
    rows = [row for report in reports for row in report["rows"]]
    ids = [str(row["id"]) for row in rows]
    if len(rows) != 500 or len(set(ids)) != 500:
        raise ValueError("Expected 500 distinct question IDs")
    if any(set(row["systems"]) != {"first", "conditional"} for row in rows):
        raise ValueError("Missing first or conditional outcome")
    if sum(row["unsupported"] for row in rows) != 30:
        raise ValueError("Expected exactly 30 abstention questions")
    rows.sort(key=lambda row: row["id"])
    answerable = [row for row in rows if not row["unsupported"]]
    unsupported = [row for row in rows if row["unsupported"]]

    def arm_summary(arm: str) -> dict:
        return {
            "diagnostic_mean": mean([row["systems"][arm]["score"] for row in rows]),
            "answerable_f1": mean([row["systems"][arm]["f1"] for row in answerable]),
            "answerable_em": mean([row["systems"][arm]["em"] for row in answerable]),
            "abstention_correct": sum(row["systems"][arm]["score"] == 1 for row in unsupported),
            "abstention_n": len(unsupported),
            "source_session_recall": mean([row["systems"][arm]["source_session_recall"]
                                           for row in answerable]),
            "calls": sum(row["systems"][arm]["calls"] for row in rows),
        }

    rng = random.Random(20261114)
    differences = [row["systems"]["conditional"]["score"]
                   - row["systems"]["first"]["score"] for row in rows]
    draws = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    paired = {
        "mean": mean(differences), "question_bootstrap_95": [draws[250], draws[9750]],
        "improved": sum(value > 0 for value in differences),
        "worsened": sum(value < 0 for value in differences),
        "tied": sum(value == 0 for value in differences),
    }
    result = {
        "protocol": "care-longmemeval-cleaned-oracle-v1",
        "scope": "diagnostic cleaned oracle, not official LongMemEval accuracy",
        "dataset_sha256": next(iter(hashes)), "n": len(rows),
        "question_type_counts": dict(Counter(row["question_type"] for row in rows)),
        "first": arm_summary("first"),
        "conditional": arm_summary("conditional"),
        "conditional_minus_first": paired,
        "by_question_type": {
            kind: {arm: mean([row["systems"][arm]["score"] for row in rows
                              if row["question_type"] == kind])
                   for arm in ("first", "conditional")}
            for kind in sorted({row["question_type"] for row in rows})},
        "rows": rows,
    }
    output = Path("analysis/CARE_LongMemEval_transfer_results_20261114.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("n", "first", "conditional", "conditional_minus_first")}, indent=2))


if __name__ == "__main__":
    main()
