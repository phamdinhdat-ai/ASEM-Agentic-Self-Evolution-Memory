"""Pair the 50-question full-history sensitivity study with saved CARE rows."""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path
import random

from scripts.run_care_full_history_reader import select_audit_questions


def main() -> None:
    root = Path("build/care/full_history_40k")
    paths = [Path(p) for p in glob.glob(str(root / "shard-*/**/report.json"), recursive=True)]
    if len(paths) != 5:
        raise ValueError(f"Expected five full-history shard reports; got {len(paths)}")
    dense = [row for path in paths for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    full = {row["id"]: row for row in dense}
    if len(full) != 50 or len(dense) != 50:
        raise ValueError("Full-history cohort must contain 50 unique questions")
    dataset = json.loads(Path("datasets/locomo/locomo10.json").read_text(encoding="utf-8"))
    expected = {row["id"] for shard in range(5)
                for row in select_audit_questions(dataset, shard)}
    if set(full) != expected:
        raise ValueError("Full-history question IDs differ from the frozen selection")
    care_paths = [Path(p) for p in glob.glob("build/care/relation_full/**/report.json",
                                             recursive=True)]
    if len(care_paths) != 5:
        raise ValueError("Expected five complete CARE reports")
    care = {row["id"]: row["systems"]["bridge_v2"] for path in care_paths
            for row in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    if not expected <= set(care):
        raise ValueError("CARE results do not cover the full-history cohort")
    rows = [{"id": qid, "sample_id": full[qid]["sample_id"],
             "category": full[qid]["category"], "full_score": full[qid]["score"],
             "care_score": care[qid]["f1"],
             "full_prompt_tokens": full[qid]["prompt_tokens_server"],
             "full_completion_tokens": full[qid]["completion_tokens"],
             "care_calls": care[qid]["new_calls"],
             "full_turns_visible": full[qid]["turns_visible"]} for qid in sorted(expected)]
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["sample_id"]].append(row)
    names = sorted(clusters)
    rng = random.Random(20261113)
    draws = []
    for _ in range(10000):
        sampled = [row for _ in names for row in clusters[rng.choice(names)]]
        draws.append(sum(row["care_score"] - row["full_score"] for row in sampled)
                     / len(sampled))
    draws.sort()
    answerable = [row for row in rows if row["category"] != 5]
    unsupported = [row for row in rows if row["category"] == 5]
    result = {
        "protocol": "care-full-history-40k-v1",
        "scope": "frozen 50-question sensitivity cohort; not complete LoCoMo",
        "n": len(rows), "n_conversations": len(clusters),
        "care": {"aggregate": sum(row["care_score"] for row in rows) / len(rows),
                 "answerable": sum(row["care_score"] for row in answerable) / len(answerable),
                 "refusal_correct": sum(row["care_score"] == 1 for row in unsupported),
                 "calls": sum(row["care_calls"] for row in rows)},
        "full_history_40k": {
            "aggregate": sum(row["full_score"] for row in rows) / len(rows),
            "answerable": sum(row["full_score"] for row in answerable) / len(answerable),
            "refusal_correct": sum(row["full_score"] == 1 for row in unsupported),
            "calls": len(rows),
            "total_model_tokens": sum(row["full_prompt_tokens"]
                                      + row["full_completion_tokens"] for row in rows),
            "mean_prompt_tokens": sum(row["full_prompt_tokens"] for row in rows) / len(rows),
            "min_prompt_tokens": min(row["full_prompt_tokens"] for row in rows),
            "max_prompt_tokens": max(row["full_prompt_tokens"] for row in rows)},
        "care_minus_full_history_40k": {
            "mean": sum(row["care_score"] - row["full_score"] for row in rows) / len(rows),
            "conversation_cluster_bootstrap_95": [draws[250], draws[9750]],
            "improved": sum(row["care_score"] > row["full_score"] for row in rows),
            "worsened": sum(row["care_score"] < row["full_score"] for row in rows),
            "tied": sum(row["care_score"] == row["full_score"] for row in rows)},
        "by_category": {str(category): {
            "n": sum(row["category"] == category for row in rows),
            "care": sum(row["care_score"] for row in rows if row["category"] == category)
                    / sum(row["category"] == category for row in rows),
            "full_history_40k": sum(row["full_score"] for row in rows
                                    if row["category"] == category)
                                / sum(row["category"] == category for row in rows)}
            for category in range(1, 6)},
        "rows": rows}
    output = Path("analysis/CARE_full_history_40k_results_20261113.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("n", "care", "full_history_40k", "care_minus_full_history_40k")},
                     indent=2))


if __name__ == "__main__":
    main()
