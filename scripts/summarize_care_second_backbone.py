"""Pair Qwen3-4B CARE sensitivity with saved Qwen3-8B outputs."""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path
import random

from scripts.run_care_locomo import score_locomo


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    path = Path("build/care/second_backbone_4b/full/care_second_backbone_4b/results/report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    rows4 = report["rows"]
    ids = report["manifest"]["question_ids"]
    if report["manifest"]["question_limit"] or len(rows4) != len(ids) or len(ids) != 50:
        raise ValueError("Expected the complete frozen 50-question 4B cohort")
    if {row["id"] for row in rows4} != set(ids):
        raise ValueError("4B question IDs mismatch")
    paths8 = glob.glob("build/care/relation_full/**/report.json", recursive=True)
    if len(paths8) != 5:
        raise ValueError("Expected five original Qwen3-8B reports")
    rows8 = {row["id"]: row for path8 in paths8
             for row in json.loads(Path(path8).read_text(encoding="utf-8"))["rows"]}
    dataset = json.loads(Path("datasets/locomo/locomo10.json").read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in dataset}
    paired = []
    for row4 in rows4:
        qid = row4["id"]
        old = rows8[qid]["systems"]["bridge_v2"]
        sid = row4["sample_id"]
        qa = samples[sid]["qa"][int(qid.split(":")[-1])]
        first8 = (old.get("first_pass") or {}).get("answer", old["answer"])
        paired.append({"id": qid, "sample_id": sid, "category": row4["category"],
                       "qwen3_4b_first": row4["systems"]["first"]["score"],
                       "qwen3_4b_conditional": row4["systems"]["conditional"]["score"],
                       "qwen3_8b_first": score_locomo(first8, str(qa.get("answer", "")),
                                                       row4["category"]),
                       "qwen3_8b_conditional": old["f1"]})
    if len({row["sample_id"] for row in paired}) != 10:
        raise ValueError("Expected ten conversation clusters")
    clusters = defaultdict(list)
    for row in paired:
        clusters[row["sample_id"]].append(row)
    rng = random.Random(20261114)
    cluster_names = sorted(clusters)

    def interval(field_a: str, field_b: str) -> list[float]:
        draws = []
        for _ in range(10000):
            sample = [row for _ in cluster_names for row in clusters[rng.choice(cluster_names)]]
            draws.append(mean([row[field_a] - row[field_b] for row in sample]))
        draws.sort()
        return [draws[250], draws[9750]]

    fields = ("qwen3_4b_first", "qwen3_4b_conditional",
              "qwen3_8b_first", "qwen3_8b_conditional")
    result = {
        "protocol": "care-second-backbone-qwen3-4b-v1",
        "scope": "frozen 50-question sensitivity cohort, not complete LoCoMo",
        "n": len(paired), "n_conversations": len(clusters),
        "scores": {field: {"aggregate": mean([row[field] for row in paired]),
                            "answerable": mean([row[field] for row in paired
                                                if row["category"] != 5]),
                            "refusal_correct": sum(row[field] == 1 for row in paired
                                                   if row["category"] == 5)}
                   for field in fields},
        "repair_effect_4b": {"mean": mean([row["qwen3_4b_conditional"]
                                             - row["qwen3_4b_first"] for row in paired]),
                             "conversation_cluster_bootstrap_95": interval(
                                 "qwen3_4b_conditional", "qwen3_4b_first")},
        "repair_effect_8b": {"mean": mean([row["qwen3_8b_conditional"]
                                             - row["qwen3_8b_first"] for row in paired]),
                             "conversation_cluster_bootstrap_95": interval(
                                 "qwen3_8b_conditional", "qwen3_8b_first")},
        "calls_4b": {arm: report["aggregate"][arm]["calls"]
                     for arm in ("first", "conditional")},
        "model_tokens_4b": {arm: report["aggregate"][arm]["model_tokens"]
                             for arm in ("first", "conditional")},
        "rows": paired,
    }
    output = Path("analysis/CARE_second_backbone_results_20261114.json")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("n", "scores", "repair_effect_4b", "repair_effect_8b")}, indent=2))


if __name__ == "__main__":
    main()
