"""Merge the complete matched-reader dense run with saved CARE traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import random

from scripts.run_care_locomo import score_locomo


ARMS = ("bm25_8_first", "bm25_8_conditional",
        "dense_8_first", "dense_8_conditional")
PAIRS = (("dense_8_first", "bm25_8_first"),
         ("bm25_8_conditional", "bm25_8_first"),
         ("dense_8_conditional", "dense_8_first"),
         ("bm25_8_conditional", "dense_8_conditional"))


def load_rows(pattern: str) -> dict[str, dict]:
    paths = [Path(path) for path in glob.glob(pattern, recursive=True)]
    if len(paths) != 5:
        raise ValueError(f"Expected five shard reports matching {pattern}; found {len(paths)}")
    rows = [row for path in paths for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate question in shard reports")
    return by_id


def summary(rows: list[dict], arm: str) -> dict:
    scores = [row[arm]["score"] for row in rows]
    answerable = [row[arm] for row in rows if row["category"] != 5]
    unsupported = [row[arm] for row in rows if row["category"] == 5]
    return {"n": len(rows), "aggregate": sum(scores) / len(scores),
            "answerable": sum(v["score"] for v in answerable) / len(answerable),
            "refusal_correct": sum(v["score"] == 1 for v in unsupported),
            "refusal_n": len(unsupported),
            "source_recall": sum(v["source_recall"] for v in answerable) / len(answerable),
            "model_calls": sum(row[arm]["model_calls"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/locomo/locomo10.json"))
    parser.add_argument("--care-pattern", default="build/care/relation_full/**/report.json")
    parser.add_argument("--dense-pattern", default="build/care/matched_reader/shard-*/**/report.json")
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/CARE_matched_reader_results_20261113.json"))
    args = parser.parse_args()
    care = load_rows(args.care_pattern)
    dense = load_rows(args.dense_pattern)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    gold = {f"{sample['sample_id']}:{index}": qa
            for sample in dataset for index, qa in enumerate(sample["qa"])}
    if set(care) != set(dense) or set(care) != set(gold):
        raise ValueError("Four-arm ablation does not cover the same questions")
    paired = []
    for qid in sorted(gold):
        qa = gold[qid]
        category = int(qa["category"])
        sources = set(qa.get("evidence") or [])
        original = care[qid]["systems"]["bridge_v2"]
        first = original["first_pass"] or original

        def sparse_value(answer: dict, calls: int) -> dict:
            bundle = set(answer["bundle_ids"])
            return {"score": score_locomo(answer["answer"], str(qa.get("answer", "")), category),
                    "source_recall": len(sources & bundle) / len(sources) if sources else 0,
                    "model_calls": calls}

        row = {"id": qid, "sample_id": care[qid]["sample_id"],
               "category": category,
               "bm25_8_first": sparse_value(first, 1),
               "bm25_8_conditional": sparse_value(original, original["new_calls"])}
        for arm in ("dense_8_first", "dense_8_conditional"):
            value = dense[qid]["systems"][arm]
            row[arm] = {"score": value["score"],
                        "source_recall": value["evidence_recall"] or 0,
                        "model_calls": value["model_calls"],
                        "prompt_tokens": value["prompt_tokens"],
                        "completion_tokens": value["completion_tokens"]}
        if abs(row["bm25_8_conditional"]["score"] - original["f1"]) > 1e-10:
            raise ValueError(f"CARE score disagreement at {qid}")
        paired.append(row)
    clusters = defaultdict(list)
    for row in paired:
        clusters[row["sample_id"]].append(row)
    names = sorted(clusters)
    contrasts = {}
    for a, b in PAIRS:
        rng = random.Random(20261113)
        values = [row[a]["score"] - row[b]["score"] for row in paired]
        cluster_totals = {name: (sum(row[a]["score"] - row[b]["score"]
                                     for row in clusters[name]), len(clusters[name]))
                          for name in names}
        draws = []
        for _ in range(10000):
            sample = [cluster_totals[rng.choice(names)] for _ in names]
            draws.append(sum(total for total, _ in sample)
                         / sum(count for _, count in sample))
        draws.sort()
        contrasts[f"{a}_minus_{b}"] = {"mean": sum(values) / len(values),
                                      "improved": sum(v > 1e-12 for v in values),
                                      "worsened": sum(v < -1e-12 for v in values),
                                      "tied": sum(abs(v) <= 1e-12 for v in values),
                                      "cluster_bootstrap_95": [draws[250], draws[9750]]}
    output = {"protocol": "care-matched-reader-dense-top8-v1",
              "n_questions": len(paired), "n_conversations": len(clusters),
              "arms": {arm: summary(paired, arm) for arm in ARMS},
              "contrasts": contrasts,
              "interpretation": "Within-public-LoCoMo ablation; not held-out validation"}
    resource = json.loads(Path("analysis/ASEM_CARE_resource_metrics_20261113.json")
                          .read_text(encoding="utf-8"))["care"]
    sparse_costs = {
        "bm25_8_first": (resource["first_prompt_tokens"]["total"],
                         resource["first_completion_tokens"]["total"]),
        "bm25_8_conditional": (resource["prompt_tokens"]["total"],
                               resource["completion_tokens"]["total"]),
    }
    for arm in ARMS:
        if arm in sparse_costs:
            prompt, completion = sparse_costs[arm]
        else:
            prompt = sum(row[arm]["prompt_tokens"] for row in paired)
            completion = sum(row[arm]["completion_tokens"] for row in paired)
        output["arms"][arm]["prompt_tokens"] = prompt
        output["arms"][arm]["completion_tokens"] = completion
        output["arms"][arm]["total_model_tokens"] = prompt + completion
        output["arms"][arm]["mean_model_tokens"] = (prompt + completion) / len(paired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
