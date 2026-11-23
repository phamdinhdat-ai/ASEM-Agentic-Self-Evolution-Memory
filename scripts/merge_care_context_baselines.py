"""Validate and merge full Qwen3-8B NoMemory and FullContext-8K LoCoMo shards."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from scripts.run_care_context_baselines import ARMS, MODEL, summarize
from scripts.run_care_full_locomo import SHARDS, selected_questions
from scripts.run_care_locomo import _save_json, score_locomo


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(rows: list[dict], arm: str, seed: int = 20261112) -> list[float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(
            row["care_score"] - row["baseline_score"])
    keys = list(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(10_000):
        values = [v for _ in keys for v in groups[rng.choice(keys)]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return [draws[250], draws[9750]]


def merge(dataset_path: Path, root: Path, care_root: Path) -> dict:
    data_bytes = dataset_path.read_bytes()
    dataset = json.loads(data_bytes)
    data_sha = hashlib.sha256(data_bytes).hexdigest()
    rows, receipts, coverage = [], [], {}
    seen = set()
    reference_settings = None
    for index in range(5):
        base = root / f"shard-{index}" / f"care_context_s{index}"
        report_path, status_path = base / "results/report.json", base / "status.json"
        if not report_path.exists() or not status_path.exists():
            raise FileNotFoundError(f"Missing baseline shard {index}")
        report, status = (json.loads(p.read_text(encoding="utf-8"))
                          for p in (report_path, status_path))
        manifest = report["manifest"]
        expected = selected_questions(dataset, index)
        if (status["state"] != "complete"
                or manifest["protocol"] != "care-context-baselines-locomo-v1"
                or manifest["dataset_sha256"] != data_sha
                or manifest["model"] != MODEL
                or manifest["shard_index"] != index
                or manifest["shard_samples"] != list(SHARDS[index])
                or manifest["question_limit"] != 0
                or manifest["question_ids"] != [item["id"] for item in expected]
                or manifest["arms"] != list(ARMS)
                or len(report["rows"]) != len(expected)):
            raise ValueError(f"Baseline shard {index} identity/coverage mismatch")
        settings = {k: manifest[k] for k in (
            "window_tokens", "max_answer_tokens", "safety_tokens",
            "full_context_policy", "no_memory_prompt_sha256",
            "full_context_prompt_sha256", "scorer", "data_adapter",
            "gold_in_model_context")}
        if reference_settings is None:
            reference_settings = settings
        elif settings != reference_settings:
            raise ValueError("Baseline shard prompt/budget settings differ")
        for row, item in zip(report["rows"], expected):
            if (row["id"] != item["id"] or row["question"] != item["question"]
                    or row["sample_id"] != item["sample_id"]
                    or row["category"] != item["category"]
                    or set(row["systems"]) != set(ARMS)
                    or row["id"] in seen):
                raise ValueError(f"Baseline question mismatch: shard {index}")
            seen.add(row["id"])
            for arm in ARMS:
                value = row["systems"][arm]
                actual = score_locomo(value["answer"], item["answer"], item["category"])
                if abs(actual - value["score"]) > 1e-10:
                    raise ValueError(f"Baseline score mismatch: {row['id']} {arm}")
        rows.extend(report["rows"])
        coverage.update(manifest["coverage"])
        receipts.append({"index": index, "report_sha256": sha(report_path),
                         "n": len(report["rows"]), "seconds": status["seconds"]})
    if len(rows) != 1986 or len(seen) != 1986:
        raise ValueError("Baseline full coverage mismatch")
    care = {}
    for index in range(5):
        path = care_root / f"shard-{index}" / f"care_relation_full_s{index}" / "results/report.json"
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            care[row["id"]] = row
    if set(care) != seen:
        raise ValueError("CARE and baseline question IDs differ")
    comparisons = {}
    for arm in ARMS:
        comparisons[arm] = {}
        for group_name, subset in (
            ("all", rows),
            ("answerable", [r for r in rows if r["category"] != 5]),
            ("adversarial", [r for r in rows if r["category"] == 5]),
        ):
            paired = [{"id": r["id"], "sample_id": r["sample_id"],
                       "care_score": care[r["id"]]["systems"]["bridge_v2"]["f1"],
                       "baseline_score": r["systems"][arm]["score"]}
                      for r in subset]
            deltas = [p["care_score"] - p["baseline_score"] for p in paired]
            comparisons[arm][group_name] = {
                "n": len(paired),
                "care_mean": sum(p["care_score"] for p in paired) / len(paired),
                "baseline_mean": sum(p["baseline_score"] for p in paired) / len(paired),
                "mean_delta": sum(deltas) / len(deltas),
                "bootstrap_conversation_95": bootstrap(paired, arm),
                "wins": sum(d > 1e-10 for d in deltas),
                "losses": sum(d < -1e-10 for d in deltas),
                "ties": sum(abs(d) <= 1e-10 for d in deltas),
            }
    return {
        "protocol": "care-context-baselines-locomo-merge-v1",
        "dataset_sha256": data_sha, "model": MODEL,
        "settings": reference_settings, "n_questions": len(rows),
        "n_predictions": len(rows) * len(ARMS),
        "category_counts": dict(Counter(str(r["category"]) for r in rows)),
        "shards": receipts, "context_coverage": coverage,
        "aggregate": {arm: summarize(rows, arm) for arm in ARMS},
        "by_category": {str(cat): {arm: summarize(
            [r for r in rows if r["category"] == cat], arm) for arm in ARMS}
            for cat in range(1, 6)},
        "paired_vs_care": comparisons,
        "rows": rows,
        "note": "FullContext-8K is a fixed first-five plus recent-history truncation, not full untruncated LoCoMo.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("datasets/locomo/locomo10.json"))
    parser.add_argument("--root", type=Path, default=Path("build/care/context_baselines"))
    parser.add_argument("--care-root", type=Path, default=Path("build/care/relation_full"))
    parser.add_argument("--output", type=Path,
                        default=Path("build/care/context_baselines_merged.json"))
    args = parser.parse_args()
    result = merge(args.dataset, args.root, args.care_root)
    _save_json(args.output, result)
    print(json.dumps({"n_questions": result["n_questions"],
                      "aggregate": result["aggregate"],
                      "paired_vs_care": result["paired_vs_care"],
                      "context_coverage": result["context_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
