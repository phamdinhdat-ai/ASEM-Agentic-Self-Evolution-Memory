"""Verify and merge the five completed full LoCoMo CARE shard reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from scripts.run_care_full_locomo import ARMS, SHARDS, _aggregate, selected_questions
from scripts.run_care_locomo import _save_json, score_locomo


def _bootstrap(rows: list[dict], *, seed: int = 20261109) -> list[float] | None:
    groups = defaultdict(list)
    for row in rows:
        systems = row["systems"]
        groups[row["sample_id"]].append(
            systems["relation_temporal"]["f1"] - systems["bridge_v2"]["f1"])
    if not groups:
        return None
    keys = list(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        values = [delta for _ in keys for delta in groups[rng.choice(keys)]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return [draws[250], draws[9750]]


def merge(dataset_path: Path, root: Path) -> dict:
    payload = dataset_path.read_bytes()
    dataset = json.loads(payload)
    data_sha = hashlib.sha256(payload).hexdigest()
    all_expected = {item["id"] for shard in range(len(SHARDS))
                    for item in selected_questions(dataset, shard)}
    rows = []
    shards = []
    reference_code = None
    reference_model = None
    seen = set()
    for index in range(len(SHARDS)):
        base = root / f"shard-{index}" / f"care_relation_full_s{index}"
        status_path = base / "status.json"
        report_path = base / "results/report.json"
        if not report_path.exists():
            raise FileNotFoundError(f"Missing shard {index} report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        manifest = report["manifest"]
        if manifest["protocol"] != "care-relation-full-locomo-v1":
            raise ValueError(f"Unexpected shard {index} protocol")
        if manifest["dataset_sha256"] != data_sha or manifest["shard_index"] != index:
            raise ValueError(f"Wrong data or shard index in shard {index}")
        if manifest["shard_samples"] != list(SHARDS[index]) or manifest["arms"] != list(ARMS):
            raise ValueError(f"Wrong sample or arm manifest in shard {index}")
        expected = selected_questions(dataset, index)
        if manifest["question_ids"] != [item["id"] for item in expected]:
            raise ValueError(f"Selected ID mismatch in shard {index}")
        if reference_code is None:
            reference_code = manifest["code_sha256"]
            reference_model = manifest["model"]
        elif manifest["code_sha256"] != reference_code or manifest["model"] != reference_model:
            raise ValueError("Shard code or model versions differ")
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("state") != "complete":
                raise ValueError(f"Shard {index} status is not complete")
        shard_rows = report["rows"]
        if len(shard_rows) != len(expected):
            raise ValueError(f"Incomplete row count in shard {index}")
        for row, item in zip(shard_rows, expected):
            if row["id"] != item["id"] or row["question"] != item["question"]:
                raise ValueError(f"Question mismatch in shard {index}")
            if set(row["systems"]) != set(ARMS):
                raise ValueError(f"Missing arm in shard {index}, {row['id']}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate ID across shards: {row['id']}")
            seen.add(row["id"])
            for arm in ARMS:
                value = row["systems"][arm]
                actual = score_locomo(value["answer"], item["answer"], item["category"])
                if abs(actual - value["f1"]) > 1e-10:
                    raise ValueError(f"Score mismatch: {row['id']} {arm}")
        rows.extend(shard_rows)
        shards.append({"index": index, "samples": manifest["shard_samples"],
                       "n": len(shard_rows), "report": str(report_path),
                       "aggregate": report["aggregate"]})
    if seen != all_expected:
        raise ValueError("Full dataset coverage mismatch")
    aggregate = {arm: _aggregate(rows, arm) for arm in ARMS}
    by_category = {str(category): {arm: _aggregate(
        [row for row in rows if row["category"] == category], arm) for arm in ARMS}
        for category in (1, 2, 3, 4, 5)}
    subsets = {"all": rows,
               "answerable": [row for row in rows if row["category"] != 5],
               "adversarial": [row for row in rows if row["category"] == 5]}
    paired = {name: {"n": len(group),
             "mean_delta": (sum(row["systems"]["relation_temporal"]["f1"]
                                - row["systems"]["bridge_v2"]["f1"] for row in group)
                            / len(group) if group else None),
             "bootstrap_conversation_95": _bootstrap(group),
             "wins": sum(row["systems"]["relation_temporal"]["f1"]
                         > row["systems"]["bridge_v2"]["f1"] for row in group),
             "losses": sum(row["systems"]["relation_temporal"]["f1"]
                           < row["systems"]["bridge_v2"]["f1"] for row in group)}
              for name, group in subsets.items()}
    interventions = []
    for row in rows:
        baseline, treatment = (row["systems"][arm] for arm in ARMS)
        trace = treatment.get("relation_proof_trace", [])
        if any(part["status"] in {"incomplete", "contradicted"} for part in trace):
            interventions.append({"id": row["id"], "sample_id": row["sample_id"],
                "category": row["category"], "question": row["question"],
                "baseline_answer": baseline["answer"], "treatment_answer": treatment["answer"],
                "baseline_f1": baseline["f1"], "treatment_f1": treatment["f1"],
                "trace": trace})
    return {"protocol": "care-relation-full-locomo-merge-v1",
        "dataset_sha256": data_sha, "code_sha256": reference_code,
        "model": reference_model, "n_conversations": len(SHARDS) * 2,
        "n_questions": len(rows), "n_predictions": len(rows) * len(ARMS),
        "category_counts": dict(Counter(str(row["category"]) for row in rows)),
        "shards": shards, "aggregate": aggregate, "by_category": by_category,
        "paired": paired, "interventions": interventions,
        "note": "Full LoCoMo QA data under text+caption adapter; not official multimodal run."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("build/care/locomo10.json"))
    parser.add_argument("--root", type=Path, default=Path("build/care/relation_full"))
    parser.add_argument("--output", type=Path,
                        default=Path("build/care/relation_full_merged.json"))
    args = parser.parse_args()
    result = merge(args.dataset, args.root)
    _save_json(args.output, result)
    print(json.dumps({"n_questions": result["n_questions"], "aggregate": result["aggregate"],
                      "paired": result["paired"],
                      "n_interventions": len(result["interventions"])}, indent=2))
