"""Summarize a completed three-arm CARE-Bridge LoCoMo run by paired question."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random


def cluster_interval(rows, left: str, right: str) -> tuple[float, float]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(
            row["systems"][right]["f1"] - row["systems"][left]["f1"])
    keys = list(groups)
    rng = random.Random(0)
    draws = []
    for _ in range(10000):
        values = [delta for _ in keys for delta in groups[rng.choice(keys)]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return draws[250], draws[9750]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("build/care/locomo10.json"))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    gold = {f"{conv['sample_id']}:{i}": qa
            for conv in dataset for i, qa in enumerate(conv["qa"])}
    rows = report["rows"]
    for row in rows:
        systems = row["systems"]
        print(json.dumps({"id": row["id"], "category": row["category"],
            "question": row["question"], "gold": gold[row["id"]].get("answer", ""),
            "systems": {arm: {"answer": result["answer"], "f1": result["f1"],
                "recall": result["evidence_recall"], "calls": result["new_calls"],
                "seconds": result["seconds"], "error": result["error"],
                "escalated": result.get("escalated"),
                "invalid_reason": result.get("invalid_reason"),
                "first_pass": result.get("first_pass")}
                for arm, result in systems.items()}}, ensure_ascii=False))
    paired = {}
    for left, right in (("proof_bm25", "bridge"), ("care_v1", "bridge")):
        differences = [row["systems"][right]["f1"] - row["systems"][left]["f1"]
                       for row in rows]
        paired[f"{right}_minus_{left}"] = {"mean": sum(differences) / len(differences),
            "improved": sum(value > 0 for value in differences),
            "worsened": sum(value < 0 for value in differences),
            "tied": sum(value == 0 for value in differences),
            "conversation_bootstrap_95": cluster_interval(rows, left, right)}
    print(json.dumps({"aggregate": report["aggregate"], "paired": paired}, ensure_ascii=False))


if __name__ == "__main__":
    main()
