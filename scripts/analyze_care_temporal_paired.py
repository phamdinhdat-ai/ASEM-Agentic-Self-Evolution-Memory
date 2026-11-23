"""Analyze a completed paired CARE Temporal pilot with conversation bootstrap."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random


def paired_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    deltas = [row["systems"]["temporal"]["f1"]
              - row["systems"]["bridge_v2"]["f1"] for row in rows]
    groups = defaultdict(list)
    for row, delta in zip(rows, deltas):
        groups[row["sample_id"]].append(delta)
    keys = list(groups)
    rng = random.Random(20261109)
    draws = []
    for _ in range(5000):
        sample = [delta for _ in keys for delta in groups[rng.choice(keys)]]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    return {"n": len(rows), "mean_delta": sum(deltas) / len(deltas),
            "wins": sum(delta > 0 for delta in deltas),
            "losses": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
            "conversation_bootstrap_95": [draws[125], draws[4875]]}


def analyze(report: dict, dataset: list[dict]) -> dict:
    gold = {f"{conversation['sample_id']}:{i}": qa for conversation in dataset
            for i, qa in enumerate(conversation["qa"])}
    rows = [row for row in report["rows"] if all(arm in row["systems"]
                                                  for arm in ("bridge_v2", "temporal"))]
    sections = {"all": rows,
                "primary": [row for row in rows if row["slice"] == "primary"],
                "primary_answerable": [row for row in rows
                                       if row["slice"] == "primary" and row["category"] != 5],
                "primary_adversarial": [row for row in rows
                                        if row["slice"] == "primary" and row["category"] == 5],
                "temporal_adversarial": [row for row in rows
                                         if row["slice"] == "temporal_adversarial"]}
    interventions = []
    changed = []
    errors = Counter()
    for row in rows:
        base, temporal = (row["systems"][arm] for arm in ("bridge_v2", "temporal"))
        for arm, result in (("bridge_v2", base), ("temporal", temporal)):
            if result.get("error"):
                errors[arm] += 1
        first_reason = (temporal.get("first_pass") or {}).get("invalid_reason")
        final_reason = temporal.get("invalid_reason")
        item = {"id": row["id"], "slice": row["slice"], "category": row["category"],
                "question": row["question"], "gold": gold[row["id"]].get("answer"),
                "bridge_answer": base["answer"], "temporal_answer": temporal["answer"],
                "bridge_f1": base["f1"], "temporal_f1": temporal["f1"],
                "bridge_calls": base["new_calls"], "temporal_calls": temporal["new_calls"],
                "first_invalid_reason": first_reason, "final_invalid_reason": final_reason}
        if str(first_reason).startswith("temporal_") or str(final_reason).startswith("temporal_"):
            interventions.append(item)
        if base["answer"] != temporal["answer"] or base["f1"] != temporal["f1"]:
            changed.append(item)
    return {"manifest": report["manifest"], "aggregate": report["aggregate"],
            "by_slice": report["by_slice"], "paired": {
                name: paired_summary(group) for name, group in sections.items()},
            "errors": dict(errors), "interventions": interventions,
            "changed_answers": changed,
            "note": "Paired pilot, not full LoCoMo; bootstrap clusters by conversation."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("build/care/locomo10.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(json.loads(args.report.read_text(encoding="utf-8")),
                     json.loads(args.dataset.read_text(encoding="utf-8")))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
