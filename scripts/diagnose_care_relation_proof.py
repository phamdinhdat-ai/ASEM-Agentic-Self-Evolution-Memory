"""Post-hoc relation proof coverage on saved CARE predictions; no model calls."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from asem.care.relation_proof import RelationProofGuard
from scripts.run_care_locomo import turns_for


def diagnose(dataset_path: Path, report_path: Path, arm: str) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    turns = {sample["sample_id"]: turns_for(sample["conversation"]) for sample in dataset}
    maps = {key: {turn.source_id: turn for turn in value} for key, value in turns.items()}
    guards = {key: RelationProofGuard(value) for key, value in turns.items()}
    counts = Counter()
    flags = []
    for row in report["rows"]:
        output = row.get("systems", {}).get(arm)
        if not output:
            continue
        ids = output.get("source_ids") or []
        cited = [maps[row["sample_id"]][sid] for sid in ids
                 if sid in maps[row["sample_id"]]]
        result = guards[row["sample_id"]].check(
            row["question"], output.get("answer", ""), cited)
        counts["rows"] += 1
        counts[result.status] += 1
        if result.status in {"incomplete", "contradicted"}:
            counts["flagged_category_" + str(row["category"])] += 1
            if output.get("f1") == 1:
                counts["flagged_saved_correct"] += 1
            flags.append({"id": row["id"], "category": row["category"],
                          "question": row["question"], "answer": output.get("answer"),
                          "source_ids": ids, "status": result.status,
                          "missing": result.missing,
                          "counterevidence_ids": result.counterevidence_ids,
                          "saved_f1": output.get("f1")})
    return {"report": str(report_path), "arm": arm, "counts": dict(counts),
            "flags": flags, "note": "Retrospective development diagnostic; no retry or new score."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("build/care/locomo10.json"))
    parser.add_argument("--arm", default="bridge_v2")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.dataset, args.report, args.arm)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
