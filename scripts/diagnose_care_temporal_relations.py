"""Post-hoc guard diagnostics on saved CARE rows; no model calls or scoring claims."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from asem.care.temporal_relations import TemporalProofGuard
from scripts.run_care_locomo import turns_for


def diagnose(dataset_path: Path, rows_path: Path, arm: str = "bridge") -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    conversations = {sample["sample_id"]: sample["conversation"] for sample in dataset}
    turn_maps = {sample_id: {turn.source_id: turn for turn in turns_for(conversation)}
                 for sample_id, conversation in conversations.items()}
    counts = Counter()
    flags = []
    guard = TemporalProofGuard()
    for row in rows:
        output = row.get("systems", {}).get(arm)
        if not output:
            continue
        counts["rows"] += 1
        ids = output.get("source_ids", [])
        sources = [turn_maps[row["sample_id"]][sid] for sid in ids
                   if sid in turn_maps[row["sample_id"]]]
        result = guard.check(row["question"], output.get("answer", ""), sources)
        counts[result.status] += 1
        if result.status == "incompatible":
            counts["flagged_category_" + str(row["category"])] += 1
            flags.append({"id": row["id"], "category": row["category"],
                          "question": row["question"], "answer": output["answer"],
                          "source_ids": ids, "reason": result.reason,
                          "saved_f1": output.get("f1")})
    return {"dataset": str(dataset_path), "rows": str(rows_path), "arm": arm,
            "counts": dict(counts), "flags": flags,
            "note": "Retrospective source check only; no rerun, retry, or post-guard score."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("build/care/locomo10.json"))
    parser.add_argument("--arm", default="bridge")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = diagnose(args.dataset, args.rows, args.arm)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
