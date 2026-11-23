"""Summarize CARE, context baselines and retrieval with context coverage diagnostics."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/locomo/locomo10.json"
CONTEXT = ROOT / "build/care/context_baselines_merged.json"
SIM = ROOT / "build/care/sim_retrieval_merged.json"
CARE = ROOT / "build/care/relation_full_merged.json"
OUTPUT = ROOT / "analysis/ASEM_CARE_overview_20261112_data.json"
ARMS = ("care", "sim_retrieval", "full_context_8k", "no_memory")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    return sum(values) / len(values) if values else None


def ci(rows, arm, seed=20261112):
    groups = defaultdict(list)
    for row in rows:
        groups[row["sample_id"]].append(row["scores"]["care"] - row["scores"][arm])
    keys = list(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        values = [v for _ in keys for v in groups[rng.choice(keys)]]
        draws.append(mean(values))
    draws.sort()
    return [draws[250], draws[9750]]


def turn_ids(conversation):
    sessions = sorted((int(match.group(1)), key) for key in conversation
                      if (match := re.fullmatch(r"session_(\d+)", key)))
    return [str(turn["dia_id"]) for _, key in sessions for turn in conversation[key]]


def main():
    dataset_bytes = DATASET.read_bytes()
    data = json.loads(dataset_bytes)
    sha = hashlib.sha256(dataset_bytes).hexdigest()
    context, sim, care = read(CONTEXT), read(SIM), read(CARE)
    if any(x["dataset_sha256"] != sha or x["n_questions"] != 1986
           for x in (context, sim, care)):
        raise ValueError("Dataset/coverage mismatch")
    care_rows = {}
    for index in range(5):
        path = (ROOT / "build/care/relation_full" / f"shard-{index}" /
                f"care_relation_full_s{index}" / "results/report.json")
        for row in read(path)["rows"]:
            care_rows[row["id"]] = row
    context_rows = {row["id"]: row for row in context["rows"]}
    sim_rows = {row["id"]: row for row in sim["rows"]}
    if not len(care_rows) == len(context_rows) == len(sim_rows) == 1986:
        raise ValueError("Question coverage mismatch")
    kept_by_sample, retention = {}, {}
    for sample in data:
        sid = sample["sample_id"]
        ids = turn_ids(sample["conversation"])
        item = context["context_coverage"][sid]
        head, recent = item["head_turns"], item["recent_turns"]
        kept = ids if head + recent >= len(ids) else ids[:head] + (ids[-recent:] if recent else [])
        if len(ids) != item["total_turns"] or len(kept) != item["included_turns"]:
            raise ValueError(f"Context retention mismatch {sid}")
        kept_by_sample[sid] = set(kept)
        retention[sid] = {"total_turns": len(ids), "kept_turns": len(kept),
                          "fraction": len(kept) / len(ids), "n_questions": len(sample["qa"])}
    all_ids = {f"{sample['sample_id']}:{ordinal}"
               for sample in data for ordinal, _ in enumerate(sample["qa"])}
    if all_ids != set(care_rows) or all_ids != set(context_rows) or all_ids != set(sim_rows):
        raise ValueError("Exact question IDs differ")
    rows = []
    unknown_evidence_ids = set()
    malformed_evidence_questions = []
    for sample in data:
        sid = sample["sample_id"]
        source_ids = set(turn_ids(sample["conversation"]))
        kept = kept_by_sample[sid]
        for ordinal, qa in enumerate(sample["qa"]):
            qid = f"{sid}:{ordinal}"
            evidence = set(str(value) for value in (qa.get("evidence") or []))
            unknown = evidence - source_ids
            unknown_evidence_ids.update(unknown)
            if unknown:
                malformed_evidence_questions.append(qid)
            covered = len(evidence & kept)
            qrow = care_rows[qid]
            frow = context_rows[qid]
            srow = sim_rows[qid]
            rows.append({
                "id": qid, "sample_id": sid, "category": int(qa["category"]),
                "evidence_count": len(evidence), "context_evidence_count": covered,
                "context_evidence_fraction": (covered / len(evidence)
                    if evidence and not unknown else None),
                "context_evidence_group": ("malformed" if unknown else
                    "unannotated" if not evidence else
                    "none" if covered == 0 else "all" if covered == len(evidence) else "partial"),
                "scores": {
                    "care": qrow["systems"]["bridge_v2"]["f1"],
                    "relation_temporal": qrow["systems"]["relation_temporal"]["f1"],
                    "sim_retrieval": srow["systems"]["sim_retrieval"]["score"],
                    "full_context_8k": frow["systems"]["full_context_8k"]["score"],
                    "no_memory": frow["systems"]["no_memory"]["score"],
                },
            })
    summary = {}
    for group, subset in (
        ("all", rows),
        ("answerable", [r for r in rows if r["category"] != 5]),
        ("adversarial", [r for r in rows if r["category"] == 5]),
        *((f"category_{cat}", [r for r in rows if r["category"] == cat]) for cat in range(1, 6)),
        *((f"context_evidence_{label}", [r for r in rows
           if r["category"] != 5 and r["context_evidence_group"] == label])
          for label in ("all", "partial", "none", "unannotated", "malformed")),
    ):
        summary[group] = {"n": len(subset),
                          "mean": {arm: mean([r["scores"][arm] for r in subset])
                                   for arm in (*ARMS, "relation_temporal")}}
        if group in ("all", "answerable", "adversarial",
                     "context_evidence_all", "context_evidence_partial",
                     "context_evidence_none") and subset:
            summary[group]["care_minus_ci95"] = {
                arm: ci(subset, arm) for arm in ARMS if arm != "care"}
    answerable = [r for r in rows if r["category"] != 5]
    output = {
        "protocol": "care-overview-context-audit-v1", "dataset_sha256": sha,
        "n_questions": len(rows), "n_conversations": len(data),
        "unknown_gold_evidence_ids": sorted(unknown_evidence_ids),
        "malformed_evidence_questions": malformed_evidence_questions,
        "retention": retention,
        "total_turns": sum(x["total_turns"] for x in retention.values()),
        "kept_turns": sum(x["kept_turns"] for x in retention.values()),
        "mean_gold_evidence_fraction_in_context": mean([
            r["context_evidence_fraction"] for r in answerable
            if r["context_evidence_fraction"] is not None]),
        "evidence_group_counts": {label: sum(r["context_evidence_group"] == label
                                     for r in answerable)
                                  for label in ("all", "partial", "none", "unannotated", "malformed")},
        "summary": summary,
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in (("care", CARE), ("context", CONTEXT), ("sim", SIM))},
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
