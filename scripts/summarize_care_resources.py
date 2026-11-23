"""Audit answer-model token receipts and latency for the locked LoCoMo comparison.

CARE calls and events must first be downloaded from the five completed Kaggle
shards into build/care/resource_audit/care_shard_{i}_{calls.json,events.jsonl}.
The other arms retain their receipts in the merged reports.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "build/care/resource_audit"
OUTPUT = ROOT / "analysis/ASEM_CARE_resource_metrics_20261113.json"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def describe(values: list[float]) -> dict:
    return {"total": sum(values), "mean": mean(values), "median": median(values),
            "p95": percentile(values, 0.95), "max": max(values)}


def care_events() -> dict[str, list[dict]]:
    by_arm: dict[str, list[dict]] = {"bridge_v2": [], "relation_temporal": []}
    merged = json.loads((ROOT / "build/care/relation_full_merged.json")
                        .read_text(encoding="utf-8"))
    for shard in range(5):
        calls = json.loads((AUDIT / f"care_shard_{shard}_calls.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in (AUDIT / f"care_shard_{shard}_events.jsonl")
                  .read_text(encoding="utf-8").splitlines() if line]
        report = json.loads(Path(merged["shards"][shard]["report"]).read_text(encoding="utf-8"))
        reported = {(row["id"], arm): value for row in report["rows"]
                    for arm, value in row["systems"].items()}
        assert len(events) == len(reported)
        assert all(event["value"] == reported[(event["id"], event["arm"])]
                   for event in events), shard
        assert len(calls) == sum(event["value"]["new_calls"] for event in events), shard
        offset = 0
        for event in events:
            n_calls = event["value"]["new_calls"]
            receipts = calls[offset:offset + n_calls]
            offset += n_calls
            assert n_calls in (1, 2) and len(receipts) == n_calls
            assert all(c["prompt_tokens"] is not None and c["completion_tokens"] is not None
                       for c in receipts)
            by_arm[event["arm"]].append({"id": event["id"],
                "calls": n_calls,
                "prompt_tokens": sum(c["prompt_tokens"] for c in receipts),
                "prompt_length_squared_sum": sum(c["prompt_tokens"] ** 2 for c in receipts),
                "completion_tokens": sum(c["completion_tokens"] for c in receipts),
                "first_prompt_tokens": receipts[0]["prompt_tokens"],
                "first_completion_tokens": receipts[0]["completion_tokens"],
                "repair_prompt_tokens": receipts[1]["prompt_tokens"] if n_calls == 2 else 0,
                "repair_completion_tokens": receipts[1]["completion_tokens"] if n_calls == 2 else 0,
                "seconds": event["value"]["seconds"],
                "escalated": event["value"].get("escalated", False)})
        assert offset == len(calls)
    return by_arm


def summarize_care(rows: list[dict]) -> dict:
    assert len(rows) == 1986 and len({r["id"] for r in rows}) == len(rows)
    repair = [r for r in rows if r["calls"] == 2]
    return {"questions": len(rows), "calls": sum(r["calls"] for r in rows),
        "repairs": len(repair), "prompt_tokens": describe([r["prompt_tokens"] for r in rows]),
        "completion_tokens": describe([r["completion_tokens"] for r in rows]),
        "total_tokens": describe([r["prompt_tokens"] + r["completion_tokens"] for r in rows]),
        "first_prompt_tokens": describe([r["first_prompt_tokens"] for r in rows]),
        "first_completion_tokens": describe([r["first_completion_tokens"] for r in rows]),
        "repair_prompt_tokens_on_repaired": describe([r["repair_prompt_tokens"] for r in repair]),
        "repair_completion_tokens_on_repaired": describe([r["repair_completion_tokens"] for r in repair]),
        "max_prompt_tokens_per_call": max(max(r["first_prompt_tokens"], r["repair_prompt_tokens"])
                                          for r in rows),
        "max_total_tokens_per_call": max(max(
            r["first_prompt_tokens"] + r["first_completion_tokens"],
            r["repair_prompt_tokens"] + r["repair_completion_tokens"]) for r in rows),
        "prompt_length_squared_sum": sum(r["prompt_length_squared_sum"] for r in rows),
        "seconds": describe([r["seconds"] for r in rows])}


def baseline_rows(report: str, arm: str) -> list[dict]:
    data = json.loads((ROOT / f"build/care/{report}").read_text(encoding="utf-8"))
    return [row["systems"][arm] for row in data["rows"]]


def summarize_baseline(rows: list[dict]) -> dict:
    assert len(rows) == 1986
    assert all(r["prompt_tokens_server"] is not None and r["completion_tokens"] is not None
               for r in rows)
    assert all(r["prompt_tokens_local"] == r["prompt_tokens_server"] for r in rows)
    return {"questions": len(rows), "calls": sum(r["new_calls"] for r in rows),
        "prompt_tokens": describe([r["prompt_tokens_server"] for r in rows]),
        "completion_tokens": describe([r["completion_tokens"] for r in rows]),
        "total_tokens": describe([r["prompt_tokens_server"] + r["completion_tokens"] for r in rows]),
        "prompt_length_squared_sum": sum(r["prompt_tokens_server"] ** 2 for r in rows),
        "seconds": describe([r["seconds"] for r in rows])}


def main() -> None:
    care = care_events()
    result = {
        "protocol": "care-resource-audit-v1",
        "care": summarize_care(care["bridge_v2"]),
        "relation_temporal": summarize_care(care["relation_temporal"]),
        "sim_retrieval": summarize_baseline(baseline_rows(
            "sim_retrieval_merged.json", "sim_retrieval")),
        "full_context_8k": summarize_baseline(baseline_rows(
            "context_baselines_merged.json", "full_context_8k")),
        "no_memory": summarize_baseline(baseline_rows(
            "context_baselines_merged.json", "no_memory")),
        "latency_scope": "CARE includes per-question retrieval and model API calls; "
            "SimRetrieval excludes corpus embedding and query retrieval; context baselines "
            "exclude fixed-context construction. Model setup is excluded from all arms.",
        "kaggle_receipt_sha256": {
            f"shard_{i}": {kind: hashlib.sha256((AUDIT / f"care_shard_{i}_{name}")
                                     .read_bytes()).hexdigest()
                           for kind, name in (("calls", "calls.json"),
                                              ("events", "events.jsonl"))}
            for i in range(5)},
    }
    assert result["care"]["calls"] == 2990
    assert result["care"]["repairs"] == 1004
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
