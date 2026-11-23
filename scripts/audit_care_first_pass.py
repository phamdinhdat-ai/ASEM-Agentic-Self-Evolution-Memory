"""Score the saved first CARE read against its final repaired answer.

This is a trace replay, not a new inference run or a matched retriever ablation.
Gold answers enter only this post-generation audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from scripts.run_care_locomo import score_locomo


def summarize(rows: list[dict]) -> dict:
    differences = [row["final_score"] - row["first_score"] for row in rows]
    return {
        "n": len(rows),
        "first_score": sum(row["first_score"] for row in rows) / len(rows),
        "final_score": sum(row["final_score"] for row in rows) / len(rows),
        "delta": sum(differences) / len(rows),
        "improved": sum(value > 1e-12 for value in differences),
        "worsened": sum(value < -1e-12 for value in differences),
        "tied": sum(abs(value) <= 1e-12 for value in differences),
    }


def audit(dataset_path: Path, report_paths: list[Path], *, seed: int = 20261113,
          draws: int = 10000) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    gold = {
        f"{conversation['sample_id']}:{index}": str(qa.get("answer", ""))
        for conversation in dataset for index, qa in enumerate(conversation["qa"])
    }
    expected = len(gold)
    source_rows = [row for path in report_paths
                   for row in json.loads(path.read_text(encoding="utf-8"))["rows"]]
    ids = [row["id"] for row in source_rows]
    if len(ids) != expected or len(set(ids)) != expected or set(ids) != set(gold):
        raise ValueError("CARE reports do not cover the locked questions exactly once")

    paired = []
    for row in source_rows:
        result = row["systems"]["bridge_v2"]
        first = (result.get("first_pass") or {}).get("answer", result["answer"])
        first_score = score_locomo(first, gold[row["id"]], row["category"])
        final_score = score_locomo(result["answer"], gold[row["id"]], row["category"])
        if abs(result["f1"] - final_score) > 1e-10:
            raise ValueError(f"Stored final score disagrees with scorer: {row['id']}")
        paired.append({"id": row["id"], "sample_id": row["sample_id"],
                       "category": row["category"],
                       "escalated": bool(result.get("escalated")),
                       "first_score": first_score, "final_score": final_score})

    clusters = defaultdict(list)
    for row in paired:
        clusters[row["sample_id"]].append(row)
    names = sorted(clusters)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        sampled = [row for _ in names for row in clusters[rng.choice(names)]]
        samples.append(summarize(sampled)["delta"])
    samples.sort()
    return {
        "protocol": "saved-first-pass-trace-replay-v1",
        "dataset": str(dataset_path),
        "reports": [str(path) for path in report_paths],
        "interpretation": "Within-CARE first versus final answer; not a causal retriever ablation",
        "all": summarize(paired),
        "repaired_only": summarize([r for r in paired if r["escalated"]]),
        "answerable": summarize([r for r in paired if r["category"] != 5]),
        "unsupported": summarize([r for r in paired if r["category"] == 5]),
        "by_category": {str(category): summarize([r for r in paired
                                                     if r["category"] == category])
                        for category in range(1, 6)},
        "conversation_cluster_bootstrap_95": [samples[int(.025 * draws)],
                                               samples[int(.975 * draws)]],
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/locomo/locomo10.json"))
    parser.add_argument("--reports", type=Path, nargs="*",
                        default=sorted(Path("build/care/relation_full").glob("**/report.json")))
    parser.add_argument("--output", type=Path,
                        default=Path("analysis/CARE_first_pass_audit_20261113.json"))
    args = parser.parse_args()
    if not args.reports:
        parser.error("No CARE shard reports found")
    result = audit(args.dataset, args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("all", "answerable", "unsupported",
                       "conversation_cluster_bootstrap_95")}, indent=2))


if __name__ == "__main__":
    main()
