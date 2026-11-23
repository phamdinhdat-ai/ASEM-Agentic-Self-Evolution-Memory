"""Verify Kaggle post-hoc BERTScore output and summarize the full CARE run."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/care/relation_full"
MERGED = ROOT / "build/care/relation_full_merged.json"
DATASET = ROOT / "datasets/locomo/locomo10.json"
INPUT = ROOT / "build/kaggle/care-bertscore-full/input-manifest.json"
SCORES = ROOT / "build/care/relation_full_bertscore.json"
SUMMARY = ROOT / "analysis/ASEM_CARE_full_20261112.md"
ARMS = ("bridge_v2", "relation_temporal")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap(deltas: dict[str, list[float]], seed: int = 20261112) -> tuple[float, float]:
    keys = list(deltas)
    rng = random.Random(seed)
    draws = []
    for _ in range(10_000):
        values = [value for _ in keys for value in deltas[rng.choice(keys)]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return draws[250], draws[9750]


def main() -> None:
    report, source, manifest = read(SCORES), read(MERGED), read(INPUT)
    assert report["protocol"] == "care-full-locomo-posthoc-bertscore-v1"
    assert report["model"] == "roberta-large"
    assert report["model_revision"] == "722cf37b1afa9454edce342e7895e588b6ff1d59"
    assert report["layers"] == 17 and report["idf"] is False
    assert report["dataset_sha256"] == source["dataset_sha256"] == sha(DATASET)
    assert report["merged_sha256"] == manifest["merged_sha256"] == sha(MERGED)
    assert report["source_receipts"] == manifest["source_receipts"]
    assert report["payload_sha256"] == manifest["payload_sha256"]
    assert len(report["records"]) == manifest["n_predictions"] == 3972
    lookup = {f"{sample['sample_id']}:{i}": qa for sample in read(DATASET)
              for i, qa in enumerate(sample["qa"])}
    original = {}
    for receipt in manifest["source_receipts"]:
        index = receipt["shard"]
        path = RUN / f"shard-{index}" / f"care_relation_full_s{index}" / "results/report.json"
        assert sha(path) == receipt["sha256"]
        rows = read(path)["rows"]
        assert len(rows) == receipt["n"]
        for row in rows:
            original[row["id"]] = row
    assert len(original) == 1986
    by_pair = {}
    for row in report["records"]:
        key = (row["id"], row["arm"])
        if key in by_pair or row["arm"] not in ARMS:
            raise ValueError("Duplicate or unexpected prediction")
        qa = lookup[row["id"]]
        category = int(qa["category"])
        reference = ("Not mentioned" if category == 5 else str(qa["answer"]).strip())
        if category == 3:
            reference = reference.split(";", 1)[0].strip()
        expected = original[row["id"]]["systems"][row["arm"]]
        if (row["sample_id"] != original[row["id"]]["sample_id"]
                or row["category"] != category
                or row["reference"] != reference
                or row["prediction"] != expected["answer"]
                or row["locomo_score"] != expected["f1"]
                or row["error"] != expected["error"]):
            raise ValueError(f"Prediction/reference mismatch: {key}")
        if any(not math.isfinite(value) for mode in row["bertscore"].values()
               for value in mode.values()):
            raise ValueError(f"Non-finite BERTScore: {key}")
        by_pair[key] = row
    if set(by_pair) != {(qid, arm) for qid in original for arm in ARMS}:
        raise ValueError("Incomplete BERTScore coverage")
    groups = defaultdict(list)
    for row in report["records"]:
        for group in ("all", "answerable" if row["category"] != 5 else "adversarial",
                      f"category_{row['category']}"):
            groups[(row["arm"], group)].append(row)
    aggregates = {}
    for (arm, group), rows in groups.items():
        aggregates[(arm, group)] = {
            "n": len(rows),
            "locomo": sum(r["locomo_score"] for r in rows) / len(rows),
            "raw_f1": sum(r["bertscore"]["raw"]["f1"] for r in rows) / len(rows),
            "rescaled_f1": sum(r["bertscore"]["rescaled"]["f1"] for r in rows) / len(rows),
        }
    for item in report["aggregate"]:
        computed = aggregates[(item["arm"], item["group"])]
        if (item["n"] != computed["n"]
                or abs(item["locomo_mean"] - computed["locomo"]) > 1e-8
                or abs(item["raw"]["f1"] - computed["raw_f1"]) > 1e-8
                or abs(item["rescaled"]["f1"] - computed["rescaled_f1"]) > 1e-8):
            raise ValueError(f"Aggregate mismatch: {item['arm']} {item['group']}")
    paired = {}
    for group_name, predicate in (
        ("all", lambda category: True),
        ("answerable", lambda category: category != 5),
        ("adversarial", lambda category: category == 5),
    ):
        diffs = defaultdict(list)
        wins = Counter()
        for qid, row in original.items():
            if not predicate(row["category"]):
                continue
            baseline = by_pair[(qid, "bridge_v2")]["bertscore"]["raw"]["f1"]
            treatment = by_pair[(qid, "relation_temporal")]["bertscore"]["raw"]["f1"]
            delta = treatment - baseline
            diffs[row["sample_id"]].append(delta)
            wins["win" if delta > 1e-8 else "loss" if delta < -1e-8 else "tie"] += 1
        flat = [x for values in diffs.values() for x in values]
        paired[group_name] = {"n": len(flat), "delta": sum(flat) / len(flat),
                              "bootstrap_95": bootstrap(diffs), **wins}
    altered = [qid for qid, row in original.items()
               if row["systems"]["bridge_v2"]["answer"]
               != row["systems"]["relation_temporal"]["answer"]]
    lines = [
        "# CARE full LoCoMo: final results and post-hoc BERTScore",
        "",
        "All five Kaggle shards completed. The sealed merge covers 10 conversations, "
        "1,986 questions and 3,972 question-arm predictions, with zero inference errors.",
        "This is a text plus image-caption adapter, not an official multimodal run.",
        "",
        "BERTScore uses roberta-large at pinned revision "
        "`722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, no IDF, "
        "slow tokenizer and the official bert-score package. It is a post-hoc "
        "semantic similarity metric; it does not verify factual support or citations. "
        "Category 5 uses the standard `Not mentioned` reference and is reported separately. "
        "Category 3 uses gold text before the semicolon, as in the LoCoMo scorer.",
        "",
        "| Group | n | Arm | LoCoMo score | BERTScore raw F1 | BERTScore rescaled F1 |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for group in ("all", "answerable", "adversarial"):
        for arm in ARMS:
            value = aggregates[(arm, group)]
            lines.append(f"| {group} | {value['n']} | {arm} | {value['locomo']:.4f} | "
                         f"{value['raw_f1']:.4f} | {value['rescaled_f1']:.4f} |")
    lines += ["", "## By category", "",
              "| Category | n | Arm | LoCoMo score | BERTScore raw F1 | BERTScore rescaled F1 |",
              "|---|---:|---|---:|---:|---:|"]
    for category in range(1, 6):
        for arm in ARMS:
            value = aggregates[(arm, f"category_{category}")]
            lines.append(f"| {category} | {value['n']} | {arm} | {value['locomo']:.4f} | "
                         f"{value['raw_f1']:.4f} | {value['rescaled_f1']:.4f} |")
    lines += ["", "## Paired BERTScore raw F1, Relation+Temporal minus Bridge v2", "",
              "| Group | n | Mean delta | Conversation bootstrap 95% | Wins | Losses | Ties |",
              "|---|---:|---:|---|---:|---:|---:|"]
    for group in ("all", "answerable", "adversarial"):
        value = paired[group]
        lo, hi = value["bootstrap_95"]
        lines.append(f"| {group} | {value['n']} | {value['delta']:+.5f} | "
                     f"[{lo:+.5f}, {hi:+.5f}] | {value.get('win', 0)} | "
                     f"{value.get('loss', 0)} | {value.get('tie', 0)} |")
    lines += ["", f"Answers differ for {len(altered)} of 1,986 paired questions.",
              "The LoCoMo aggregate mixes token F1 for categories 1-4 and binary "
              "refusal accuracy for category 5. Interpret answerable F1 and "
              "adversarial accuracy separately.", "",
              f"Metric environment: {report['versions']}.",
              f"Input payload SHA256: `{report['payload_sha256']}`.",
              f"Kaggle output SHA256: `{sha(SCORES)}`.",
              ""]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY), "aggregate": {
        f"{arm}/{group}": value for (arm, group), value in aggregates.items()
        if group in ("all", "answerable", "adversarial")},
        "paired": paired, "altered_answers": len(altered)}, indent=2))


if __name__ == "__main__":
    main()
