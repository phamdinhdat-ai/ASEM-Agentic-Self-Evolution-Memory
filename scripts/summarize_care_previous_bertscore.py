"""Verify and compare CARE and previous methods under identical BERTScore settings."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "build/care/care_previous_comparison.json"
CARE = ROOT / "build/care/relation_full_bertscore.json"
OLD = ROOT / "build/care/care_previous_bertscore.json"
MANIFEST = ROOT / "build/kaggle/care-previous-bertscore/input-manifest.json"
OUTPUT = ROOT / "build/care/care_previous_bertscore_comparison.json"
REPORT = ROOT / "analysis/ASEM_CARE_vs_previous_20261112_bertscore.md"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(groups: dict[str, list[float]], seed: int = 20261112) -> list[float]:
    keys = list(groups)
    rng = random.Random(seed)
    values = []
    for _ in range(10_000):
        draw = [delta for _ in keys for delta in groups[rng.choice(keys)]]
        values.append(sum(draw) / len(draw))
    values.sort()
    return [values[250], values[9750]]


def main() -> None:
    comparison, care, old, manifest = (read(path) for path in
                                       (COMPARISON, CARE, OLD, MANIFEST))
    if (old["protocol"] != manifest["protocol"]
            or old["dataset_sha256"] != comparison["dataset_sha256"]
            or old["dataset_sha256"] != care["dataset_sha256"]
            or old["merged_sha256"] != comparison["merged_sha256"]
            or old["source_receipts"] != manifest["source_receipts"]
            or old["payload_sha256"] != manifest["payload_sha256"]
            or len(old["records"]) != manifest["n_predictions"] == 2620):
        raise ValueError("Old BERTScore input/output identity mismatch")
    for key in ("model", "model_revision", "layers", "idf", "use_fast_tokenizer",
                "baseline_sha256", "baseline_values", "versions"):
        if old[key] != care[key]:
            raise ValueError(f"CARE and previous BERTScore environments differ: {key}")
    care_scores = {(r["id"], r["arm"]): r for r in care["records"]}
    old_scores = {}
    original = {(r["id"], arm): r
                for arm, comparison_data in comparison["comparisons"].items()
                for r in comparison_data["rows"]}
    for row in old["records"]:
        key = row["id"], row["arm"]
        if key in old_scores or key not in original:
            raise ValueError(f"Duplicate or foreign old BERTScore row: {key}")
        expected = original[key]
        if (row["sample_id"] != expected["sample_id"]
                or row["category"] != expected["category"]
                or row["prediction"] != expected["old_answer"]
                or row["reference"] != expected["reference"]
                or row["locomo_score"] != expected["old_score"]):
            raise ValueError(f"Old BERTScore prediction mismatch: {key}")
        if any(not math.isfinite(value) for mode in row["bertscore"].values()
               for value in mode.values()):
            raise ValueError(f"Non-finite old BERTScore: {key}")
        old_scores[key] = row
    if set(old_scores) != set(original):
        raise ValueError("Old BERTScore coverage mismatch")
    summary = {}
    for arm in ("hybrid_attributed", "graph_memory", "bm25"):
        groups = {
            "all": comparison["comparisons"][arm]["rows"],
            "answerable": [r for r in comparison["comparisons"][arm]["rows"]
                           if r["category"] != 5],
            "adversarial": [r for r in comparison["comparisons"][arm]["rows"]
                            if r["category"] == 5],
        }
        summary[arm] = {}
        for group, rows in groups.items():
            diffs = defaultdict(list)
            wins = Counter()
            for row in rows:
                qid = row["id"]
                old_f1 = old_scores[(qid, arm)]["bertscore"]["raw"]["f1"]
                care_f1 = care_scores[(qid, "bridge_v2")]["bertscore"]["raw"]["f1"]
                delta = care_f1 - old_f1
                diffs[row["sample_id"]].append(delta)
                wins["win" if delta > 1e-8 else "loss" if delta < -1e-8 else "tie"] += 1
            summary[arm][group] = {
                "n": len(rows), "histories": len(diffs),
                "old_raw_f1": sum(old_scores[(r["id"], arm)]["bertscore"]["raw"]["f1"]
                                  for r in rows) / len(rows),
                "care_raw_f1": sum(care_scores[(r["id"], "bridge_v2")]["bertscore"]["raw"]["f1"]
                                   for r in rows) / len(rows),
                "old_rescaled_f1": sum(old_scores[(r["id"], arm)]["bertscore"]["rescaled"]["f1"]
                                       for r in rows) / len(rows),
                "care_rescaled_f1": sum(care_scores[(r["id"], "bridge_v2")]["bertscore"]["rescaled"]["f1"]
                                        for r in rows) / len(rows),
                "delta_raw_f1": sum(v for values in diffs.values() for v in values) / len(rows),
                "bootstrap_conversation_95": bootstrap(diffs), **wins}
    output = {"protocol": "care-vs-previous-bertscore-v1",
              "current_bertscore_sha256": sha(CARE),
              "previous_bertscore_sha256": sha(OLD),
              "comparison_sha256": sha(COMPARISON),
              "model": old["model"], "model_revision": old["model_revision"],
              "versions": old["versions"], "summary": summary}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# CARE Bridge v2 versus previous methods: paired BERTScore", "",
             "Both sides were rescored with the same pinned roberta-large revision, "
             "layer 17, no IDF, slow tokenizer and package versions. Raw predictions "
             "and original references were retained. Category 5 uses `Not mentioned` "
             "as the refusal reference and is reported separately.",
             "", "| Previous method | Group | n | Previous raw F1 | CARE raw F1 | Delta | 95% cluster bootstrap | CARE W/L/T |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for arm in ("hybrid_attributed", "graph_memory", "bm25"):
        for group in ("all", "answerable", "adversarial"):
            s = summary[arm][group]
            lo, hi = s["bootstrap_conversation_95"]
            lines.append(f"| {arm} | {group} | {s['n']} | {s['old_raw_f1']:.4f} | "
                         f"{s['care_raw_f1']:.4f} | {s['delta_raw_f1']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | "
                         f"{s.get('win', 0)}/{s.get('loss', 0)}/{s.get('tie', 0)} |")
    lines += ["", "## Baseline-rescaled F1", "",
              "| Previous method | Group | Previous rescaled F1 | CARE rescaled F1 |",
              "|---|---|---:|---:|"]
    for arm in ("hybrid_attributed", "graph_memory", "bm25"):
        for group in ("all", "answerable", "adversarial"):
            s = summary[arm][group]
            lines.append(f"| {arm} | {group} | {s['old_rescaled_f1']:.4f} | "
                         f"{s['care_rescaled_f1']:.4f} |")
    lines += ["", "BERTScore is semantic text similarity and does not validate facts, "
              "entity binding or citation support. Compare the answerable group "
              "and LoCoMo category-specific score alongside it. Old graph/BM25 "
              "cohorts are incomplete; the comparison does not isolate a single "
              "architectural component.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
