"""Audit full NoMemory/FullContext-8K results against CARE, including BERTScore."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/locomo/locomo10.json"
BASELINES = ROOT / "build/care/context_baselines_merged.json"
BASELINE_BERT = ROOT / "build/care/context_baselines_bertscore.json"
BASELINE_BERT_MANIFEST = ROOT / "build/kaggle/care-context-bertscore-full/input-manifest.json"
CARE_BERT = ROOT / "build/care/relation_full_bertscore.json"
CARE_MERGED = ROOT / "build/care/relation_full_merged.json"
OUTPUT = ROOT / "analysis/ASEM_CARE_vs_context_baselines_20261112.md"
ARMS = ("no_memory", "full_context_8k")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(groups: dict[str, list[float]], seed: int = 20261112) -> list[float]:
    keys = list(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(10_000):
        values = [x for _ in keys for x in groups[rng.choice(keys)]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return [draws[250], draws[9750]]


def paired(rows: list[dict], arm: str, metric: str,
           care_bert: dict | None = None, baseline_bert: dict | None = None) -> dict:
    grouped = defaultdict(list)
    scores = []
    for row in rows:
        qid = row["id"]
        if metric == "locomo":
            care_score = row["care_score"]
            baseline_score = row["systems"][arm]["score"]
        else:
            care_score = care_bert[(qid, "bridge_v2")]["bertscore"]["raw"]["f1"]
            baseline_score = baseline_bert[(qid, arm)]["bertscore"]["raw"]["f1"]
        grouped[row["sample_id"]].append(care_score - baseline_score)
        scores.append((care_score, baseline_score))
    deltas = [x - y for x, y in scores]
    return {
        "n": len(rows),
        "care_mean": sum(x for x, _ in scores) / len(scores),
        "baseline_mean": sum(y for _, y in scores) / len(scores),
        "delta": sum(deltas) / len(deltas),
        "bootstrap_conversation_95": bootstrap(grouped),
        "wins": sum(d > 1e-8 for d in deltas),
        "losses": sum(d < -1e-8 for d in deltas),
        "ties": sum(abs(d) <= 1e-8 for d in deltas),
    }


def main() -> None:
    baseline = read(BASELINES)
    baseline_bert = read(BASELINE_BERT)
    bert_manifest = read(BASELINE_BERT_MANIFEST)
    care_bert = read(CARE_BERT)
    care_merged = read(CARE_MERGED)
    if (baseline["n_questions"] != 1986
            or care_merged["n_questions"] != 1986
            or baseline["dataset_sha256"] != care_merged["dataset_sha256"]
            or baseline["dataset_sha256"] != sha(DATASET)):
        raise ValueError("Dataset/coverage mismatch")
    if (baseline_bert["protocol"] != bert_manifest["protocol"]
            or baseline_bert["merged_sha256"] != bert_manifest["merged_sha256"]
            or baseline_bert["merged_sha256"] != sha(BASELINES)
            or baseline_bert["dataset_sha256"] != baseline["dataset_sha256"]
            or baseline_bert["source_receipts"] != baseline["shards"]
            or baseline_bert["payload_sha256"] != bert_manifest["payload_sha256"]
            or len(baseline_bert["records"]) != 3972):
        raise ValueError("Baseline BERTScore identity/coverage mismatch")
    for key in ("model", "model_revision", "layers", "idf", "use_fast_tokenizer",
                "baseline_sha256", "baseline_values", "versions"):
        if baseline_bert[key] != care_bert[key]:
            raise ValueError(f"BERTScore environments differ: {key}")
    gold = {f"{sample['sample_id']}:{i}": qa for sample in read(DATASET)
            for i, qa in enumerate(sample["qa"])}
    baseline_scores = {}
    for record in baseline_bert["records"]:
        key = record["id"], record["arm"]
        if key in baseline_scores or record["arm"] not in ARMS:
            raise ValueError("Duplicate/foreign baseline BERTScore record")
        baseline_scores[key] = record
    care_scores = {(r["id"], r["arm"]): r for r in care_bert["records"]}
    if len(baseline_scores) != 3972 or len(care_scores) != 3972:
        raise ValueError("BERTScore coverage mismatch")
    care_rows = {}
    for index in range(5):
        path = (ROOT / "build/care/relation_full" / f"shard-{index}"
                / f"care_relation_full_s{index}" / "results/report.json")
        for row in read(path)["rows"]:
            care_rows[row["id"]] = row
    rows = []
    for row in baseline["rows"]:
        qid = row["id"]
        qa = gold[qid]
        category = int(qa["category"])
        reference = ("Not mentioned" if category == 5 else str(qa["answer"]).strip())
        if category == 3:
            reference = reference.split(";", 1)[0].strip()
        if (qid not in care_rows or row["category"] != category
                or row["question"] != qa["question"]):
            raise ValueError(f"Question mismatch: {qid}")
        for arm in ARMS:
            score = baseline_scores[(qid, arm)]
            value = row["systems"][arm]
            if (score["prediction"] != value["answer"]
                    or score["reference"] != reference
                    or score["locomo_score"] != value["score"]):
                raise ValueError(f"Baseline BERTScore answer/reference mismatch: {qid} {arm}")
        rows.append({**row, "care_score": care_rows[qid]["systems"]["bridge_v2"]["f1"]})
    groups = {
        "all": rows,
        "answerable": [r for r in rows if r["category"] != 5],
        "adversarial": [r for r in rows if r["category"] == 5],
    }
    summary = {
        arm: {group: {
            "locomo": paired(subset, arm, "locomo"),
            "bertscore": paired(subset, arm, "bertscore", care_scores, baseline_scores),
        } for group, subset in groups.items()} for arm in ARMS
    }
    coverage = baseline["context_coverage"]
    included = sum(x["included_turns"] for x in coverage.values())
    total = sum(x["total_turns"] for x in coverage.values())
    truncation = {
        "n_conversations": len(coverage),
        "total_turns": total, "included_turns": included,
        "turn_fraction": included / total,
        "all_truncated": all(x["truncated"] for x in coverage.values()),
        "min_fraction": min(x["included_turns"] / x["total_turns"] for x in coverage.values()),
        "max_fraction": max(x["included_turns"] / x["total_turns"] for x in coverage.values()),
    }
    lines = [
        "# CARE Bridge v2 versus Qwen3-8B NoMemory and FullContext-8K",
        "",
        "All methods answered the same 1,986 LoCoMo questions with pinned Qwen3-8B. "
        "The NoMemory prompt follows the prior best-guess baseline. FullContext-8K "
        "uses the first five dated turns plus the longest recent suffix that fits "
        "the 8,192-token model window; it is not an untruncated full-history result.",
        "",
        "| Arm | Group | n | Baseline LoCoMo | CARE LoCoMo | Delta | Baseline BERTScore raw F1 | CARE BERTScore raw F1 | Delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        for group in ("all", "answerable", "adversarial"):
            l, b = summary[arm][group]["locomo"], summary[arm][group]["bertscore"]
            lines.append(f"| {arm} | {group} | {l['n']} | {l['baseline_mean']:.4f} | "
                         f"{l['care_mean']:.4f} | {l['delta']:+.4f} | "
                         f"{b['baseline_mean']:.4f} | {b['care_mean']:.4f} | "
                         f"{b['delta']:+.4f} |")
    lines += [
        "", "## FullContext-8K coverage", "",
        f"{included}/{total} conversation turns retained across 10 conversations "
        f"({included / total:.1%}); all conversations were truncated. "
        f"Per-conversation range: {truncation['min_fraction']:.1%}–"
        f"{truncation['max_fraction']:.1%}.",
        "",
        "## Per-category LoCoMo score", "",
        "| Category | n | CARE | NoMemory | FullContext-8K |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cat in range(1, 6):
        subset = [r for r in rows if r["category"] == cat]
        lines.append(f"| {cat} | {len(subset)} | "
                     f"{sum(r['care_score'] for r in subset) / len(subset):.4f} | "
                     f"{baseline['by_category'][str(cat)]['no_memory']['mean_score']:.4f} | "
                     f"{baseline['by_category'][str(cat)]['full_context_8k']['mean_score']:.4f} |")
    lines += [
        "", "## Interpretation", "",
        "- The LoCoMo overall score combines answerable token F1 and binary "
        "category-5 refusal accuracy; compare those groups separately.",
        "- BERTScore is semantic similarity, not factual correctness or citation support.",
        "- NoMemory's old-style best-guess prompt and CARE's refusal contract differ. "
        "A category-5 gap partly reflects that contract, not only memory ability.",
        "- FullContext-8K omits most of each long conversation. This experiment "
        "does not establish how CARE compares with a truly untruncated "
        "long-context model.",
        "",
        f"Baseline merge SHA256: `{sha(BASELINES)}`. "
        f"Baseline BERTScore SHA256: `{sha(BASELINE_BERT)}`.",
        "",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "summary": summary,
                      "context_coverage": truncation}, indent=2))


if __name__ == "__main__":
    main()
