"""Audit paired CARE, NoMemory, FullContext-8K, SimRetrieval metrics."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "analysis/ASEM_CARE_four_way_20261112.md"
PATHS = {
    "care": ROOT / "build/care/relation_full_merged.json",
    "context": ROOT / "build/care/context_baselines_merged.json",
    "sim": ROOT / "build/care/sim_retrieval_merged.json",
    "care_bert": ROOT / "build/care/relation_full_bertscore.json",
    "context_bert": ROOT / "build/care/context_baselines_bertscore.json",
    "sim_bert": ROOT / "build/care/sim_retrieval_bertscore.json",
    "context_manifest": ROOT / "build/kaggle/care-context-bertscore-full/input-manifest.json",
    "sim_manifest": ROOT / "build/kaggle/care-sim-retrieval-bertscore-full/input-manifest.json",
}
ARMS = ("bridge_v2", "no_memory", "full_context_8k", "sim_retrieval")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ci(deltas, samples):
    groups = defaultdict(list)
    for delta, sample in zip(deltas, samples):
        groups[sample].append(delta)
    keys = list(groups)
    rng = random.Random(20261112)
    draws = []
    for _ in range(10000):
        values = [value for _ in keys for value in groups[rng.choice(keys)]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return draws[250], draws[9750]


def main():
    data = {name: read(path) for name, path in PATHS.items()}
    dataset_sha = sha(ROOT / "datasets/locomo/locomo10.json")
    for name in ("care", "context", "sim"):
        if data[name]["dataset_sha256"] != dataset_sha or data[name]["n_questions"] != 1986:
            raise ValueError(f"Dataset/coverage mismatch: {name}")
    for name in ("context", "sim"):
        bert = data[name + "_bert"]
        manifest = data[name + "_manifest"]
        if (bert["dataset_sha256"] != dataset_sha
                or bert["merged_sha256"] != sha(PATHS[name])
                or bert["payload_sha256"] != manifest["payload_sha256"]
                or bert["source_receipts"] != data[name]["shards"]):
            raise ValueError(f"BERTScore identity mismatch: {name}")
        for key in ("model", "model_revision", "layers", "idf",
                    "use_fast_tokenizer", "baseline_sha256", "baseline_values", "versions"):
            if bert[key] != data["care_bert"][key]:
                raise ValueError(f"BERTScore environment mismatch: {name} {key}")
    answers = {arm: {} for arm in ARMS}
    for index in range(5):
        path = (ROOT / "build/care/relation_full" / f"shard-{index}" /
                f"care_relation_full_s{index}" / "results/report.json")
        for row in read(path)["rows"]:
            answers["bridge_v2"][row["id"]] = (row["sample_id"], row["category"],
                                                row["systems"]["bridge_v2"]["f1"])
    for row in data["context"]["rows"]:
        for arm in ("no_memory", "full_context_8k"):
            answers[arm][row["id"]] = (row["sample_id"], row["category"],
                                       row["systems"][arm]["score"])
    for row in data["sim"]["rows"]:
        answers["sim_retrieval"][row["id"]] = (row["sample_id"], row["category"],
                                                row["systems"]["sim_retrieval"]["score"])
    ids = set(answers["bridge_v2"])
    if len(ids) != 1986 or any(set(value) != ids for value in answers.values()):
        raise ValueError("Four-way question IDs differ")
    bert = {}
    for source in ("care_bert", "context_bert", "sim_bert"):
        for record in data[source]["records"]:
            key = record["arm"], record["id"]
            if key in bert or record["id"] not in ids:
                raise ValueError("Duplicate/foreign BERTScore record")
            bert[key] = record
    if any(len([key for key in bert if key[0] == arm]) != 1986 for arm in ARMS):
        raise ValueError("Four-way BERTScore coverage mismatch")
    gold = {f"{sample['sample_id']}:{i}": qa
            for sample in read(ROOT / "datasets/locomo/locomo10.json")
            for i, qa in enumerate(sample["qa"])}
    for source, arms in (("context", ("no_memory", "full_context_8k")),
                         ("sim", ("sim_retrieval",))):
        for row in data[source]["rows"]:
            qa = gold[row["id"]]
            category = int(qa["category"])
            reference = "Not mentioned" if category == 5 else str(qa["answer"]).strip()
            if category == 3:
                reference = reference.split(";", 1)[0].strip()
            for arm in arms:
                record = bert[(arm, row["id"])]
                value = row["systems"][arm]
                if (record["prediction"] != value["answer"]
                        or record["reference"] != reference
                        or record["locomo_score"] != value["score"]):
                    raise ValueError(f"BERTScore prediction/reference mismatch {arm} {row['id']}")
    summary = {}
    for group, selected in (("all", ids),
                            ("answerable", {qid for qid in ids if answers["bridge_v2"][qid][1] != 5}),
                            ("adversarial", {qid for qid in ids if answers["bridge_v2"][qid][1] == 5})):
        summary[group] = {}
        ordered = sorted(selected)
        for arm in ARMS:
            locomo = [answers[arm][qid][2] for qid in ordered]
            raw = [bert[(arm, qid)]["bertscore"]["raw"]["f1"] for qid in ordered]
            rescaled = [bert[(arm, qid)]["bertscore"]["rescaled"]["f1"] for qid in ordered]
            summary[group][arm] = {"n": len(ordered), "locomo": sum(locomo) / len(locomo),
                                   "bertscore_raw": sum(raw) / len(raw),
                                   "bertscore_rescaled": sum(rescaled) / len(rescaled)}
            if arm != "bridge_v2":
                delta = [answers["bridge_v2"][qid][2] - answers[arm][qid][2] for qid in ordered]
                bdelta = [bert[("bridge_v2", qid)]["bertscore"]["raw"]["f1"] -
                          bert[(arm, qid)]["bertscore"]["raw"]["f1"] for qid in ordered]
                samples = [answers["bridge_v2"][qid][0] for qid in ordered]
                summary[group][arm]["care_minus_baseline"] = sum(delta) / len(delta)
                summary[group][arm]["care_minus_baseline_ci95"] = ci(delta, samples)
                summary[group][arm]["bertscore_delta"] = sum(bdelta) / len(bdelta)
                summary[group][arm]["bertscore_delta_ci95"] = ci(bdelta, samples)
    lines = ["# CARE versus FullContext-8K, NoMemory, SimRetrieval", "",
             "All four arms use the same 1,986 LoCoMo questions, pinned Qwen3-8B, "
             "8,192-token window, 224-token answer cap, temperature 0, no thinking, "
             "and identical LoCoMo and BERTScore scorers. CARE results were preserved.", "",
             "| Group | Arm | n | LoCoMo | BERTScore raw F1 | BERTScore rescaled F1 | CARE minus arm LoCoMo |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group in ("all", "answerable", "adversarial"):
        for arm in ARMS:
            s = summary[group][arm]
            delta = "--" if arm == "bridge_v2" else f"{s['care_minus_baseline']:+.4f}"
            lines.append(f"| {group} | {arm} | {s['n']} | {s['locomo']:.4f} | "
                         f"{s['bertscore_raw']:.4f} | {s['bertscore_rescaled']:.4f} | {delta} |")
    lines += ["", "## Protocol limits", "",
              "FullContext-8K keeps the first five turns and a recent suffix; its middle turns "
              "are omitted to fit the context window. SimRetrieval uses top-18 cosine matches "
              "from original dated turns with MiniLM-L6-v2, a 10,000-character evidence cap "
              "and one answer call. CARE uses its own retrieval, source checks and optional repair. "
              "NoMemory follows the prior best-guess prompt. These systems have different "
              "answer prompts, so score gaps measure whole pipelines, not the isolated memory "
              "mechanism. Category-5 refusal policy especially differs for NoMemory.", "",
              "BERTScore measures semantic overlap, not factual correctness. The LoCoMo overall "
              "score combines token F1 for categories 1-4 with binary refusal accuracy for category 5.", "",
              "## Files and integrity", ""]
    for name in ("care", "context", "sim", "care_bert", "context_bert", "sim_bert"):
        lines.append(f"- `{PATHS[name].relative_to(ROOT)}` SHA256 `{sha(PATHS[name])}`")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
