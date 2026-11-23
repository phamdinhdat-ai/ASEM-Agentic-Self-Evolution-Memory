"""Build a Kaggle BERTScore notebook for full NoMemory/FullContext-8K predictions."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MERGED = ROOT / "build/care/context_baselines_merged.json"
DATASET = ROOT / "datasets/locomo/locomo10.json"
TEMPLATE = ROOT / "build/kaggle/care-bertscore-full/CARE_Full_BERTScore.ipynb"
OUTPUT = ROOT / "build/kaggle/care-context-bertscore-full"


def main() -> None:
    merged = json.loads(MERGED.read_text(encoding="utf-8"))
    if merged["n_questions"] != 1986 or merged["n_predictions"] != 3972:
        raise ValueError("Context baselines are not complete")
    data_sha = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    if merged["dataset_sha256"] != data_sha:
        raise ValueError("Context baseline dataset mismatch")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    gold = {f"{sample['sample_id']}:{i}": qa for sample in dataset
            for i, qa in enumerate(sample["qa"])}
    records = []
    for row in merged["rows"]:
        qa = gold[row["id"]]
        category = int(qa["category"])
        reference = ("Not mentioned" if category == 5 else str(qa["answer"]).strip())
        if category == 3:
            reference = reference.split(";", 1)[0].strip()
        for arm in ("no_memory", "full_context_8k"):
            value = row["systems"][arm]
            records.append({
                "id": row["id"], "sample_id": row["sample_id"],
                "category": category, "arm": arm,
                "prediction": value["answer"], "reference": reference,
                "locomo_score": value["score"], "error": value["error"],
            })
    if len(records) != 3972 or len({(r["id"], r["arm"]) for r in records}) != 3972:
        raise ValueError("Duplicate or missing baseline predictions")
    payload = {
        "protocol": "care-context-baselines-posthoc-bertscore-v1",
        "dataset_sha256": data_sha,
        "merged_sha256": hashlib.sha256(MERGED.read_bytes()).hexdigest(),
        "source_receipts": merged["shards"],
        "reference_policy": "category 1-4 gold (category 3 before semicolon); category 5 standard Not mentioned",
        "records": records,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    source, count = re.subn(
        r"payload_bytes = gzip.decompress\(base64.b64decode\('[^']+'\)\)",
        f"payload_bytes = gzip.decompress(base64.b64decode({encoded!r}))",
        source)
    if count != 1:
        raise ValueError("BERTScore notebook payload template changed")
    old_hash = re.search(
        r"assert hashlib\.sha256\(payload_bytes\)\.hexdigest\(\) == '([a-f0-9]{64})'",
        source)
    if old_hash is None:
        raise ValueError("BERTScore notebook hash template changed")
    source = source.replace(old_hash.group(1), digest)
    if source.count("care_bertscore_full.json") != 1:
        raise ValueError("BERTScore notebook output name changed")
    source = source.replace("care_bertscore_full.json", "care_context_bertscore_full.json")
    notebook["cells"][0]["source"] = source.splitlines(keepends=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "CARE_Context_BERTScore.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": "phamdinhdatai/asem-care-context-bertscore-full",
        "title": "ASEM CARE Context BERTScore Full",
        "code_file": path.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [],
    }
    (OUTPUT / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT / "input-manifest.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "records"}
                   | {"payload_sha256": digest, "n_predictions": len(records)},
                   indent=2), encoding="utf-8")
    print(json.dumps({"notebook": str(path), "payload_sha256": digest,
                      "n_predictions": len(records), "notebook_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
