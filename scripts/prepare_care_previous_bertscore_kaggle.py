"""Build a Kaggle BERTScore notebook for saved previous-method predictions."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "build/care/care_previous_comparison.json"
TEMPLATE = ROOT / "build/kaggle/care-bertscore-full/CARE_Full_BERTScore.ipynb"
OUTPUT = ROOT / "build/kaggle/care-previous-bertscore"


def main() -> None:
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    records = []
    for arm in ("hybrid_attributed", "graph_memory", "bm25"):
        for row in comparison["comparisons"][arm]["rows"]:
            records.append({
                "id": row["id"], "sample_id": row["sample_id"],
                "category": row["category"], "arm": arm,
                "prediction": row["old_answer"], "reference": row["reference"],
                "locomo_score": row["old_score"], "error": None,
            })
    if len(records) != 2620 or len({(r["id"], r["arm"]) for r in records}) != len(records):
        raise ValueError("Expected 2,620 unique previous-method predictions")
    payload = {
        "protocol": "care-previous-locomo-posthoc-bertscore-v1",
        "dataset_sha256": comparison["dataset_sha256"],
        "merged_sha256": comparison["merged_sha256"],
        "source_receipts": {
            "graph": comparison["graph_receipts"],
            "bm25": comparison["bm25_receipts"],
            "comparison_sha256": hashlib.sha256(COMPARISON.read_bytes()).hexdigest(),
        },
        "reference_policy": "same category-specific references as CARE full BERTScore",
        "records": records,
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode("ascii")
    notebook = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    source, n = re.subn(
        r"payload_bytes = gzip.decompress\(base64.b64decode\('[^']+'\)\)",
        f"payload_bytes = gzip.decompress(base64.b64decode({encoded!r}))",
        source)
    if n != 1:
        raise ValueError("BERTScore notebook payload template changed")
    old_hash = re.search(r"assert hashlib\.sha256\(payload_bytes\)\.hexdigest\(\) == '([a-f0-9]{64})'", source)
    if old_hash is None:
        raise ValueError("BERTScore notebook hash template changed")
    source = source.replace(old_hash.group(1), digest)
    checks = ("assert len(records) == 3972",
              "assert len({(r[\"id\"], r[\"arm\"]) for r in records}) == 3972")
    if any(source.count(check) != 1 for check in checks):
        raise ValueError("BERTScore notebook coverage checks changed")
    for check in checks:
        source = source.replace(check, check.replace("3972", str(len(records))))
    if source.count("care_bertscore_full.json") != 1:
        raise ValueError("BERTScore notebook output name changed")
    source = source.replace("care_bertscore_full.json", "care_previous_bertscore.json")
    notebook["cells"][0]["source"] = source.splitlines(keepends=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "CARE_Previous_BERTScore.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": "phamdinhdatai/asem-care-previous-bertscore",
        "title": "ASEM CARE Previous BERTScore",
        "code_file": "CARE_Previous_BERTScore.ipynb",
        "language": "python", "kernel_type": "notebook",
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
    print(json.dumps({"notebook": str(OUTPUT / "CARE_Previous_BERTScore.ipynb"),
                      "n_predictions": len(records), "payload_sha256": digest,
                      "notebook_bytes": (OUTPUT / "CARE_Previous_BERTScore.ipynb").stat().st_size}))


if __name__ == "__main__":
    main()
