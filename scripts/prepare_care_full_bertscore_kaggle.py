"""Prepare post-hoc BERTScore over the sealed full CARE LoCoMo predictions."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path
import textwrap

from scripts.merge_care_full_locomo import merge


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/locomo/locomo10.json"
RUN = ROOT / "build/care/relation_full"
MERGED = ROOT / "build/care/relation_full_merged.json"
OUTPUT = ROOT / "build/kaggle/care-bertscore-full"
MODEL = "roberta-large"
REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    verified = merge(DATASET, RUN)
    saved = json.loads(MERGED.read_text(encoding="utf-8"))
    for key in ("dataset_sha256", "n_questions", "n_predictions", "category_counts", "aggregate"):
        if saved[key] != verified[key]:
            raise ValueError(f"Merged report mismatch: {key}")
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    lookup = {f"{sample['sample_id']}:{index}": qa
              for sample in dataset for index, qa in enumerate(sample["qa"])}
    records = []
    receipts = []
    for shard in range(5):
        path = RUN / f"shard-{shard}" / f"care_relation_full_s{shard}" / "results/report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        receipts.append({"shard": shard, "sha256": sha(path), "n": len(report["rows"])})
        for row in report["rows"]:
            qa = lookup[row["id"]]
            category = int(qa["category"])
            reference = ("Not mentioned" if category == 5
                         else str(qa["answer"]).strip())
            if category == 3:
                reference = reference.split(";", 1)[0].strip()
            if not reference:
                raise ValueError(f"Empty reference: {row['id']}")
            for arm in ("bridge_v2", "relation_temporal"):
                value = row["systems"][arm]
                records.append({
                    "id": row["id"],
                    "sample_id": row["sample_id"],
                    "category": category,
                    "arm": arm,
                    "prediction": value["answer"],
                    "reference": reference,
                    "locomo_score": value["f1"],
                    "error": value["error"],
                })
    if len(records) != 3972 or len({(r["id"], r["arm"]) for r in records}) != len(records):
        raise ValueError("Expected exactly 3,972 unique question-arm predictions")
    payload = {
        "protocol": "care-full-locomo-posthoc-bertscore-v1",
        "dataset_sha256": sha(DATASET),
        "merged_sha256": sha(MERGED),
        "source_receipts": receipts,
        "reference_policy": "category 1-4 original gold (category 3 before semicolon); category 5 uses standard Not mentioned reference",
        "records": records,
    }
    encoded = base64.b64encode(gzip.compress(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        mtime=0)).decode("ascii")
    payload_sha = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    source = textwrap.dedent('''
        import base64, gzip, hashlib, json, math, subprocess, sys, time
        from collections import defaultdict
        from pathlib import Path
        from importlib.metadata import version

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "bert-score==0.3.13", "transformers==5.8.0",
                               "huggingface-hub==1.14.0"])
        import bert_score
        import torch
        from bert_score import BERTScorer
        from huggingface_hub import snapshot_download

        started = time.monotonic()
        payload_bytes = gzip.decompress(base64.b64decode(__PAYLOAD__))
        assert hashlib.sha256(payload_bytes).hexdigest() == __PAYLOAD_SHA__
        payload = json.loads(payload_bytes)
        records = payload["records"]
        assert len(records) == 3972
        assert len({(r["id"], r["arm"]) for r in records}) == 3972
        torch.set_num_threads(4)
        model_path = snapshot_download(
            "roberta-large", revision="722cf37b1afa9454edce342e7895e588b6ff1d59",
            allow_patterns=["config.json", "model.safetensors", "vocab.json",
                            "merges.txt", "tokenizer.json", "tokenizer_config.json",
                            "special_tokens_map.json"])
        baseline = Path(bert_score.__file__).parent / "rescale_baseline/en/roberta-large.tsv"
        scorer = BERTScorer(model_type=model_path, num_layers=17, lang="en", idf=False,
                            device="cuda:0", use_fast_tokenizer=False,
                            baseline_path=str(baseline), rescale_with_baseline=False)
        limit = scorer._tokenizer.model_max_length
        too_long = [(r["id"], r["arm"]) for r in records
                    if max(len(scorer._tokenizer.encode(s, truncation=False))
                           for s in (r["prediction"], r["reference"])) > limit]
        if too_long:
            raise ValueError(f"BERTScore tokenizer would truncate {len(too_long)} rows")
        raw = torch.zeros((len(records), 3), dtype=torch.float32)
        nonempty = [i for i, r in enumerate(records) if r["prediction"].strip()]
        with torch.inference_mode():
            if nonempty:
                values = scorer.score(
                    [records[i]["prediction"] for i in nonempty],
                    [records[i]["reference"] for i in nonempty], batch_size=16)
                raw[nonempty] = torch.stack(values, dim=1)
            scaled = (raw - scorer.baseline_vals) / (1 - scorer.baseline_vals)
        for i, row in enumerate(records):
            row["bertscore"] = {
                mode: {key: float(array[i, j]) for j, key in enumerate(
                    ("precision", "recall", "f1"))}
                for mode, array in (("raw", raw), ("rescaled", scaled))}
            assert all(math.isfinite(v) for scores in row["bertscore"].values()
                       for v in scores.values())
        groups = defaultdict(list)
        for row in records:
            for group in ("all", "answerable" if row["category"] != 5 else "adversarial",
                          "category_" + str(row["category"])):
                groups[(row["arm"], group)].append(row)
        aggregate = []
        for (arm, group), rows in sorted(groups.items()):
            aggregate.append({
                "arm": arm, "group": group, "n": len(rows),
                "locomo_mean": sum(r["locomo_score"] for r in rows) / len(rows),
                "raw": {key: sum(r["bertscore"]["raw"][key] for r in rows) / len(rows)
                        for key in ("precision", "recall", "f1")},
                "rescaled": {key: sum(r["bertscore"]["rescaled"][key] for r in rows) / len(rows)
                             for key in ("precision", "recall", "f1")},
            })
        result = {
            "protocol": payload["protocol"], "dataset_sha256": payload["dataset_sha256"],
            "merged_sha256": payload["merged_sha256"],
            "source_receipts": payload["source_receipts"],
            "payload_sha256": __PAYLOAD_SHA__,
            "reference_policy": payload["reference_policy"],
            "metric": "BERTScore", "model": "roberta-large",
            "model_revision": "722cf37b1afa9454edce342e7895e588b6ff1d59",
            "layers": 17, "idf": False, "use_fast_tokenizer": False,
            "batch_size": 16, "device": "cuda:0",
            "versions": {name: version(name) for name in
                         ("bert-score", "transformers", "torch", "huggingface-hub")},
            "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "baseline_values": scorer.baseline_vals.tolist(),
            "empty_predictions": len(records) - len(nonempty),
            "rescaling": "(raw-baseline)/(1-baseline); negatives retained",
            "aggregate": aggregate, "records": records,
            "seconds": time.monotonic() - started,
        }
        out = Path("/kaggle/working/care_bertscore_full.json")
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(out), "n": len(records), "seconds": result["seconds"],
                          "summary": [r for r in aggregate if r["group"] in
                                      ("all", "answerable", "adversarial")]}))
    ''').replace("__PAYLOAD__", repr(encoded)).replace("__PAYLOAD_SHA__", repr(payload_sha))
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                   "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "CARE_Full_BERTScore.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": "phamdinhdatai/asem-care-full-locomo-bertscore",
        "title": "ASEM CARE Full LoCoMo BERTScore",
        "code_file": "CARE_Full_BERTScore.ipynb",
        "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [],
    }
    (OUTPUT / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT / "input-manifest.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "records"}
                   | {"payload_sha256": payload_sha, "n_predictions": len(records)},
                   indent=2), encoding="utf-8")
    print(json.dumps({"notebook": str(OUTPUT / "CARE_Full_BERTScore.ipynb"),
                      "n_predictions": len(records), "payload_sha256": payload_sha,
                      "notebook_bytes": (OUTPUT / "CARE_Full_BERTScore.ipynb").stat().st_size}))


if __name__ == "__main__":
    main()
