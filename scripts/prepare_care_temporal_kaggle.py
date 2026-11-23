"""Create a private, code-only paired CARE Temporal Kaggle notebook."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "asem/care/__init__.py",
    "asem/care/sourcefirst.py",
    "asem/care/bridge.py",
    "asem/care/temporal_bridge.py",
    "asem/care/temporal_relations.py",
    "scripts/run_care_locomo.py",
    "scripts/run_care_bridge_locomo.py",
    "scripts/run_care_temporal_locomo.py",
    "scripts/run_care_kaggle.py",
    "scripts/run_kaggle_vllm.py",
    "kaggle/requirements-api.txt",
    "kaggle/requirements-vllm.txt",
    "kaggle/vllm-experiment-lock.json",
)


def _cell(kind: str, source: str) -> dict:
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(output: Path) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    contents = {name: (ROOT / name).read_bytes() for name in FILES}
    # The full asem package imports unrelated heavy modules. The Kaggle
    # snapshot only contains asem.care, so this stub avoids loading them.
    contents["asem/__init__.py"] = b'"""CARE pilot namespace; source code is unchanged."""\n'
    contents["scripts/__init__.py"] = b'"""CARE pilot script namespace."""\n'
    manifest = {"protocol": "care-temporal-kaggle-paired-v1", "scope": "code/config only; no benchmark data or secrets",
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}}
    contents["bundle-manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    archive_sha = hashlib.sha256(payload).hexdigest()
    (output / "care_bundle.zip").write_bytes(payload)
    bootstrap = f'''from pathlib import Path
import base64, hashlib, io, json, os, sys, zipfile
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {archive_sha!r}
workspace = Path("/kaggle/working/care_code").resolve()
workspace.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    for member in archive.infolist():
        destination = (workspace / member.filename).resolve()
        if not destination.is_relative_to(workspace) or ((member.external_attr >> 16) & 0o170000) == 0o120000:
            raise ValueError("Unsafe code bundle member")
    archive.extractall(workspace)
manifest = json.loads((workspace / "bundle-manifest.json").read_text())
for name, expected in manifest["files"].items():
    assert hashlib.sha256((workspace / name).read_bytes()).hexdigest() == expected, name
os.chdir(workspace)
sys.path.insert(0, str(workspace))
print("Verified CARE code-only bundle", len(manifest["files"]), "files")
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [_cell("markdown", "# CARE Temporal Relations paired LoCoMo pilot\nPrivate paired test on 30 fresh hash-offset 21-26 questions plus 10 unseen category-5 temporal-lexical questions. Arms: frozen CARE-Bridge v2 and opt-in CARE Temporal Bridge. Same pinned Qwen3-8B backbone on T4 x2. Gold is used only after generation. This is a pilot, not a full benchmark.\n"),
                  _cell("code", "PER_CATEGORY = 6\nSTART_PER_CATEGORY = 21\nCATEGORIES = (1, 2, 3, 4, 5)\nMAX_NEW_CALLS = 180\nHF_SECRET_LABEL = 'HF_TOKEN'  # optional existing secret, never paste its value\n"),
                  _cell("code", bootstrap),
                  _cell("code", "from scripts.run_care_kaggle import run\nrun(output='/kaggle/working/care_temporal_paired', per_category=PER_CATEGORY, start_per_category=START_PER_CATEGORY, categories=CATEGORIES, max_new_calls=MAX_NEW_CALLS, worker_module='scripts.run_care_temporal_locomo', hf_secret_label=HF_SECRET_LABEL)\n"),
                  _cell("markdown", "Results: `/kaggle/working/care_temporal_paired/status.json`, `results/report.json`, `results/rows.json`, `pilot.log`, and `vllm.log`. Model weights and dataset stay under `/kaggle/temp/care_sourcefirst`.\n")]}
    notebook_path = output / "CARE_Temporal_Kaggle.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": "phamdinhdatai/asem-care-temporal-relations-paired-pilot",
                "title": "ASEM CARE Temporal Relations Paired Pilot",
                "code_file": notebook_path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return {"notebook": str(notebook_path), "archive_sha256": archive_sha,
            "bundle_bytes": len(payload), "notebook_bytes": notebook_path.stat().st_size,
            "file_hashes": manifest["files"]}


if __name__ == "__main__":
    print(json.dumps(build(Path("build/kaggle/care-temporal-paired")), indent=2))
