"""Package pinned Qwen3-4B CARE transfer on the frozen 50-question cohort."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.prepare_care_full_kaggle import FILES, ROOT, _cell
from scripts.run_care_full_history_reader import select_audit_questions
from scripts.run_care_second_backbone import MODEL


def build(output: Path, *, question_limit: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads((ROOT / "datasets/locomo/locomo10.json").read_text(encoding="utf-8"))
    ids = [row["id"] for shard in range(5)
           for row in select_audit_questions(dataset, shard)]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("Frozen cohort must contain 50 unique questions")
    contents = {name: (ROOT / name).read_bytes()
                for name in (*FILES, "scripts/run_care_second_backbone.py")}
    lock = json.loads(contents["kaggle/vllm-experiment-lock.json"])
    lock["model"] = {**MODEL, "license": "Apache-2.0"}
    contents["kaggle/vllm-experiment-lock.json"] = (
        json.dumps(lock, indent=2) + "\n").encode("utf-8")
    contents["cohort_ids.json"] = (json.dumps({"protocol": "care-full-history-40k-v1",
                                                 "question_ids": ids}, indent=2) + "\n").encode()
    contents["asem/__init__.py"] = b'"""CARE namespace."""\n'
    contents["scripts/__init__.py"] = b'"""CARE script namespace."""\n'
    manifest = {"protocol": "care-second-backbone-qwen3-4b-v1",
                "scope": "code/config and frozen question IDs; no gold or secrets",
                "files": {name: hashlib.sha256(data).hexdigest()
                          for name, data in sorted(contents.items())}}
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
    (output / "care_second_backbone_bundle.zip").write_bytes(payload)
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
            raise ValueError("Unsafe bundle member")
    archive.extractall(workspace)
manifest = json.loads((workspace / "bundle-manifest.json").read_text())
for name, expected in manifest["files"].items():
    assert hashlib.sha256((workspace / name).read_bytes()).hexdigest() == expected, name
os.chdir(workspace)
sys.path.insert(0, str(workspace))
print("Verified Qwen3-4B CARE bundle", len(manifest["files"]), "files")
'''
    pilot = bool(question_limit)
    slug = "asem-care-second-backbone-4b-pilot" if pilot else "asem-care-second-backbone-4b"
    title = "ASEM CARE Second Backbone Qwen3-4B" + (" Pilot" if pilot else " Full Cohort")
    output_name = "care_second_backbone_4b" + ("_pilot" if pilot else "")
    run_source = (
        "from scripts.run_care_kaggle import run\n"
        f"run(output='/kaggle/working/{output_name}', per_category=0, "
        "start_per_category=0, categories=(1,2,3,4,5), "
        f"max_new_calls={10 if pilot else 120}, "
        "worker_module='scripts.run_care_second_backbone', "
        f"worker_extra_args=('--question-limit', '{question_limit}'))\n"
    )
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3",
                                             "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [
                    _cell("markdown", f"# {title}\nPinned Qwen3-4B, CARE first and conditional "
                          "pass on the frozen 50-question LoCoMo sensitivity cohort. "
                          "Gold remains outside model inputs.\n"),
                    _cell("code", bootstrap), _cell("code", run_source),
                ]}
    notebook_path = output / "CARE_Second_Backbone_4B.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"phamdinhdatai/{slug}", "title": title,
                "code_file": notebook_path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"notebook": str(notebook_path), "bundle_sha256": archive_sha,
            "question_limit": question_limit, "notebook_bytes": notebook_path.stat().st_size}


def main() -> None:
    print(json.dumps([
        build(ROOT / "build/kaggle/care-second-backbone-4b-pilot", question_limit=2),
        build(ROOT / "build/kaggle/care-second-backbone-4b")], indent=2))


if __name__ == "__main__":
    main()
