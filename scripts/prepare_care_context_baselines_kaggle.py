"""Build private Qwen3-8B NoMemory and FullContext-8K LoCoMo notebooks."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.prepare_care_full_kaggle import FILES, ROOT, _cell


BUNDLE_FILES = (*FILES, "scripts/run_care_context_baselines.py")


def build(output: Path, shard: int, *, question_limit: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    contents = {name: (ROOT / name).read_bytes() for name in BUNDLE_FILES}
    contents["asem/__init__.py"] = b'"""CARE namespace."""\n'
    contents["scripts/__init__.py"] = b'"""CARE script namespace."""\n'
    manifest = {
        "protocol": "care-context-baselines-locomo-v1",
        "scope": "code/config only; no benchmark data or secrets",
        "files": {name: hashlib.sha256(data).hexdigest()
                  for name, data in sorted(contents.items())},
    }
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
    (output / "care_context_bundle.zip").write_bytes(payload)
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
print("Verified baseline code-only bundle", len(manifest["files"]), "files")
'''
    is_pilot = bool(question_limit)
    slug = (f"asem-care-context-baseline-pilot-s{shard}"
            if is_pilot else f"asem-care-context-baseline-shard-{shard}")
    title = (f"ASEM CARE Context Baseline Pilot S{shard}"
             if is_pilot else f"ASEM CARE Context Baseline Shard {shard}")
    output_name = f"care_context_{'pilot_' if is_pilot else ''}s{shard}"
    run_source = (
        "from scripts.run_care_kaggle import run\n"
        f"run(output='/kaggle/working/{output_name}', per_category=0, "
        "start_per_category=0, categories=(1,2,3,4,5), "
        f"max_new_calls={20 if is_pilot else 1000}, "
        "worker_module='scripts.run_care_context_baselines', "
        f"worker_extra_args=('--shard-index', '{shard}', '--question-limit', '{question_limit}'))\n"
    )
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [
            _cell("markdown", f"# {title}\nQwen3-8B, locked LoCoMo data, NoMemory "
                  "and FullContext-8K. FullContext keeps the first five dated turns "
                  "and the longest recent suffix that fits 8K. This is a context-limited "
                  "baseline, not a full-history long-context result. Gold is scored only "
                  "after model responses.\n"),
            _cell("code", bootstrap),
            _cell("code", run_source),
            _cell("markdown", f"Results: `/kaggle/working/{output_name}/results/report.json`, "
                  "`status.json`, `pilot.log`, and `vllm.log`.\n"),
        ],
    }
    path = output / "CARE_Context_Baselines.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {
        "id": f"phamdinhdatai/{slug}", "title": title,
        "code_file": path.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
        "competition_sources": [], "kernel_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return {"notebook": str(path), "archive_sha256": archive_sha,
            "bundle_bytes": len(payload), "notebook_bytes": path.stat().st_size,
            "question_limit": question_limit}


def main() -> None:
    base = ROOT / "build/kaggle/care-context-baselines"
    print(json.dumps([build(base / f"shard-{i}", i) for i in range(5)], indent=2))
    print(json.dumps(build(ROOT / "build/kaggle/care-context-baseline-pilot",
                           0, question_limit=4), indent=2))


if __name__ == "__main__":
    main()
