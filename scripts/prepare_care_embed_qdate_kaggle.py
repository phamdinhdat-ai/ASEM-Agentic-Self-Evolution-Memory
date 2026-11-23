"""Package two private Kaggle notebooks.

* ``asem-care-embed-qwen3-0-6b``: GPU embedding of every LoCoMo and
  LongMemEval-oracle turn/question with Qwen3-Embedding-0.6B (no LLM).
* ``asem-care-longmemeval-qdate``: question-date paired run on the 133
  temporal-reasoning questions with the Qwen3-8B vLLM server.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.prepare_care_full_kaggle import ROOT, _cell
from scripts.prepare_care_longmemeval_kaggle import FILES_TRANSFER


def _bundle(extra: str) -> tuple[bytes, str]:
    names = (*FILES_TRANSFER, extra)
    contents = {name: (ROOT / name).read_bytes() for name in names}
    contents["asem/__init__.py"] = b'"""CARE namespace."""\n'
    contents["scripts/__init__.py"] = b'"""CARE script namespace."""\n'
    manifest = {"files": {n: hashlib.sha256(d).hexdigest() for n, d in sorted(contents.items())}}
    contents["bundle-manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    payload = buffer.getvalue()
    return payload, hashlib.sha256(payload).hexdigest()


def _bootstrap(payload: bytes, sha: str) -> str:
    return f'''from pathlib import Path
import base64, hashlib, io, json, os, sys, zipfile
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {sha!r}
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
print("Verified code bundle", len(manifest["files"]), "files")
'''


def _write(output: Path, slug: str, title: str, intro: str, extra: str, run_source: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    payload, sha = _bundle(extra)
    (output / "bundle.zip").write_bytes(payload)
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                             "name": "python3"}, "language_info": {"name": "python"}},
                "cells": [_cell("markdown", f"# {title}\n{intro}\n"),
                          _cell("code", _bootstrap(payload, sha)), _cell("code", run_source)]}
    path = output / "notebook.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"phamdinhdatai/{slug}", "title": title, "code_file": path.name,
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"slug": slug, "bundle_sha256": sha}


def build() -> list[dict]:
    embed_run = (
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.51.0', 'nltk'], check=True)\n"
        "subprocess.run([sys.executable, '-m', 'scripts.embed_care_turns', '--lock', "
        "'kaggle/vllm-experiment-lock.json', '--cache', '/kaggle/temp/care_embed/datasets', "
        "'--output', '/kaggle/working/care_embed'], check=True)\n")
    qdate_run = (
        "from scripts.run_care_kaggle import run\n"
        "run(output='/kaggle/working/care_longmemeval_qdate', per_category=0, start_per_category=0, "
        "categories=(1,), max_new_calls=700, dataset_name='longmemeval_oracle.json', "
        "worker_module='scripts.run_care_longmemeval_qdate')\n")
    return [
        _write(ROOT / "build/kaggle/care-embed", "asem-care-embed-qwen3-0-6b",
               "ASEM CARE Embed Qwen3-0.6B",
               "Phase 0: Qwen3-Embedding-0.6B (fp16, GPU) embeddings of all LoCoMo and "
               "LongMemEval-oracle turns and questions. No LLM; retrieval is scored offline.",
               "scripts/embed_care_turns.py", embed_run),
        _write(ROOT / "build/kaggle/care-longmemeval-qdate", "asem-care-longmemeval-qdate",
               "ASEM CARE LongMemEval QDate",
               "Paired diagnostic: does exposing question_date to the reader help the 133 "
               "temporal-reasoning questions? Not a benchmark result.",
               "scripts/run_care_longmemeval_qdate.py", qdate_run),
    ]


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
