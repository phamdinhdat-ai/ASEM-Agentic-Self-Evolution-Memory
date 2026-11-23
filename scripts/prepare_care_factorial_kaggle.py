"""Package the pre-registered factorial run (docs/CARE_factorial_protocol_20261117.md).

Builds two pilots (4 questions each) and five shards per benchmark. Every
kernel ships the same code bundle; its SHA-256 is printed and must match the
protocol file before the full run is launched.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from scripts.prepare_care_full_kaggle import FILES, ROOT, _cell

EXTRA = ("asem/care/hybrid.py", "asem/care/graph_memory.py", "scripts/run_care_factorial.py", "scripts/embed_care_turns.py",
         "scripts/run_care_longmemeval.py", "scripts/run_care_context_baselines.py",
         "scripts/run_care_sim_retrieval.py")
DATASET = {"locomo": "locomo10.json", "longmemeval": "longmemeval_oracle.json"}
MAX_CALLS = {"locomo": 3500, "longmemeval": 1600}


def bundle() -> tuple[bytes, str]:
    contents = {name: (ROOT / name).read_bytes() for name in (*FILES, *EXTRA)}
    contents["asem/__init__.py"] = b'"""CARE namespace."""\n'
    contents["scripts/__init__.py"] = b'"""CARE script namespace."""\n'
    manifest = {"protocol": "care-factorial-v1",
                "files": {n: hashlib.sha256(d).hexdigest() for n, d in sorted(contents.items())}}
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


def bootstrap(payload: bytes, sha: str) -> str:
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
print("Verified factorial bundle", len(manifest["files"]), "files")
'''


def build(output: Path, benchmark: str, shard: int, payload: bytes, sha: str, limit: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    short = "locomo" if benchmark == "locomo" else "lme"
    pilot = bool(limit)
    slug = f"asem-care-fx-{short}-{'pilot' if pilot else 's' + str(shard)}"
    title = f"ASEM CARE FX {short.upper()} {'Pilot' if pilot else 'S' + str(shard)}"
    out_dir = f"/kaggle/working/care_fx_{short}_{'pilot' if pilot else 's' + str(shard)}"
    run_source = (
        "from scripts.run_care_kaggle import run\n"
        f"run(output='{out_dir}', per_category=0, start_per_category=0, categories=(1,), "
        f"max_new_calls={MAX_CALLS[benchmark]}, dataset_name='{DATASET[benchmark]}', "
        "worker_module='scripts.run_care_factorial', gpu_memory_utilization='0.80', "
        "extra_packages=('networkx==3.4.2',), "
        f"worker_extra_args=('--benchmark', '{benchmark}', '--shard-index', '{shard}', "
        f"'--question-limit', '{limit}'))\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [_cell("markdown", f"# {title}\nPre-registered factorial (first bundle BM25/dense x "
                                "repair none/lexical/hybrid/graph) plus baselines; Qwen3-8B reader, "
                                "Qwen3-Embedding-0.6B on GPU. Gold only after inference.\n"),
                          _cell("code", bootstrap(payload, sha)), _cell("code", run_source)]}
    path = output / "notebook.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    meta = {"id": f"phamdinhdatai/{slug}", "title": title, "code_file": path.name, "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
            "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"slug": slug}


def main() -> None:
    payload, sha = bundle()
    root = ROOT / "build/kaggle/care-fx"
    built = [build(root / "pilot-locomo", "locomo", 0, payload, sha, limit=4),
             build(root / "pilot-lme", "longmemeval", 0, payload, sha, limit=4)]
    if "--pilots-only" not in sys.argv:
        built += [build(root / f"locomo-s{i}", "locomo", i, payload, sha) for i in range(5)]
        built += [build(root / f"lme-s{i}", "longmemeval", i, payload, sha) for i in range(5)]
    print(json.dumps({"bundle_sha256": sha, "kernels": [b["slug"] for b in built]}, indent=2))


if __name__ == "__main__":
    main()
