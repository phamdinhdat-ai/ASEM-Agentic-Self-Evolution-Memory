"""Package the CARE component ablation (docs/CARE_ablation_protocol_20261121.md).

Two pilots (4 questions) and five shards per benchmark (LoCoMo, LongMemEval cleaned oracle). The bundle
is separate from the frozen protocol-v3 bundle and ships only what the ablation needs. Its SHA-256 is
printed and must be copied into the protocol file before any kernel is pushed.

    python -m scripts.prepare_care_ablation_kaggle [--pilots-only]
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from scripts.prepare_care_factorial_kaggle import bootstrap
from scripts.prepare_care_full_kaggle import FILES, ROOT, _cell

EXTRA = ("asem/care/hybrid.py", "asem/care/ablation.py", "asem/care/memo_backend.py",
         "scripts/run_care_ablation.py", "scripts/run_care_longmemeval.py")
DATASET = {"locomo": "locomo10.json", "longmemeval": "longmemeval_oracle.json"}
MAX_CALLS = {"locomo": 2600, "longmemeval": 1300}      # physical model calls per shard (see protocol, section Cost)


def bundle() -> tuple[bytes, str]:
    contents = {name: (ROOT / name).read_bytes() for name in (*FILES, *EXTRA)}
    contents["asem/__init__.py"] = b'"""CARE namespace."""\n'
    contents["scripts/__init__.py"] = b'"""CARE script namespace."""\n'
    manifest = {"protocol": "care-ablation-v1",
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


def build(output: Path, benchmark: str, shard: int, payload: bytes, sha: str, limit: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    short = "locomo" if benchmark == "locomo" else "lme"
    pilot = bool(limit)
    tag = "pilot" if pilot else f"s{shard}"
    slug = f"asem-care-ab-{short}-{tag}"
    title = f"ASEM CARE AB {short.upper()} {'Pilot' if pilot else 'S' + str(shard)}"
    out_dir = f"/kaggle/working/care_ab_{short}_{tag}"
    run_source = (
        "from scripts.run_care_kaggle import run\n"
        f"run(output='{out_dir}', per_category=0, start_per_category=0, categories=(1,), "
        f"max_new_calls={MAX_CALLS[benchmark]}, dataset_name='{DATASET[benchmark]}', "
        "worker_module='scripts.run_care_ablation', "
        f"worker_extra_args=('--benchmark', '{benchmark}', '--shard-index', '{shard}', "
        f"'--question-limit', '{limit}'))\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [_cell("markdown", f"# {title}\nCARE component ablation: one mechanism switched off per arm, "
                                "BM25 first bundle, lexical repair, Qwen3-8B reader. Gold only after inference.\n"),
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
    root = ROOT / "build/kaggle/care-ab"
    built = [build(root / "pilot-locomo", "locomo", 0, payload, sha, limit=4),
             build(root / "pilot-lme", "longmemeval", 0, payload, sha, limit=4)]
    if "--pilots-only" not in sys.argv:
        built += [build(root / f"locomo-s{i}", "locomo", i, payload, sha) for i in range(5)]
        built += [build(root / f"lme-s{i}", "longmemeval", i, payload, sha) for i in range(5)]
    print(json.dumps({"bundle_sha256": sha, "kernels": [b["slug"] for b in built]}, indent=2))


if __name__ == "__main__":
    main()
