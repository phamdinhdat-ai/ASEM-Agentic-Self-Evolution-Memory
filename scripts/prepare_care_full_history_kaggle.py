"""Package 40K full-history CARE-reader sensitivity notebooks for Kaggle."""

from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.prepare_care_full_kaggle import FILES, ROOT, _cell


FILES_FULL_HISTORY = (*FILES, "scripts/run_care_context_baselines.py",
                      "scripts/run_care_full_history_reader.py")


def build(output: Path, shard: int, *, question_limit: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    contents = {name: (ROOT / name).read_bytes() for name in FILES_FULL_HISTORY}
    contents["asem/__init__.py"] = b'"""CARE namespace."""\n'
    contents["scripts/__init__.py"] = b'"""CARE script namespace."""\n'
    manifest = {"protocol": "care-full-history-40k-v1",
                "scope": "code/config only; benchmark downloaded from locked public URL",
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
    (output / "care_full_history_bundle.zip").write_bytes(payload)
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
print("Verified full-history code bundle", len(manifest["files"]), "files")
'''
    pilot = bool(question_limit)
    slug = ("asem-care-full-history-40k-pilot" if pilot
            else f"asem-care-full-history-40k-shard-{shard}")
    title = ("ASEM CARE Full History 40K Pilot" if pilot
             else f"ASEM CARE Full History 40K Shard {shard}")
    output_name = f"care_full_history_40k_{'pilot_' if pilot else ''}s{shard}"
    run_source = (
        "from scripts.run_care_kaggle import run\n"
        f"run(output='/kaggle/working/{output_name}', per_category=0, "
        "start_per_category=0, categories=(1,2,3,4,5), "
        f"max_new_calls={question_limit if pilot else 10}, "
        "server_context_tokens=40960, "
        "worker_module='scripts.run_care_full_history_reader', "
        f"worker_extra_args=('--shard-index', '{shard}', '--question-limit', '{question_limit}'))\n"
    )
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3",
                                             "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [
                    _cell("markdown", f"# {title}\nOne-pass CARE reader over all original "
                          "source turns in a 40K context. Fixed 50-question sensitivity "
                          "cohort: five per conversation, sampled by ID hash across "
                          "categories. Qwen3-8B, no repair, gold scored after generation.\n"),
                    _cell("code", bootstrap), _cell("code", run_source),
                ]}
    notebook_path = output / "CARE_Full_History_40K.ipynb"
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
    root = ROOT / "build/kaggle/care-full-history-40k"
    print(json.dumps([build(root / f"shard-{index}", index)
                      for index in range(5)] +
                     [build(ROOT / "build/kaggle/care-full-history-40k-pilot", 1,
                            question_limit=2)], indent=2))


if __name__ == "__main__":
    main()
