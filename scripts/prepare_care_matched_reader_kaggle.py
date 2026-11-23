"""Package a full LoCoMo matched-reader ablation for private Kaggle runs."""

from __future__ import annotations

import base64
import glob
import hashlib
import io
import json
from pathlib import Path
import zipfile

from scripts.prepare_care_full_kaggle import FILES, ROOT, _cell


def locked_rankings() -> bytes:
    paths = [Path(p) for p in glob.glob(
        str(ROOT / "build/care/sim_retrieval/shard-*/**/report.json"), recursive=True)]
    if len(paths) != 5:
        raise ValueError(f"Expected five complete dense retrieval reports, got {len(paths)}")
    rankings = {}
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for row in report["rows"]:
            if row["id"] in rankings:
                raise ValueError(f"Duplicate dense question: {row['id']}")
            ids = row["systems"]["sim_retrieval"]["source_ids"]
            if len(ids) < 8 or len(set(ids[:8])) != 8:
                raise ValueError(f"Incomplete dense top-8: {row['id']}")
            rankings[row["id"]] = ids
    dataset = json.loads((ROOT / "datasets/locomo/locomo10.json").read_text(encoding="utf-8"))
    expected = {f"{sample['sample_id']}:{i}" for sample in dataset
                for i, _ in enumerate(sample["qa"])}
    if set(rankings) != expected:
        raise ValueError("Dense ranking question IDs differ from the locked LoCoMo dataset")
    return (json.dumps({"protocol": "dense-source-id-rankings-from-complete-sim-retrieval-v1",
                        "rankings": rankings}, sort_keys=True, separators=(",", ":")) + "\n").encode()


def build(output: Path, shard: int, rankings_bytes: bytes,
          *, question_limit: int = 0, slug_prefix: str = "asem-care-matched-reader",
          title_prefix: str = "ASEM CARE Matched Reader",
          output_prefix: str = "care_matched_reader",
          encoder_note: str = "dense top-8 first bundle from locked source-ID rankings") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    files = (*FILES, "scripts/run_care_matched_reader.py")
    contents = {name: (ROOT / name).read_bytes() for name in files}
    contents["asem/__init__.py"] = b'"""CARE namespace."""\n'
    contents["scripts/__init__.py"] = b'"""CARE script namespace."""\n'
    contents["dense_rankings_locked.json"] = rankings_bytes
    bundle_manifest = {"protocol": "care-matched-reader-dense-top8-v1",
                       "scope": "code plus source-ID rankings; no gold answers or secrets",
                       "files": {name: hashlib.sha256(value).hexdigest()
                                 for name, value in sorted(contents.items())}}
    contents["bundle-manifest.json"] = json.dumps(bundle_manifest, sort_keys=True).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(contents.items()):
            info = zipfile.ZipInfo(name, (2020, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, value)
    payload = buffer.getvalue()
    archive_sha = hashlib.sha256(payload).hexdigest()
    (output / "care_matched_reader_bundle.zip").write_bytes(payload)
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
print("Verified matched reader bundle", len(manifest["files"]), "files")
'''
    pilot = bool(question_limit)
    slug = (f"{slug_prefix}-pilot-s{shard}" if pilot else f"{slug_prefix}-shard-{shard}")
    title = (f"{title_prefix} Pilot S{shard}" if pilot else f"{title_prefix} Shard {shard}")
    output_name = f"{output_prefix}_{'pilot_' if pilot else ''}s{shard}"
    run_source = (
        "from scripts.run_care_kaggle import run\n"
        f"run(output='/kaggle/working/{output_name}', per_category=0, "
        "start_per_category=0, categories=(1,2,3,4,5), "
        f"max_new_calls={20 if pilot else 1000}, "
        "worker_module='scripts.run_care_matched_reader', "
        f"worker_extra_args=('--shard-index', '{shard}', '--question-limit', '{question_limit}'))\n"
    )
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3",
                                             "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [
                    _cell("markdown", f"# {title}\nSame CARE reader and prompt, {encoder_note}; "
                          "conditional repair "
                          "uses the unchanged CARE gate. Qwen3-8B, full LoCoMo shard. "
                          "Gold is used only after inference.\n"),
                    _cell("code", bootstrap), _cell("code", run_source),
                ]}
    notebook_path = output / "CARE_Matched_Reader.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
    metadata = {"id": f"phamdinhdatai/{slug}", "title": title,
                "code_file": notebook_path.name, "language": "python",
                "kernel_type": "notebook", "is_private": True,
                "enable_gpu": True, "enable_internet": True,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": [],
                "competition_sources": [], "kernel_sources": []}
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"notebook": str(notebook_path), "bundle_sha256": archive_sha,
            "ranking_sha256": hashlib.sha256(rankings_bytes).hexdigest(),
            "question_limit": question_limit, "notebook_bytes": notebook_path.stat().st_size}


def main() -> None:
    rankings = locked_rankings()
    root = ROOT / "build/kaggle/care-matched-reader"
    print(json.dumps([build(root / f"shard-{index}", index, rankings)
                      for index in range(5)] +
                     [build(ROOT / "build/kaggle/care-matched-reader-pilot", 0,
                            rankings, question_limit=4)], indent=2))


if __name__ == "__main__":
    main()
