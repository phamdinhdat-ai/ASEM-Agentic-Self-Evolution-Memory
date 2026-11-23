"""Package the protocol-v3 confirmatory run: LongMemEval-S (full histories), pilot + 5 shards.

Reuses the factorial worker and bundle builder unchanged; only the dataset file, the
kernel slugs and the call cap differ. See docs/CARE_protocol_v3_20261118.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_care_factorial_kaggle import DATASET, bootstrap, bundle, build
from scripts.prepare_care_full_kaggle import ROOT, _cell

DATASET_FILE = "longmemeval_s_cleaned.json"
MAX_CALLS = 2000  # ~11 reader calls per question across all arms; 100 questions per shard


def build_v3(output: Path, shard: int, payload: bytes, sha: str, limit: int = 0) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    pilot = bool(limit)
    tag = "pilot" if pilot else f"s{shard}"
    slug = f"asem-care-v3-lmes-{tag}"
    # Kaggle derives the slug from the title, so the title must give the id above exactly.
    title = f"ASEM CARE v3 LMES {'Pilot' if pilot else 'S' + str(shard)}"
    out_dir = f"/kaggle/working/care_v3_lmes_{tag}"
    run_source = (
        "from scripts.run_care_kaggle import run\n"
        f"run(output='{out_dir}', per_category=0, start_per_category=0, categories=(1,), "
        f"max_new_calls={MAX_CALLS}, dataset_name='{DATASET_FILE}', "
        "worker_module='scripts.run_care_factorial', gpu_memory_utilization='0.70', "
        "extra_packages=('networkx==3.4.2',), "
        f"worker_extra_args=('--benchmark', 'longmemeval', '--shard-index', '{shard}', "
        f"'--question-limit', '{limit}'))\n")
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"}},
                "cells": [_cell("markdown", f"# {title}\nProtocol v3: LongMemEval-S full histories, "
                                "arms fixed in docs/CARE_protocol_v3_20261118.md. Gold only after inference.\n"),
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
    root = ROOT / "build/kaggle/care-v3"
    built = [build_v3(root / "pilot", 0, payload, sha, limit=4)]
    built += [build_v3(root / f"s{i}", i, payload, sha) for i in range(5)]
    print(json.dumps({"bundle_sha256": sha, "kernels": [b["slug"] for b in built]}, indent=2))


if __name__ == "__main__":
    main()
