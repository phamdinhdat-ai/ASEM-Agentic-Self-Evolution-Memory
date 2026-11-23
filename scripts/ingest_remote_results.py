"""Bring results of jobs run on another machine (scripts/run_care_on_gpu_box.py) into the repository.

Each result folder (``care_ab_locomo_s0``, ``care_v3_lmes_s2``, ...) is checked with the same gate the Kaggle
schedulers apply (row count, distinct ids, protocol / dataset / worker-code hashes, gold not in context, arms,
call cap). Valid folders are copied to ``kaggle_outputs`` exactly where a Kaggle download would land, and a marker
is written so the running scheduler counts the job as done and does not push it. Invalid folders are reported and
left out. Nothing is deleted. A Kaggle kernel that is still queued for an ingested job keeps running unless you
delete it on kaggle.com.

    python -m scripts.ingest_remote_results --protocol ab  --src /path/to/care_results
    python -m scripts.ingest_remote_results --protocol v3  --src /path/to/care_results
(--src may be a folder of result folders, a single result folder, or an extracted care_results_*.tar.gz)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import scripts.schedule_care_ablation as ab
import scripts.schedule_care_v3 as v3


def _targets(protocol: str) -> dict[str, tuple]:
    """result-folder name -> (job id, validate(job, dir), output root, marker directory)."""
    if protocol == "ab":
        return {ab.out_name(job): (job, ab.validate_dir, ab.OUTPUT, ab.INGEST) for job in ab.PILOTS + ab.SHARDS}
    return {f"care_v3_lmes_s{i}": (i, v3.validate_dir, v3.OUTPUT, v3.INGEST) for i in range(5)}


def ingest(protocol: str, src: Path, *, force: bool = False) -> dict[str, str]:
    targets = _targets(protocol)
    folders = [src] if src.name in targets else sorted(p for p in src.iterdir() if p.is_dir())
    report: dict[str, str] = {}
    for folder in folders:
        if folder.name not in targets:
            continue
        job, validate, output, marker_dir = targets[folder.name]
        key = str(job)
        ok, detail = validate(job, folder)
        if not ok:
            report[folder.name] = f"REJECTED: {detail}"
            continue
        label = job if protocol == "ab" else f"s{job}"
        destination = output / label / folder.name
        if destination.exists() and not force:
            report[folder.name] = "SKIPPED: a validated output already exists (use --force to overwrite)"
            continue
        shutil.copytree(folder, destination, dirs_exist_ok=True)
        marker_dir.mkdir(parents=True, exist_ok=True)
        (marker_dir / label).write_text(detail + "\n", encoding="utf-8")
        report[folder.name] = f"INGESTED: {detail}"
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=("ab", "v3"), required=True)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    report = ingest(args.protocol, args.src, force=args.force)
    for name, result in report.items():
        print(f"{name}: {result}")
    if not report:
        print("no recognised result folders found")
    return 0 if report and all(r.startswith(("INGESTED", "SKIPPED")) for r in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
