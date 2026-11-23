#!/usr/bin/env python3
"""Run CARE Kaggle jobs on any Linux machine with two NVIDIA GPUs (standalone, standard library only).

The source of truth is the Kaggle notebook that ``prepare_care_ablation_kaggle`` / ``prepare_care_v3_kaggle``
built: it embeds the code bundle (verified by SHA-256 on extraction) and the exact ``run(...)`` call. This
script executes those two cells unchanged except for two path substitutions (``/kaggle/working`` -> workdir and a
cache directory), so the bundle, model revision, vLLM flags, decoding and worker arguments are identical to the
Kaggle run. Each job runs in its own subprocess. Nothing else from the repository is needed on the machine.

Hardware: the audited server uses tensor-parallel 2, fp16, an 8,192-token window and one sequence at a time, so two
GPUs of about 15 GB each (T4 class) are required. Other GPU generations may change fp16 rounding; the protocols'
reproducibility check (``care`` arm equals the saved CARE answers) reports whether they did.

    python run_care_on_gpu_box.py --notebooks care-ab --jobs pilot-locomo pilot-lme --workdir /data/care
    python run_care_on_gpu_box.py --notebooks care-ab --all --workdir /data/care --archive
    python run_care_on_gpu_box.py --notebooks care-v3 --jobs s0 s1 --workdir /data/care

``--notebooks`` is a copy of ``build/kaggle/care-ab`` or ``build/kaggle/care-v3`` (one folder per job holding
``notebook.ipynb``). Use ``--dry-run`` to print what would run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import time

MIN_GPU_MIB = 14000
ARCHIVE_PATTERNS = ("care_ab_*", "care_v3_lmes_*")


def job_source(notebook: Path, workdir: Path) -> str:
    """Return the two notebook code cells joined, with Kaggle paths replaced by ``workdir``."""
    cells = [c for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    if len(cells) != 2:
        raise ValueError(f"{notebook}: expected 2 code cells, found {len(cells)}")
    text = "\n".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in cells)
    if text.count("run(output=") != 1:
        raise ValueError(f"{notebook}: cannot locate the run(...) call")
    text = text.replace("run(output=", f"run(cache={str(workdir / 'cache')!r}, output=", 1)
    text = text.replace("/kaggle/working", str(workdir).replace("\\", "/"))
    if "/kaggle" in text.replace(str(workdir), ""):
        raise ValueError(f"{notebook}: an unhandled /kaggle path remains")
    compile(text, str(notebook), "exec")             # fail early on a bad substitution
    return text


def bundle_sha(source: str) -> str:
    m = re.search(r"hexdigest\(\) == '([0-9a-f]{64})'", source)
    if not m:
        raise ValueError("cannot find the bundle SHA-256 in the notebook")
    return m.group(1)


def output_name(source: str) -> str:
    m = re.search(r"output='[^']*/(care_[a-z0-9_]+)'", source)
    if not m:
        raise ValueError("cannot find the output directory name")
    return m.group(1)


def check_machine() -> list[str]:
    problems = []
    if platform.system() != "Linux":
        problems.append("Linux is required (the runner uses process groups and a venv layout from the Kaggle image)")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except Exception as exc:
        return problems + [f"nvidia-smi failed: {exc}"]
    gpus = [line.split(",") for line in out]
    big = [g for g in gpus if float(g[1]) >= MIN_GPU_MIB]
    if len(big) < 2:
        problems.append(f"need two GPUs with >= {MIN_GPU_MIB} MiB, found {[(g[0].strip(), g[1].strip()) for g in gpus]}")
    elif any("T4" not in g[0] for g in big[:2]):
        print("note: GPUs are not T4; fp16 rounding may differ from the Kaggle runs (the reproducibility check will say)")
    return problems


def run_job(name: str, notebook_dir: Path, workdir: Path, *, dry_run: bool, expect_sha: str = "") -> bool:
    source = job_source(notebook_dir / name / "notebook.ipynb", workdir)
    if expect_sha and bundle_sha(source) != expect_sha:
        print(f"[FAIL] {name}: bundle SHA-256 {bundle_sha(source)} is not the expected {expect_sha}")
        return False
    out = workdir / output_name(source)
    status = out / "status.json"
    if status.exists() and json.loads(status.read_text(encoding="utf-8")).get("state") == "complete":
        print(f"[skip] {name}: already complete in {out}")
        return True
    print(f"[run ] {name} -> {out}", flush=True)
    if dry_run:
        return True
    script = workdir / f"job_{name}.py"
    script.write_text(source, encoding="utf-8")
    started = time.monotonic()
    code = subprocess.run([sys.executable, str(script)], cwd=workdir).returncode
    print(f"[{'done' if code == 0 else 'FAIL'}] {name} in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    return code == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebooks", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--jobs", nargs="+", default=[])
    ap.add_argument("--all", action="store_true", help="every job folder under --notebooks, pilots first")
    ap.add_argument("--archive", action="store_true", help="tar.gz the result folders at the end")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-machine-check", action="store_true")
    ap.add_argument("--expect-sha", default="", help="refuse notebooks whose embedded bundle SHA-256 differs")
    args = ap.parse_args()
    available = sorted(p.parent.name for p in args.notebooks.glob("*/notebook.ipynb"))
    jobs = sorted(available, key=lambda n: (not n.startswith("pilot"), n)) if args.all else args.jobs
    unknown = [j for j in jobs if j not in available]
    if not jobs or unknown:
        print(f"jobs not found: {unknown or 'none given'}; available: {available}")
        return 2
    args.workdir.mkdir(parents=True, exist_ok=True)
    if not (args.dry_run or args.skip_machine_check):
        problems = check_machine()
        if problems:
            print("machine check failed:\n  " + "\n  ".join(problems))
            return 3
    failed = [j for j in jobs if not run_job(j, args.notebooks, args.workdir, dry_run=args.dry_run, expect_sha=args.expect_sha)]
    if args.archive and not args.dry_run:
        archive = args.workdir / f"care_results_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for pattern in ARCHIVE_PATTERNS:
                for folder in sorted(args.workdir.glob(pattern)):
                    if folder.is_dir():
                        tar.add(folder, arcname=folder.name)
        print(f"results archived in {archive}")
    print("FAILED: " + ", ".join(failed) if failed else "all jobs finished")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
