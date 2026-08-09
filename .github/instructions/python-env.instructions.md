---
description: "Use when running tests, evaluation, benchmarks, or checking modified syntax in this project. Hard rule: activate the memory-r1 conda environment first, because the base Python lacks this project's dependencies."
name: "Python Environment"
applyTo: "**"
---

# Python Environment (memory-r1) — Hard Rule

This project runs **only** in the `memory-r1` conda environment. The base conda Python
(`/home/datpd/miniconda3/bin/python`) does **not** have this project's dependencies
(e.g. PyYAML, faiss, langchain), so any command will fail with `ModuleNotFoundError`
or similar.

## When this applies — BEFORE running ANY of these

- Tests: `pytest ...`
- Evaluation / benchmarks: `scripts/run_locomo10_experiments.py`, `scripts/run_locomo_benchmark.py`, `eval/*`, `bash run.sh <mode>`
- Syntax / import checks: `python -c "import asem"`, `python -m py_compile`, type checks
- Any `python` command against this codebase

## Required activation

Always activate `memory-r1` first, then run the command:

```bash
conda activate memory-r1
pytest tests/
```

Non-interactive / one-liner form:

```bash
conda activate memory-r1 && pytest tests/
conda activate memory-r1 && python scripts/run_locomo10_experiments.py --limit 10
```

If `conda activate` is not available (e.g. scripts, CI, non-login shells), use the
explicit source form:

```bash
source /home/datpd/miniconda3/etc/profile.d/conda.sh && conda activate memory-r1 && pytest tests/
```

## Rules

- NEVER use the base `python`/`python3` for this project — activate `memory-r1`.
- After activating, verify with: `python -c "import yaml, asem; print('ok')"` — should print `ok` without traceback.
- Do not silently "fix" the base environment by installing packages; the project env is `memory-r1`.
