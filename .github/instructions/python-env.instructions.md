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

## Windows host notes

The activation commands above assume the Linux dev box. On Windows the env lives at
`C:\Users\Dat Pham\anaconda3\envs\memory-r1` (a second Miniconda install also exists
at `miniconda3`, which is how the wrong base interpreter gets picked up — check
`conda env list`, the same env name can appear twice).

Call the interpreter by absolute path:

```powershell
$env:PYTHONPATH="."
$env:KMP_DUPLICATE_LIB_OK="TRUE"   # required on Windows, see below
& "C:\Users\Dat Pham\anaconda3\envs\memory-r1\python.exe" -m pytest tests/
```

**`KMP_DUPLICATE_LIB_OK=TRUE` is required.** `memory-r1` links two OpenMP runtimes,
so importing `torch`/`accelerate` aborts with:

```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
```

Without the variable the process dies mid-run with exit code `-1073741819`
(`0xC0000005`, access violation) — which looks like a crashing test but is a
native-load conflict. The variable suppresses the check; it does not otherwise
change the test outcome.

Verify the environment before trusting a result:

```powershell
& "...\memory-r1\python.exe" -c "import yaml, asem, transformers; print(transformers.__version__)"
```

This host has `transformers` **4.57.6**. A `5.8.0` in the environment means you are
on the base interpreter, and HuggingFace-backend tests will fail for reasons that
have nothing to do with your change. Record the interpreter path with any test
result you quote.
