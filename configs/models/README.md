# Small-model config system — phase-separated benchmark

This directory holds backbone configs for **small language models** (≈1B / 1.5B / 4B)
and a registry that ties them together for a **two-phase** LoCoMo benchmark:

| Phase | What it does | Writes |
|-------|--------------|--------|
| **ingest** | Builds each system's memory bank **once** and persists it to SQLite | `data/benchmarks/ingested_banks/<tag>/...` |
| **retrieve** | Loads those banks and answers QA — only the *backbone LLM* changes | `data/benchmarks/results/phased/...` |

Because the notes and their vectors are fixed after ingestion, a retrieval
sweep across model sizes measures **only the effect of the backbone**.

## The one rule: one embedder

Every config here must use the **same** `embedder_name`
(`sentence-transformers/all-MiniLM-L6-v2`). The embedder produces the vectors
stored in the bank, so the ingest and retrieve phases must agree on it. The LLM
may differ freely between phases.

## Models

| Tag | Params | Backend | Model id | Config |
|-----|--------|---------|----------|--------|
| `qwen3_1.7b_hf` | 1.7B | local HF | `Qwen/Qwen3-1.7B` | `qwen3_1.7b_hf.yaml` |
| `qwen2.5_1.5b_hf` | 1.5B | local HF | `Qwen/Qwen2.5-1.5B-Instruct` | `qwen2.5_1.5b_hf.yaml` |
| `qwen3_4b_hf` | 4B | local HF | `Qwen/Qwen3-4B` | `qwen3_4b_hf.yaml` |
| `qwen3_1.7b_api` | 1.7B | OpenAI-compatible | `Qwen/Qwen3-1.7B` | `qwen3_1.7b_api.yaml` |
| `qwen2.5_1.5b_api` | 1.5B | OpenAI-compatible | `Qwen/Qwen2.5-1.5B-Instruct` | `qwen2.5_1.5b_api.yaml` |
| `qwen3_4b_api` | 4B | OpenAI-compatible | `Qwen/Qwen3-4B` | `qwen3_4b_api.yaml` |

Groups (`--models-group`): `hf_small`, `api_small`, `all`.

> **Note — Qwen3 is a reasoning family.** If a config emits ` thinking ... <｜end▁of▁thinking｜>`
> traces, point it at the non-thinking build (e.g. `Qwen/Qwen3-4B-Instruct-2507`)
> or keep `enable_reasoning: false` (API) / prefer Qwen2.5 for local HF.

## Phase control

Each config carries a `phase:` block used as the CLI default:

```yaml
phase:
  mode: "combined"      # "ingest" | "retrieve" | "combined"
  bank_tag: "shared"    # directory name reused by every retrieval run
  ingest_config: null   # null => this config builds the banks
```

`--phase`, `--tag` and `--ingest-config` on the CLI override these values.

## Quick start

```powershell
# 1) Ingest ONCE (pick any backbone as the bank builder)
python scripts/run_phase_benchmark.py --phase ingest `
    --ingest-config configs/models/qwen2.5_1.5b_hf.yaml --tag shared `
    --systems FastASEM ASEMv2

# 2) Sweep retrieval across model sizes, reusing that bank
python scripts/run_phase_benchmark.py --phase retrieve --tag shared `
    --models-group hf_small --systems FastASEM ASEMv2 `
    --metrics em rougeL --per-category

# 3) Single-model retrieval run
python scripts/run_phase_benchmark.py --phase retrieve --tag shared `
    --config configs/models/qwen3_4b_api.yaml
```

---

## Running the two phases: data in, artifacts out

### Where the data comes from (both phases)

Every input is resolved relative to the repository root:

| Input | Flag | Default | What it provides |
|-------|------|---------|------------------|
| LoCoMo dataset | `--input` | `datasets/locomo/locomo10.json` | 10 conversations; per-session turns (`conversation.session_<n>`) and dates (`session_<n>_date_time`), plus the QA pairs |
| Model registry | `--registry` | `configs/models/registry.yaml` | tag → config path map, and the groups used by `--models-group` |
| Backbone config | `--ingest-config` / `--config` | `configs/models/qwen2.5_1.5b_hf.yaml` | model id, backend, and all retrieval hyperparameters |
| API credentials | — | `.env` at repo root | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (loaded automatically; only needed by `*_api` configs) |

The runner derives three things **in memory** — none of them are written to disk:

1. `load_raw_dataset(input)` → the raw LoCoMo records used for session/turn extraction.
2. `convert_locomo10_to_eval(input, limit)` → flat QA items; each carries
   `session_id = "locomo_<idx>"`, which becomes the per-conversation directory name.
3. `group_by_conversation(eval_items)` → one group per conversation = **one ingest job**.

### The `--tag` is the experiment key

`--tag` (default `shared`, or the config's `phase.bank_tag`) is the single string that
binds the two phases into one experiment. Phase A writes under `<bank_root>/<tag>/`, and
Phase B reads that same directory and writes results prefixed with `<tag>__`. To run a
second, independent experiment — a different ingest backbone, a different system set —
just pick a new tag and nothing is shared or overwritten.

### Artifacts written

```
data/benchmarks/
├── ingested_banks/<tag>/                         # ← PHASE A OUTPUT (frozen experiment)
│   ├── manifest.json                             #   ingest config, per-conv note counts,
│   │                                             #   new link edges, elapsed time
│   └── <System>/<locomo_000N>/<bank>.sqlite      #   one bank per system per conversation
│
├── retrieval_work/<tag>/<model>/<System>/<locomo_000N>/<bank>.sqlite
│                                                 # ← PHASE B SCRATCH (per-run copy)
│
└── results/phased/                               # ← PHASE B OUTPUT
    ├── <tag>__<model>.json                       #   metrics for one retrieval backbone
    ├── <tag>__sweep.json                         #   all backbones of a multi-model sweep
    ├── <tag>__sweep_table.md                     #   rendered markdown comparison table
    └── preds/<tag>__<model>__<System>.jsonl      #   per-QA predictions (inspect failures)
```

With the defaults, one `--tag shared` experiment therefore lives entirely under three
fixed roots: `data/benchmarks/ingested_banks/shared/`,
`data/benchmarks/retrieval_work/shared/<model>/`, and `data/benchmarks/results/phased/`.

### Bank filenames per system

Phase B locates each bank by the filename its builder creates (`BANK_FILE_NAMES` in
`eval/systems.py`):

| System | Bank file | Notes |
|--------|-----------|-------|
| `ASEM` | `asem.sqlite` | turn-by-turn v1 |
| `ASEMv2` | `asem_v2.sqlite` | batch ingestion |
| `FastASEM` | `fast_asem.sqlite` | SLAFI ingestion |
| `SimRetrieval` | `simretrieval.sqlite` | dense retrieval baseline |
| `AtomicLinking` | `atomiclinking.sqlite` | |
| `RLManagerOnly` | `rlmanageronly.sqlite` | |
| `ValueRetrievalOnly` | `valueretrievalonly.sqlite` | **writes while answering** (q-updates) |
| `NoMemory`, `FullContext` | — | bankless; skipped by the ingest phase |

### Overriding the storage layout

| Flag | Default | Purpose |
|------|---------|---------|
| `--bank-root` | `data/benchmarks/ingested_banks` | Phase A root; Phase B reads from here |
| `--results-dir` | `data/benchmarks/results/phased` | Phase B JSON/Markdown/prediction output |
| `--work-root` | `data/benchmarks/retrieval_work` | Where per-run bank copies are made |
| `--no-work-copy` | off | Answer against the ingested banks in place (**mutating systems will modify them**) |

### Interrupted runs resume; they do not restart

Phase B appends one JSON line per answered QA pair to
`preds/<tag>__<model>__<System>.jsonl` and, on start-up, reloads any file that already
exists to learn which QA indices are done. Re-running the exact same retrieval command
therefore **continues** where it stopped instead of re-answering everything.

Two consequences worth knowing:

* Changing `--metrics` and re-running is safe — metrics are recomputed from the reloaded
  predictions, no LLM calls are repeated.
* To genuinely re-run a backbone, delete its prediction files first
  (`Remove-Item data/benchmarks/results/phased/preds/<tag>__<model>__*.jsonl`), otherwise
  the old answers are silently reused.

Banks are resolved per conversation, so a partially-ingested tag still works: systems with
complete banks are answered, and `--allow-missing-banks` turns the per-conversation
`FileNotFoundError` into a warning + skip.

### Recommended full run

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"     # Windows + faiss/torch
conda activate memory-r1

# Phase A — one bank tree for the whole experiment
python scripts/run_phase_benchmark.py --phase ingest --tag shared `
    --ingest-config configs/models/qwen2.5_1.5b_hf.yaml `
    --systems FastASEM ASEMv2 ASEM SimRetrieval ValueRetrievalOnly

# Inspect what was built before spending retrieval budget
Get-Content data/benchmarks/ingested_banks/shared/manifest.json

# Phase B — swap only the backbone
python scripts/run_phase_benchmark.py --phase retrieve --tag shared `
    --models-group all --systems FastASEM ASEMv2 `
    --metrics em rougeL --per-category
```

### Environment notes (Windows / conda `memory-r1`)

* `torch` is **CPU-only** (2.9.1); 4B local inference is slow — prefer the `*_api`
  configs against a vLLM/proxy endpoint, or `--limit` for smoke tests.
* If Faiss/Torch raise `OMP: Error #15`, set `$env:KMP_DUPLICATE_LIB_OK="TRUE"`.
* Set `OPENAI_BASE_URL` / `OPENAI_API_KEY` in `.env` for the API configs.

## Safety: frozen banks

The retrieval phase copies every bank into
`data/benchmarks/retrieval_work/<tag>/<model>/...` before opening it, so systems
that legitimately write while answering (e.g. `ValueRetrievalOnly`'s q-updates)
cannot contaminate the ingested banks or other models' runs. Disable with
`--no-work-copy` (not recommended for sweeps).
---

# Static banks (`static/memory_banks`) — ingest once, evaluate many

The phase runner above and the **static-bank pipeline** solve the same problem
in two ways. The phase runner re-derives its banks under
`data/benchmarks/ingested_banks/`; the static pipeline freezes them under a
tracked repo-root `static/` tree so they are explicit, inspectable long-lived
artifacts shared by every evaluation.

| | Phase benchmark | Static banks |
|---|---|---|
| Ingest entry point | `scripts/run_phase_benchmark.py --phase ingest` | `scripts/build_static_banks.py` |
| Eval entry point | `scripts/run_phase_benchmark.py --phase retrieve` | `scripts/run_static_eval.py` |
| Bank root | `data/benchmarks/ingested_banks` | `static/memory_banks/<dataset>/<tag>` |
| Metrics | `em`, `rougeL`, `bertscore_f1` | `em`, `em_loose`, `f1`, `rougeL`, `bertscore_f1`, `judge` |
| Baselines | opt-in via `--systems` | `FullContext` + `NoMemory` included by default |
| Docs | this file | `static/README.md` |

```powershell
# Phase A — build the banks ONCE (idempotent, resumable, error-isolated)
python scripts/build_static_banks.py --tag deepseek `
    --ingest-config configs/models/deepseek_api.yaml

# Phase B — full LoCoMo QA from the frozen banks, incl. the baselines
python scripts/run_static_eval.py --tag deepseek `
    --config configs/models/deepseek_api.yaml `
    --judge-config configs/models/judge_api.yaml `
    --metrics em f1 rougeL bertscore_f1 judge

# Or run the ready-made command list
bash run_static_experiments.sh
```

Configs added for this workflow:

| Tag | Config | Purpose |
|-----|--------|---------|
| `deepseek_api` | `deepseek_api.yaml` | Default ingest + answer backbone (DeepSeek via the endpoint in `.env`) |
| `judge_api` | `judge_api.yaml` | LLM-as-a-judge grading, kept separate from the answered system |

> **`max_tokens` must cover reasoning + output.** The DeepSeek v4 models think
> before answering, and reasoning tokens count against `max_tokens`. Too small a
> budget returns `finish_reason="length"` with **empty content**, which yields a
> bank with 0 notes. `deepseek-chat` emits no reasoning tokens and is the
> cheapest option for bulk ingestion.

`judge_api` is never used to build a bank; it exists so grading can be swapped
independently (and so the registry validator still sees one uniform embedder).
`--judge-config` defaults to the answer backbone when omitted.