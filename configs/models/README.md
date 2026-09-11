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
