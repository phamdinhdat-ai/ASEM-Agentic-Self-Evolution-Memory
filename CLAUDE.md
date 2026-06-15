# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project summary

ASEM (Agentic Self-Evolving Memory) is a five-stage memory framework for LLM agents that maintains a living knowledge network across sessions. The backbone LLM is never fine-tuned — all adaptation happens through an external memory bank and learned utility estimates.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run tests
pytest tests/

# Demo (deterministic local backend, no external API needed)
python main.py                         # seeded facts + queries
python main.py --interactive           # interactive chat after demo
python main.py --reset-db              # clear memory bank first
python main.py --mode config           # use configs/default.yaml (real model backend)
python main.py --mode config --config configs/langchain_ollama.yaml

# LoCoMo benchmark (from run.sh)
OPENAI_API_KEY="..." OPENAI_BASE_URL="https://api.deepseek.com" PYTHONPATH="." \
  python scripts/run_locomo_benchmark.py \
    --val data/training/val.jsonl \
    --config configs/locomo_openai.yaml \
    --results data/benchmarks/results/locomo_openai.json \
    --db-dir data/benchmarks/eval_banks_openai \
    --systems NoMemory FullContext SimRetrieval \
    --metrics em rougeL

# Profile pipeline latency
python scripts/profile_pipeline.py --config configs/default.yaml --turns 5

# Full evaluation
python eval/run_full_evaluation.py --systems-module <module> --locomo <path> ...

# Generate results table
python eval/results_table.py --results path/to/results.json --output table.md

# Download benchmarks
python data/benchmarks/download_datasets.py --name longmemeval --url <URL> --output data/benchmarks/ --extract
```

There is no linter, formatter, or type-checker configured. `mypy clean` is an open TODO item.

## Architecture

### Five-stage pipeline (`asem/pipeline.py` → `ASEMPipeline`)

```
Content → [S1: NoteConstruction] → [S2: MemoryManager (RL)] → [S3: LinkEvolver]
Query  → [S4: HybridRetriever → AnswerAgent] → [S5: UtilityUpdater (EMA)]
```

All stages are orchestrated by `ASEMPipeline.run_turn()`. The pipeline exposes three paths:
- `write_path(content, timestamp)` — Stages 1→2→3
- `read_path(query)` — Stage 4
- `update_path(reward, used_notes, query, answer)` — Stage 5

### Pluggable inference backend (`asem/backends/`)

Every ASEM component receives an `InferenceBackend` via **constructor injection**. The base contract is:

```python
class InferenceBackend(ABC):
    def generate(self, prompt: str, **kwargs) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...
```

- `build_backend(config)` factory in `asem/backends/__init__.py` dispatches to **HuggingFaceBackend** or **LangChainBackend** based on config. No stage imports a model library directly.
- **HuggingFaceBackend**: `transformers.pipeline` for generation, `sentence_transformers.SentenceTransformer` for embeddings. Supports 4-bit/8-bit quantization via `bitsandbytes`.
- **LangChainBackend**: wraps any LangChain `BaseChatModel` + `Embeddings` provider (OpenAI, Anthropic, HuggingFace Hub, Ollama). Includes async methods.
- **Training always uses HuggingFaceBackend** — GRPO requires direct model weight access. The inference backend does not affect the training path.

### Memory bank (`asem/memory_bank.py`)

`MemoryBank` combines **FAISS** (ANN search on dense embeddings) with **SQLite** (metadata store). The FAISS index is rebuilt on every mutation (add/update/delete). Notes are serialized via JSON for SQLite storage.

### Note schema (`asem/note.py`)

Each memory note `m_i` is a dataclass with 9 fields: `id`, `c` (raw content), `t` (timestamp), `K` (keywords), `G` (tags), `X` (description), `e` (dense embedding), `L` (link set), `z` (intent embedding), `q` (utility/Q-value, initialized to `q0=0.5`).

### Two-phase retrieval (`asem/retriever.py`)

- **Phase A**: cosine similarity filter with threshold `δ` (default 0.30), returning up to `k1=20` candidates
- **Phase B**: composite score = `(1−λ)·sim_norm + λ·q_norm`, returning top `k2=5`. Both components are z-score normalized within the candidate pool to prevent magnitude dominance.
- `lambda_weight` (λ=0.40) controls the sim-vs-utility trade-off.

### RL training with GRPO (`training/`)

Both Memory Manager (Stage 2) and Answer Agent (Stage 4 distillation) use GRPO via `trl.GRPOTrainer`. The Memory Manager learns to select `{ADD, UPDATE, DELETE, NOOP}`; the Answer Agent learns to distill candidates and generate answers. Reward = exact-match score against ground truth.

### Utility updates (`asem/utility_updater.py`)

Stage 5 uses EMA: `q ← q + α·(r − q)` where α=0.10. Also performs experience consolidation — summarizing the trajectory into a new note.

### Evaluation (`eval/`)

Six baselines in `eval/baselines.py`: `NoMemory`, `FullContext`, `SimRetrieval`, `AtomicLinking`, `RLManagerOnly`, `ValueRetrievalOnly`. Three benchmark datasets: LongMemEval, LoCoMo, PersonalMemBench. Metrics: Exact Match, ROUGE-L, BERTScore-F1.

## Key design decisions

- **Frozen backbone**: prevents catastrophic forgetting; all adaptation is non-parametric via the external memory bank
- **Free-form link schema**: LLM identifies semantic/causal/temporal relationships at runtime — no schema migration
- **Z-score normalization in Phase B**: prevents sim magnitude from dominating Q-value in the composite score
- **EMA over TD learning**: simpler Monte Carlo backup; handles non-stationary reward distributions
- **GRPO over PPO**: no explicit value baseline needed; advantages normalized within group
- **Joint encoding** `concat(c, K, G, X)` for embedding `e`: captures full semantic footprint

## Configuration

YAML configs in `configs/` select the backend and set all hyperparameters (`k1`, `k2`, `k`, `δ`, `λ`, `α`, `q0`). Default backend: HuggingFace with Qwen2.5-7B-Instruct + all-MiniLM-L6-v2. Environment variables (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`) are read from `.env` for LangChain/API backends.

## Prompt files

Three LLM prompt templates live in `data/prompts/`:
- `P1_note_construction.txt` — extract K, G, X from raw content
- `P2_link_generation.txt` — identify relationships between notes
- `P3_memory_evolution.txt` — revise existing note attributes given a new related note
