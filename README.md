# ASEM: Agentic Self-Evolving Memory

ASEM is a five-stage memory framework for LLM agents that maintains a living knowledge
network across sessions. The backbone model stays frozen; adaptation happens via the
external memory bank and utility estimates.

## Highlights

- Multi-attribute atomic notes (keywords, tags, description + embeddings)
- RL-trained memory manager (GRPO) for write operations
- Two-phase hybrid retrieval with value-aware re-ranking
- Non-parametric utility updates with EMA
- Pluggable inference backend (HuggingFace, LangChain/OpenAI, vLLM)

## Repository Structure

```
asem/              Core library
training/          GRPO training loops
eval/              Evaluation harness + baselines
configs/           YAML configs for different backends
data/              Prompts, training data, benchmark assets
scripts/           Run scripts (training, benchmark, results table)
tests/             Unit and integration tests
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests
pytest tests/
```

---

## Generate Training Data

Convert the LoCoMo dataset into ASEM training format:

```bash
python scripts/generate_training_data.py \
  --input  datasets/locomo/locomo10.json \
  --output data/training \
  --distractors 3 \
  --split 0.9 \
  --format jsonl
```

Output:
- `data/training/train.jsonl` — 1787 examples
- `data/training/val.jsonl` — 199 examples
- `data/training/by_category/` — per-category splits

---

## Run Benchmark — OpenAI API (gpt-4o-mini, gpt-5.2, etc.)

```bash
# Set your API credentials
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # or your proxy

# Run benchmark
PYTHONPATH="." python scripts/run_locomo_benchmark.py \
  --val     data/training/val.jsonl \
  --config  configs/locomo_openai.yaml \
  --results data/benchmarks/results/locomo_openai.json \
  --db-dir  data/benchmarks/eval_banks_openai \
  --systems NoMemory FullContext SimRetrieval AtomicLinking RLManagerOnly ValueRetrievalOnly ASEM \
  --metrics em rougeL

# Generate Markdown results table
PYTHONPATH="." python scripts/make_results_table.py \
  --results data/benchmarks/results/locomo_openai.json \
  --output  data/benchmarks/results/locomo_openai_table.md
```

PowerShell equivalent:
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:PYTHONPATH="."
python scripts/run_locomo_benchmark.py `
  --val     data/training/val.jsonl `
  --config  configs/locomo_openai.yaml `
  --results data/benchmarks/results/locomo_openai.json `
  --db-dir  data/benchmarks/eval_banks_openai `
  --systems NoMemory FullContext SimRetrieval `
  --metrics em rougeL
```

## Run Benchmark — vLLM (Qwen3-4B-2507, or any served model)

Your vLLM server exposes an OpenAI-compatible API at `http://localhost:8000/v1`.

```bash
# Point to vLLM (default port 8000, no auth needed)
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"

PYTHONPATH="." python scripts/run_locomo_benchmark.py \
  --val     data/training/val.jsonl \
  --config  configs/locomo_vllm_qwen3_4b-2507.yaml \
  --results data/benchmarks/results/locomo_vllm.json \
  --db-dir  data/benchmarks/eval_banks_vllm \
  --systems NoMemory FullContext SimRetrieval \
  --metrics em rougeL
```

PowerShell:
```powershell
$env:OPENAI_API_KEY="not-needed"
$env:OPENAI_BASE_URL="http://localhost:8000/v1"
$env:PYTHONPATH="."
python scripts/run_locomo_benchmark.py `
  --val     data/training/val.jsonl `
  --config  configs/locomo_vllm_qwen3_27b.yaml `
  --results data/benchmarks/results/locomo_vllm.json `
  --db-dir  data/benchmarks/eval_banks_vllm `
  --systems NoMemory FullContext SimRetrieval `
  --metrics em rougeL
```

If your vLLM server is on a different host/port:
```powershell
$env:OPENAI_BASE_URL="http://10.0.0.5:8080/v1"
```

The vLLM config uses `huggingface` for embeddings (local `all-MiniLM-L6-v2`), since
vLLM only serves text generation. Edit `configs/locomo_vllm_qwen3_27b.yaml` to
change the model name or embedder.

## Run Benchmark — HuggingFace Local (Qwen2.5-0.5B / 7B)

```bash
PYTHONPATH="." python scripts/run_locomo_benchmark.py \
  --val     data/training/val.jsonl \
  --config  configs/locomo_0.5b.yaml \
  --results data/benchmarks/results/locomo_hf.json \
  --db-dir  data/benchmarks/eval_banks_hf \
  --systems NoMemory FullContext \
  --metrics em rougeL
```

**Note:** HuggingFace local requires downloading the model (~1GB for 0.5B, ~15GB for 7B)
and enough GPU VRAM. The `locomo_0.5b.yaml` config uses `Qwen2.5-0.5B-Instruct`
which can run on CPU if needed.

---

## Training — Fine-tune Answer Agent with GRPO

```bash
PYTHONPATH="." python scripts/run_training.py \
  --train data/training/train.jsonl \
  --val   data/training/val.jsonl \
  --config configs/locomo_0.5b.yaml \
  --output-dir checkpoints/answer_agent_0.5b \
  --epochs 3 \
  --group-size 8
```

Train on a single QA category:
```bash
python scripts/run_training.py \
  --train data/training/train.jsonl \
  --config configs/locomo_0.5b.yaml \
  --output-dir checkpoints/answer_agent_cat1 \
  --category 1 \
  --epochs 5
```

---

## Available Configs

| Config | Backend | Model | Embedder |
|---|---|---|---|
| `configs/default.yaml` | HuggingFace local | Qwen2.5-7B-Instruct | all-MiniLM-L6-v2 |
| `configs/locomo_0.5b.yaml` | HuggingFace local | Qwen2.5-0.5B-Instruct | all-MiniLM-L6-v2 |
| `configs/locomo_openai.yaml` | LangChain → OpenAI | gpt-5.2 (configurable) | all-MiniLM-L6-v2 (HF) |
| `configs/locomo_vllm_qwen3_27b.yaml` | LangChain → OpenAI | Qwen3-27B (vLLM) | all-MiniLM-L6-v2 (HF) |

---

## Available Systems (Benchmark)

| System | Description |
|---|---|
| `NoMemory` | Frozen backbone only — no history |
| `FullContext` | All prior turns concatenated (oracle upper bound) |
| `SimRetrieval` | Flat ANN retrieval by cosine similarity |
| `AtomicLinking` | Notes + linking, no RL ops or Q-values |
| `RLManagerOnly` | RL write ops, similarity retrieval |
| `ValueRetrievalOnly` | Q-value retrieval on IEU triplets |
| `ASEM` | Full five-stage pipeline |

---

## Metrics

- **EM** — Exact Match (primary)
- **rougeL** — ROUGE-L F1
- **bertscore_f1** — BERTScore F1 (requires `bert_score` package)

---

## QA Categories (LoCoMo)

| Category | Name | Description |
|---|---|---|
| 1 | single_hop | Factual recall from one turn |
| 2 | temporal | Time-based reasoning |
| 3 | commonsense | Inference / counterfactual |
| 4 | conversational | Direct dialogue quote |
| 5 | adversarial | Deliberately mis-attributed question |

---

## Human Evaluation

- Build blinded annotation sets with [eval/human_eval.py](eval/human_eval.py)
- Open the UI at [data/benchmarks/human_eval/index.html](data/benchmarks/human_eval/index.html)

## Notes

- Training always uses the HuggingFace backend, even if inference uses LangChain/vLLM.
- The five-stage pipeline is orchestrated by `ASEMPipeline.run_turn()` in `asem/pipeline.py`.
- Stage 3 (link + evolve) is the most expensive component — see B1+B2 improvements.
