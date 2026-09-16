# ASEM: Agentic Self-Evolving Memory

ASEM is a memory framework for LLM agents that maintains a living knowledge graph
across sessions. The backbone model stays frozen; adaptation happens via the
external memory bank, learned utility estimates, and graph-based retrieval.

Two architectures are available:

- **ASEM v1** — Turn-by-turn ingestion (5-stage pipeline, per-turn LLM calls)
- **ASEM v2** — Two-phase batch ingestion + enhanced graph retrieval (99.8% fewer LLM calls)

## Highlights

- Multi-attribute atomic notes (keywords, tags, description + embeddings)
- **v2**: Batch session ingestion — process entire multi-turn conversations in ~3 LLM calls
- **v2**: Enhanced retrieval — Louvain communities, PageRank centrality, true N-hop traversal
- RL-trained memory manager (GRPO) for write operations
- Two-phase hybrid retrieval with value-aware re-ranking
- Non-parametric utility updates with EMA
- Pluggable inference backend (HuggingFace, LangChain/OpenAI, vLLM)
- **New**: Interactive memory graph visualization (HTML + PNG)

## Repository Structure

```
asem/                   Core library
  batch_ingestion.py      v2: batch session ingestion (3 LLM calls/session,
                          temporally grounded, typed bidirectional links)
  temporal.py             Session header detection + date/ISO parsing
  enhanced_retriever.py   v2: graph-enhanced retrieval (communities, PageRank, N-hop)
  visualizer.py           Memory graph visualization (interactive HTML + PNG)
  retriever.py            v1: hybrid retrieval (ANN + value-aware re-rank)
  link_evolver.py         Dynamic linking + memory evolution (B1 batched, B2 sparse)
  memory_bank.py          FAISS + SQLite storage
  memory_manager.py       RL-trained write ops (ADD/UPDATE/DELETE/NOOP)
  note.py                 Note schema + NoteConstructor
  pipeline.py             Five-stage pipeline orchestrator
  answer_agent.py         Distillation + answer generation
  utility_updater.py      EMA Q-value updates + experience consolidation
training/               GRPO training loops
eval/                   Evaluation harness + baselines + ASEMSystemV2
configs/                YAML configs for different backends
data/prompts/           LLM prompt templates (P1-P6)
scripts/                Run scripts (training, benchmarks v1+v2, visualization)
tests/                  Unit and integration tests
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run tests
pytest tests/

# 3. Demo with deterministic backend (no API needed)
python main.py --reset-db

# 4. Visualize the memory graph
python main.py --db data/benchmarks/demo_bank.sqlite --visualize
```

---

## ASEM v2 — Two-Phase Architecture (Batch Ingestion + Enhanced Retrieval)

ASEM v2 addresses the key bottleneck in v1: **turn-by-turn ingestion**. In v1, each
dialogue turn requires 3+ LLM calls — for a conversation with 419 turns and 200 QA
pairs, this wastes ~250,000 redundant LLM calls re-ingesting the same turns.

v2 separates the pipeline into two distinct phases:

### Phase 1: Offline Batch Ingestion (`asem/batch_ingestion.py`)

The entire multi-turn session dialogue is sent to the LLM in **3 batch calls**:

| Step | LLM Calls | Description |
|------|-----------|-------------|
| Batch Note Extraction | 1 | Extract ALL atomic facts from the full dialogue as structured notes |
| Batch Memory Operations | 1 | Decide ADD/UPDATE/DELETE/NOOP for all extracted notes at once |
| Batch Link Generation | 1 | Identify ALL pairwise relationships (semantic, causal, extends, etc.) |

**LLM call reduction**: 419 turns → **3 calls** (99.8% ↓)

#### Batch ingestion is temporally grounded and relation-typed

The ingest path was extended so that a batch session is not just a bag of notes,
but a *dated, typed subgraph* wired into the existing memory network. Six changes
define the current architecture:

| # | Change | Where | Effect |
|---|--------|-------|--------|
| 1 | **Session detection** — `extract_session_header()` reads the session number/date from the first turns; `session_id` falls back to `session_<n>` | `ingest_conversation()` | Session identity is derived from the data, not passed in by the caller |
| 2 | **Date parsing** — `parse_session_datetime()` maps the human-readable stamp (`"8 May 2023"`) to `(datetime, ISO-8601)` | `asem/temporal.py` | Gives notes a real event time; retriever gets an absolute-time signal |
| 3 | **Header injection** — `[Session Date: <date>]` is prepended to the dialogue when absent. `_extract_notes()` *also* supports a `{session_date}` placeholder, filled only when the template declares it | `ingest_conversation()`, `_extract_notes()` | The date reaches the LLM through the injected header today, because `P4_batch_note_extraction.txt` declares only `{dialogue}`. The placeholder is a dormant opt-in: no template currently uses it, and templates without it are unaffected |
| 4 | **Grounded note construction** — every note is stamped with `t = session datetime` (not wall-clock), plus `session_id`, `session_date`, `timestamp_iso`, `entities`, and `speaker` (regex fallback from `[Speaker] ...`) | `_embed_notes()` | Notes are anchorable in time and attributable to a speaker |
| 5 | **Typed bidirectional links** — `_batch_link()` writes `LinkRecord(target_id, relation)` on *both* endpoints, deduped by `target_id`, preserving the LLM-identified relation (`semantic` / `causal` / `temporal` / `extends` / `contradicts` / `same-topic`) | `_batch_link()` | Replaces the legacy flat-ID `L` list where the relation label was discarded |
| 6 | **Cross-session accounting** — each relation is classified once via `is_cross = (src_is_new XOR tgt_is_new)`; the step returns `(link_count, cross_session_links)` and logs total / cross / intra | `_batch_link()` | Makes "how much did this session bind into earlier memory?" a measurable quantity |

Link persistence uses an in-memory `all_note_map` built **once** before the write
loop. All endpoints are mutated on those retained objects and flushed with
`memory_bank.update(id, {"L": ...})`, so earlier links written in the same pass are
never clobbered by a later one.

**Temporal continuity on `UPDATE`** — when the LLM decides `UPDATE`, the merged note
takes `t = note.t if note.session_date else target.t`, unions `K` / `G` / `entities`,
carries over `target.L` and `target.q`, and prefers the new session's `session_id`,
`session_date`, and `timestamp_iso`. Evidence from an earlier session is therefore
revised rather than duplicated, while keeping its original timestamps when the new
fact is undated.

**Downstream consumption** — `HybridRetriever` applies a temporal boost keyed on
`session_date` / `timestamp_iso`, and the answer context prefixes each note with
`[<session_date>]`, so the ingest-side grounding is actually used at read time.

#### Known caveats in the current link accounting

- **`UPDATE` notes are counted as "new".** `_execute_ops()` returns the merged note,
  whose `id` equals the pre-existing `target.id`, into `added_notes`. Because
  `new_ids` is derived from that list, a link between a revised old note and another
  old note is classified `cross_session`. Reported cross-session numbers are
  therefore an **upper bound**, and revised notes are excluded from the neighbour
  candidate set.
- **`link_count` counts relations processed, not links newly created.** The dedupe
  guard skips the append when a link already exists, but the counter (and its
  `cross_session` share) still increments.
- The binding is one `memory_bank.update()` per endpoint per relation, so a session
  with many relations performs 2N read-modify-write passes rather than a single
  batched flush (`MemoryBank.add_many()`).

### Phase 2: Enhanced Graph Retrieval (`asem/enhanced_retriever.py`)

Retrieval augmented with global graph structure:

| Signal | v1 | v2 |
|--------|----|----|
| Local embedding similarity | ✅ | ✅ |
| Q-value utility | ✅ | ✅ |
| Multi-hop traversal | 1-hop only | **True N-hop with decay (0.7^hop)** |
| Community-aware boost | ❌ | **Louvain community detection** |
| PageRank centrality | ❌ | **Global importance weighting** |
| Intent-grounded Q | ❌ (`note.z` unused) | **Query-to-z similarity gates utility** |

**Hybrid score**: `0.35 × local_sim + 0.25 × global_graph + 0.40 × utility`

### Quick Benchmark (v2)

```bash
# Set up LLM server first (vLLM or OpenAI-compatible API)
export OPENAI_API_KEY="not-needed"
export OPENAI_BASE_URL="http://localhost:8000/v1"

# Smoke test with 10 QA pairs
python scripts/run_asem_v2.py --limit 10 --systems ASEMv2

# Full benchmark comparing v1 vs v2
python scripts/run_asem_v2.py \
  --systems ASEM ASEMv2 SimRetrieval \
  --metrics em rougeL bertscore_f1

# Per-category breakdown
python scripts/run_asem_v2.py --systems ASEMv2 --per-category
```

### Two-phase benchmark (`scripts/run_phase_benchmark.py`)

`run_asem_v2.py` ingests and answers in one process. To compare **backbones** — sizes,
local HF vs. hosted API — ingest once and then retrieve many times, so the only variable
is the retrieval model:

```powershell
conda activate memory-r1      # project env; base python lacks the deps
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# Phase A — build and persist one bank tree for the whole experiment
python scripts/run_phase_benchmark.py --phase ingest --tag shared `
    --ingest-config configs/models/qwen2.5_1.5b_hf.yaml `
    --systems FastASEM ASEMv2

# Phase B — reuse that tree, swap only the backbone
python scripts/run_phase_benchmark.py --phase retrieve --tag shared `
    --models-group hf_small --systems FastASEM ASEMv2 `
    --metrics em rougeL --per-category
```

**Inputs** — the dataset (`--input`, default `datasets/locomo/locomo10.json`), the model
registry (`--registry`, default `configs/models/registry.yaml`), the backbone config
(`--ingest-config` / `--config`), and `.env` for API credentials. Conversations come from
`conversation.session_<n>` plus `session_<n>_date_time`; QA items are grouped by
conversation, and each group is one ingest job.

**Outputs** — `--tag` is the experiment key binding both phases, so a single tag keeps
everything in one place:

| Phase | Path | Contents |
|-------|------|----------|
| A | `data/benchmarks/ingested_banks/<tag>/manifest.json` | ingest config, per-conversation note counts, new link edges, elapsed time |
| A | `data/benchmarks/ingested_banks/<tag>/<System>/<locomo_000N>/<bank>.sqlite` | one bank per system per conversation |
| B | `data/benchmarks/retrieval_work/<tag>/<model>/...` | per-run working copy, so the ingested banks stay pristine |
| B | `data/benchmarks/results/phased/<tag>__<model>.json` | metrics for one backbone |
| B | `data/benchmarks/results/phased/<tag>__sweep_table.md`, `__sweep.json` | multi-model comparison |
| B | `data/benchmarks/results/phased/preds/<tag>__<model>__<System>.jsonl` | per-QA predictions |

Roots are overridable with `--bank-root`, `--results-dir`, and `--work-root`. Channel
quality depends on the ingest phase: `--allow-missing-banks` logs a warning and skips any
system whose bank was never built, so check the manifest before a sweep. Full flag
reference, per-system bank filenames, and the one-embedder rule are in
[`configs/models/README.md`](configs/models/README.md).

### Architecture comparison

| Metric | v1 (turn-by-turn) | v2 (batch) |
|--------|------------------|------------|
| LLM calls per conversation (ingest) | ~1,257 | ~3 |
| LLM calls per conversation (total) | ~1,655 | ~403 |
| Retrieval signals | 2 (sim + q) | 5 (sim + q + community + centrality + multi-hop) |
| FAISS rebuilds per conversation | ~419 | ~1 |
| Deduplication | ❌ (re-ingests every QA) | ✅ (pre-ingest once) |
| Note timestamps | wall clock (run date) | session event date + ISO |
| Link representation | flat target IDs, relation lost | typed `LinkRecord` on both endpoints |
| Cross-session link metric | ❌ | ✅ `(total, cross_session)` per session |
| `UPDATE` semantics | overwrite | merge + temporal continuity |

### New prompt templates

| Template | Purpose |
|----------|---------|
| `data/prompts/P4_batch_note_extraction.txt` | Extract all atomic facts from multi-turn dialogue |
| `data/prompts/P5_batch_memory_ops.txt` | Batch ADD/UPDATE/DELETE/NOOP decisions |
| `data/prompts/P6_batch_link_generation.txt` | Batch pairwise relationship identification |

> `P4` may optionally declare a `{session_date}` placeholder to receive the parsed
> session date directly; `_extract_notes()` fills it only when present, so the
> template is free to stay on `{dialogue}` alone.

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
export OPENAI_BASE_URL="https://api.deepseek.com"   # or your proxy

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
python scripts/run_locomo_benchmark.py 
  --val     data/training/val.jsonl 
  --config  configs/locomo_openai.yaml 
  --results data/benchmarks/results/locomo_openai.json 
  --db-dir  data/benchmarks/eval_banks_openai 
  --systems NoMemory FullContext SimRetrieval 
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
python scripts/run_locomo_benchmark.py --val     data/training/val.jsonl --config  configs/locomo_vllm_qwen3_27b.yaml --results data/benchmarks/results/locomo_vllm.json --db-dir  data/benchmarks/eval_banks_vllm --systems NoMemory FullContext SimRetrieval --metrics em rougeL
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
| `ASEM` | Full five-stage pipeline (v1, turn-by-turn) |
| `ASEMv2` | **Two-phase batch ingestion + enhanced graph retrieval** |

---

## Metrics

- **EM** — Exact Match (primary)
- **rougeL** — ROUGE-L F1
- **bertscore_f1** — BERTScore F1 (requires `bert_score` package)

---

## Memory Graph Visualization

Visualize the knowledge graph as an interactive network with nodes (notes) and edges (links):

```bash
# After running the demo
python main.py --db data/benchmarks/demo_bank.sqlite --visualize

# Visualize an existing benchmark database
python scripts/visualize_memory.py \
  --db data/benchmarks/eval_banks_openai/asem.sqlite \
  --output memory_graph.html --stats

# Generate static PNG for papers
python scripts/visualize_memory.py \
  --db data/benchmarks/eval_banks_openai/asem.sqlite \
  --format png --output graph.png

# Open in browser
start memory_graph.html   # Windows
open memory_graph.html    # macOS
```

### Graph features

- **Nodes**: Sized by Q-value, colored by primary tag, hover for full attributes
- **Edges**: Weighted by cosine similarity, colored by inferred relation type
- **Legend**: Tag color map + edge type color map overlay
- **Physics**: ForceAtlas2 layout with zoom/pan/drag
- **Metrics**: Centrality, clustering, connected components, density

### Inferred edge types (heuristic, no LLM required)

| Condition | Type | Color |
|-----------|------|-------|
| Tag overlap ≥ 50% | same-topic | blue |
| Keyword overlap ≥ 50% | extends | green |
| Q-value difference > 0.3 | contradicts | red |
| Cosine similarity > 0.7 | semantic | gray |
| Default | linked | light gray |

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
- **v1 pipeline**: Orchestrated by `ASEMPipeline.run_turn()` in `asem/pipeline.py` (turn-by-turn).
- **v2 pipeline**: Two-phase design — `BatchIngestor.ingest_conversation()` once, then `EnhancedHybridRetriever.retrieve()` per QA.
- Stage 3 (link + evolve) is the most expensive v1 component — B1 (batch evolution) and B2 (sparse evolution) save 40-60%.
- The v1 `ASEMSystem.answer()` re-ingests history turns for every QA pair without dedup — v2 fixes this.
- `_MAX_LINK_HOPS = 3` existed as scaffolding in v1 but was never wired — v2 implements true N-hop.
- `note.z` (intent embedding) was stored but unused in v1 — v2 gates Q-values by query-to-z similarity.
