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
| `FastASEM` | v3: LLM-free graph weaving at ingest (`X = c = fact`), `direct_answer` graph-node prompt |

---

## Metrics

- **EM** — Exact Match (strict; near-zero on free-form answers, so never read it alone)
- **F1 / rougeL** — token-overlap metrics (penalise verbose, correct answers)
- **bertscore_f1** — BERTScore F1 (requires `bert_score` package)
- **judge** — LLM-as-a-judge binary verdict (`eval/llm_as_a_judge.py`, `eval/metrics.py`);
  see [judge caveats](README.md#8-hiệu-chỉnh-judge)

Always report **judge next to EM/F1**: this codebase's systems answer in prose, so EM≈0
while the judge can still mark the answer correct (see the benchmark report below).

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

> **Label mismatch.** `CATEGORY_NAMES` in `scripts/run_locomo10_experiments.py` is wrong for
> categories 1, 3 and 4. Measured from `datasets/locomo/locomo10.json`: category 1 rows carry
> **3.13 evidence turns on average (98% need >1)**, category 4 rows **1.07 (only 5% need >1)**,
> category 3 **2.08**. So the harness prints `single_hop` for the multi-hop set and
> `conversational` for the single-hop set. The table below is the corrected mapping; the left
> column keeps the raw `category` id stored in the predictions/results files.

| Category | Harness label (wrong) | Actual type | n | Evidence turns (mean) |
|---|---|---|---|---|
| 1 | ~~single_hop~~ | **multi_hop** | 282 | 3.13 |
| 2 | temporal | temporal | 321 | 1.17 |
| 3 | ~~commonsense~~ | **open_domain** (inference / counterfactual) | 96 | 2.08 |
| 4 | ~~conversational~~ | **single_hop** | 841 | 1.07 |
| 5 | adversarial | adversarial (mis-attributed question; gold = the original fact) | 446 | 1.03 |

Total 1986 QA. Any table in this repo that prints `single_hop` / `conversational` must be read
with this correction in mind.

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

---

# Benchmark report — LoCoMo10 frozen banks (`ds_fixed`) × Qwen3-4B-Instruct-2507

Báo cáo phân tích lần chạy static-bank với backbone nhỏ (4B). Mọi con số dưới đây được đọc
**trực tiếp từ artifact trên đĩa** (`preds/`, `scores/`) hoặc tính lại bằng các script audit
LLM-free ở `scratch_diag/` — không có suy đoán.

## 0. Cấu hình lần chạy

| Thành phần | Giá trị |
|---|---|
| Bank | `static/memory_banks/locomo10/ds_fixed/` (ASEM re-ingest: 2732 note; FastASEM copy từ `ds_nothink`: 3504 note) |
| Dataset | `datasets/locomo/locomo10.json`, 10 hội thoại, **1986 QA** |
| Backbone **và judge** | `configs/models/qwen3_4b_instruct_api.yaml` (Qwen3-4B-Instruct-2507 qua API OpenAI-compatible) |
| Ngân sách answer | `context_window: 8192`, `answer.max_tokens: 512` (budget prompt = 7616 token) |
| Truy hồi | `k1=20`, `k2=5`, `δ=0.30`, RRF (dense 1.0 / BM25 0.8 / entity 0.6 / temporal 0.5), `max_context_notes: 8` |
| Metrics | `em, f1, rougeL, bertscore_f1, judge` |
| Thời gian | 39 008 s (~10.8 h) cho 4 hệ thống × 1986 QA + judge |

Artifact: `data/benchmarks/results/static/locomo10/ds_fixed__qwen_qwen3_4b_instruct_2507.{md,json}`,
`preds/`, `scores/`.

> Lưu ý: config trên Kaggle không nằm trong repo; file tương đương là
> `configs/models/qwen3_4b_api.yaml`.

Tái lập phần phân tích (không gọi API, không cần GPU):

```bash
conda activate memory-r1
python scratch_diag/qwen3_result_audit.py                 # -> scratch_diag/_qwen3_audit.txt
python scratch_diag/fullcontext_evidence_survival.py     # evidence còn sống trong prompt FullContext
python scratch_diag/retrieval_recall_qwen.py             # tách INGESTION / RETRIEVAL / GENERATION
```

## 1. Điểm tổng

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM | 1986 | 0.0055 | 0.2372 | 0.2236 | 0.8563 | **0.6782** |
| FastASEM | 1986 | 0.0740 | 0.2821 | 0.2683 | 0.8688 | 0.5000 |
| FullContext | 1986 | 0.0745 | **0.3485** | 0.3361 | **0.8773** | **0.7039** |
| NoMemory | 1986 | 0.0035 | 0.0678 | 0.0625 | 0.8288 | 0.1843 |

- ASEM đạt **96.3% mức judge của FullContext** (0.678 vs 0.704).
- F1 của ASEM thấp hơn FullContext 32%, EM gần 0 → xem mục 4 để biết vì sao **không** nên đọc
  kết luận này theo hướng "ASEM kém hơn".
- FastASEM chỉ còn 0.500 judge: **11.4% câu là từ chối trả lời** và 29.6% ở nhóm adversarial.

## 2. Theo loại câu hỏi (judge), nhãn đã sửa

| Loại (đúng) | n | ASEM | FastASEM | FullContext | NoMemory | ASEM − FC |
|---|---|---|---|---|---|---|
| adversarial | 446 | 0.543 | 0.117 | 0.377 | 0.218 | **+0.166** |
| temporal | 321 | 0.704 | 0.611 | 0.583 | 0.019 | **+0.121** |
| open_domain | 96 | 0.500 | 0.438 | 0.531 | 0.271 | −0.031 |
| multi_hop (`single_hop` cũ) | 282 | 0.617 | 0.475 | 0.706 | 0.103 | −0.089 |
| single_hop (`conversational` cũ) | 841 | 0.781 | 0.677 | 0.943 | 0.247 | **−0.162** |

## 3. Kiểm định ghép cặp (McNemar chính xác + bootstrap 2000, ASEM vs FullContext)

| Tập | n | ASEM | FC | Chênh | p | 95% CI |
|---|---|---|---|---|---|---|
| Tất cả | 1986 | 0.678 | 0.704 | −2.6 pp | **0.047** | [−5.0, −0.1] pp |
| adversarial | 446 | 0.543 | 0.377 | **+16.6 pp** | <1e-4 | [+11.0, +22.4] |
| temporal | 321 | 0.704 | 0.583 | **+12.1 pp** | 0.0009 | [+5.6, +18.7] |
| single_hop | 841 | 0.781 | 0.943 | **−16.2 pp** | <1e-4 | [−19.3, −13.1] |
| multi_hop | 282 | 0.617 | 0.706 | −8.9 pp | 0.018 | [−16.0, −2.5] |
| open_domain | 96 | 0.500 | 0.531 | −3.1 pp | 0.72 | [−13.5, +8.3] |
| Bỏ adversarial | 1540 | 0.718 | 0.799 | **−8.1 pp** | <1e-4 | [−10.9, −5.5] |

**Kết luận:** không được nói "ASEM ngang bằng FullContext". Khác biệt 2.6 pp trên toàn bộ là
nhỏ nhưng **có thật** (p = 0.047) và khi loại adversarial thì thành 8.1 pp. Ưu thế ở adversarial
và temporal là vững; open_domain (n=96) không đủ lực để kết luận.

## 4. Vì sao F1/EM và judge nói ngược nhau

| System | p50 ký tự | p50 từ | % từ chối | Judge |
|---|---|---|---|---|
| ASEM | 78 | 14 | 1.7% | 0.678 |
| FastASEM | 54 | 9 | 11.4% | 0.500 |
| FullContext | 64 | 11 | 3.0% | 0.704 |
| NoMemory | 51 | 8 | 47.5% | 0.184 |

- Chênh **F1** của FullContext tập trung ở single_hop: 0.473 vs 0.268 (**+0.205**), trong khi
  chênh judge ở đó chỉ +0.162 → lợi thế F1 phần lớn là **chép lại nguyên văn** (xem mục 5).
- ASEM viết thành câu kèm mệnh đề phụ (attribution, "which suggests…") → precision giảm → EM≈0
  dù nội dung đúng. Đây là khác biệt **phong cách hệ thống**, lặp lại ở cả 10 hội thoại.
- FastASEM F1 cao hơn ASEM **không** có nghĩa tốt hơn: judge 0.500 và 11.4% từ chối.

## 5. Phát hiện quan trọng nhất: FullContext được trao sẵn đáp án

`_build_history_for_qa` trong `scripts/run_locomo10_experiments.py` dựng history = toàn bộ lượt
thoại từ session 1 **đến session chứa evidence, và bao gồm cả lượt evidence**. Replay đúng đường
production (`scratch_diag/fullcontext_evidence_survival.py`, spy backend, ngân sách 8192/512):

| Loại | history (lượt, p50) | giữ lại (p50) | % bị cắt | prompt tok p50 | **% giữ toàn bộ evidence** | vị trí evidence cuối |
|---|---|---|---|---|---|---|
| single_hop | 358 | 205 | 69.7% | 7588 | **100.0%** | **0.94** |
| adversarial | 351 | 202 | 69.1% | 7586 | 100.0% | 0.93 |
| temporal | 302 | 198 | 62.0% | 7585 | 96.0% | 0.92 |
| multi_hop | 418 | 208 | 81.6% | 7593 | **51.1%** | 0.91 |

Ở đúng nhóm FullContext thắng đậm nhất (841 câu, 42% dữ liệu), câu trả lời **nằm nguyên văn ở
cuối prompt, vị trí 0.94**. Vì vậy 0.943 của FullContext ở nhóm này là "chép lượt thoại kề bên",
không phải năng lực long-context. Ở nhóm cần ghép nhiều turn (multi_hop) evidence chỉ còn 51% và
FullContext thua ASEM 8.9 pp.

**Hệ quả cho phương pháp luận:** đây không phải bài toán needle-in-a-haystack. Nếu muốn đo khả
năng đọc ngữ cảnh dài thật, phải dựng biến thể history đóng ở session **trước** evidence.

## 6. Thất bại của ASEM nằm ở tầng nào? (replay retriever thật, không gọi LLM)

`scratch_diag/retrieval_recall_qwen.py`: note mang đáp án = note phủ ≥60% token của gold.

| Lát cắt | n | INGESTION (gold không có trong bank) | RETRIEVAL (có trong bank, không vào context) | GENERATION (có trong context vẫn sai) |
|---|---|---|---|---|
| Tất cả ASEM sai | 628 | 30% | 41% | 29% |
| single_hop: ASEM sai & FC đúng | 162 | 18% | **49%** | 33% |
| single_hop: ASEM **đúng** (đối chứng) | 653 | 8% | 8% | 84% |
| multi_hop: ASEM sai | 102 | **58%** | 29% | 13% |

- **single_hop**: một nửa lỗi là **retrieval precision** — bank có note mang đáp án nhưng nó
  không lọt vào 8 note của context. Ví dụ `#104 "What book did Caroline recommend?"` → "Charlotte's
  Web" thay vì "Becoming Nicole"; `#112` → "dog-faced cup" thay vì bức tranh tháng 7.
- **multi_hop**: 58% là **lỗi ingest** — câu hỏi đòi hợp nhất nhiều phiên ("What activities does
  Melanie partake in?" → gold 4 mục) mà không note nào chứa hợp nhất đó. Đây là giới hạn kiến trúc
  (note = fact per-turn), không phải lỗi truy hồi.

## 7. FastASEM ở adversarial: từ chối, không phải bịa

| | adversarial n | ASEM | FastASEM | FullContext | NoMemory |
|---|---|---|---|---|---|
| judge | 446 | 54.3% | 11.7% | 37.7% | 21.7% |
| % từ chối | 446 | 2.9% | **29.6%** | 9.2% | 40.1% |
| judge trên các câu bị từ chối | 446 | 8% | **0%** | 5% | 1% |

132/446 câu adversarial của FastASEM bị từ chối, và **0% trong số đó được chấm đúng** (vì gold tồn
tại). Nguyên nhân nằm trong `data/prompts/P_temporal_qa.txt`: guideline 3 yêu cầu *"STILL GIVE THE
FACT"* khi câu hỏi gán sai người, nhưng guideline 5 lại nói được phép trả lời *"I don't know"*
"including the case where the fact belongs to a different person than the question names" — mâu
thuẫn trực tiếp, và mô hình 4B chọn từ chối.

⇒ **Không** thêm bước "kiểm tra bằng chứng / từ chối" cho FastASEM (như một số phân tích đề xuất),
mà phải **bỏ quyền từ chối** đó. Gold của cat 5 là **fact gốc**, nên từ chối = sai theo thiết kế.

## 8. Hiệu chỉnh judge

Judge là **cùng Qwen3-4B** với backbone (self-judge) → phải đọc kèm sai số:

| System | judge=True & có chuỗi gold | judge=True & **thiếu** chuỗi gold | judge=True nhưng `judge_error` khác null |
|---|---|---|---|
| ASEM | 610 (30.7%) | 737 (37.1%) | **48 (2.4% toàn bộ)** |
| FastASEM | 458 (23.1%) | 535 (26.9%) | 31 (1.6%) |
| FullContext | 691 (34.8%) | 707 (35.6%) | 37 (1.9%) |
| NoMemory | 53 (2.7%) | 313 (15.8%) | 7 (0.4%) |

- **2.4% dòng ASEM có verdict tự mâu thuẫn**: `judge_correct=true` nhưng `judge_error="contradiction"`,
  reasoning ghi nguyên văn *"specifies a date in 2023, which contradicts the gold answer of 2022"*.
  Nhiễu này **xấp xỉ chính hiệu số 2.6 pp** đang được tranh luận.
- Trong các dòng temporal được chấm đúng, **28% (ASEM) / 42% (FullContext)** không chứa con số của gold.
- Judge quá dễ dãi với câu diễn giải: gold `"Single"` được chấp nhận cho câu trả lời về "transition
  changed her relationships".
- Judge là **nhị phân** (242/446 = 0.5426), nên với n=1986 sai số ±2 pp.

## 9. Hạn chế của lần chạy & việc cần làm

Đã xác nhận trong code, ảnh hưởng trực tiếp tới tính công bằng của bảng điểm:

1. `eval/systems.py:build_asem_system` **không truyền `answer.direct_mode`** → ASEM v1 chạy đường
   `distil_and_answer` + `P_distil.txt` trong khi FastASEM chạy `direct_answer` + `P_temporal_qa.txt`.
   Hai hệ thống không cùng giao thức sinh câu trả lời.
2. Cùng chỗ đó **không truyền khối `retriever:`** → ASEM dùng default `max_link_hops=1`, config ghi 2.
3. `CATEGORY_NAMES` sai nhãn (mục trên).
4. `eval/static_eval.py` **không lưu `used_notes`** vào record → không thể phân tích truy hồi hậu nghiệm
   nếu không replay retriever.
5. Eval tĩnh **không cập nhật `q`** → `λ·z(q) ≡ 0` (guard `std==0` trong `retriever._zscore`), nên Stage 5
   (RL/utility) hoàn toàn không được đo trong giao thức này.
6. Chưa có baseline RAG chuẩn (BM25/dense top-k) hay hệ thống bộ nhớ khác (Mem0, A-Mem) → chưa tách
   được đóng góp riêng của cấu trúc ASEM.
7. Chưa ghi token/độ trễ theo từng dòng. Số đo từ tooling sẵn có (`scratch_diag/measure_context.py`):
   prompt trả lời p50 ≈ **ASEM 3.4k / FastASEM 0.5k / FullContext 7.6k token** → ASEM rẻ hơn ~2.2× mỗi câu.

### Thứ tự ưu tiên

| # | Việc | File |
|---|---|---|
| 1 | Sửa nhãn category (1→multi_hop, 3→open_domain, 4→single_hop) | `scripts/run_locomo10_experiments.py` |
| 2 | Bỏ quyền từ chối ở guideline 5, làm rõ guideline 3 | `data/prompts/P_temporal_qa.txt` |
| 3 | Truyền `direct_mode` + khối `retriever` khi build ASEM | `eval/systems.py` |
| 4 | Ghi `used_notes` / id note mang gold vào record | `eval/static_eval.py` |
| 5 | Tăng precision retriever ở single_hop (rerank top-20, chỉnh `k2`/`λ`) | `asem/retriever.py` |
| 6 | Note tổng hợp theo chủ đề cho multi_hop (union nhiều phiên) | `asem/batch_ingestion.py` |
| 7 | Judge bằng model khác + lọc verdict tự mâu thuẫn | `eval/llm_as_a_judge.py` |
| 8 | Báo cáo kèm bối cảnh "history đóng tại session evidence" | `README.md` (mục 5) |

## 10. Định dạng context truy vấn (mới)

Prompt trả lời không còn là một dòng JSON với UUID. Mỗi note là một block đánh số, đọc theo câu:

```text
[1] 20 April 2023 · said by Calvin · about: Calvin
      • Calvin learned a lot and received great advice from music industry professionals, including a producer
        who told him to stay true to himself and sound unique.
      • [+1 more fact(s) in this note, not shown]
      turn: "[Calvin] I learned a lot and got some great advice from professionals in the music industry."
      topics: calvin, advice, music industry, producer
      links: [5] same-topic, [6] extends  (+7 more not in this list: extends:3, same-topic:3, semantic:1)
```

- `[n]` là định danh note **và** là giá trị trả về trong `selected_ids` (số, không phải UUID → bỏ hẳn
  failure mode "copy sai UUID thì mất luôn khả năng chọn note").
- `description` được tách thành bullet theo câu, tối đa 4 câu / 420 ký tự/note; dấu `[+k more fact(s)]`
  cho biết note còn nội dung chưa in.
- Ngày được in dạng người đọc được (`20 April 2023`), `keywords` cắt còn 8, `content` chỉ in khi thật sự
  bổ sung thông tin.
- Loại bỏ UUID, `target_id`, `also_linked` rời, `—` escape.

Đo trên bank `ds_fixed` (ASEM, 61 câu): context nhỏ hơn **31.5%** (p50 1 978 → 1 394 token; max 3 320 →
1 774). Chi tiết + before/after: [report.md §9](report.md#9-thiết-kế-lại-context-truy-vấn-numbered-blocks),
artifact `scratch_diag/context_format_preview.md`.

Cùng lúc đó `asem/note.py` ép giới hạn ở write path (`cap_description` 600 ký tự cắt theo câu,
`cap_keywords` 12) và các prompt P1/P3/P4 yêu cầu ≤ 45 từ, REWRITE chứ không append — vì bước memory
evolution đã tạo ra những `description` dài ~1 700 ký tự. Caps chỉ áp dụng cho bank **ingest lại**.

---

## 11. Tóm tắt một dòng

ASEM **thắng vững** ở adversarial (+16.6 pp) và temporal (+12.1 pp); khoảng cách ở single_hop là
**retrieval precision** và đối thủ ở đó đang được trao đáp án nguyên văn ngay sát câu hỏi; 58% lỗi
multi_hop là **ingestion**; còn khác biệt 2.6 pp ở tổng thể nằm trong cùng độ lớn với **nhiễu của
judge 4B tự chấm chính nó**.
