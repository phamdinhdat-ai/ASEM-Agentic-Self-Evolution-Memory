# ASEM: Agentic Self-Evolving Memory — Implementation Plan

## Overview

ASEM is a five-stage memory framework for LLM agents that maintains a **living knowledge
network** across sessions. It solves three root causes of agent statelesness:

| Root Cause | Solution |
|---|---|
| Semantic impoverishment of raw logs | Multi-attribute atomic note schema (Stage 1) |
| Heuristic append-only write policies | RL-trained Memory Manager using GRPO (Stage 2) |
| Utility-blind similarity retrieval | Two-phase hybrid retrieval + Q-value scoring (Stage 4) |

The backbone LLM is **never fine-tuned**. All adaptation happens through the external memory
bank `M` and its learned utility estimates `{q_i}`.

---

## Inference Backend

At runtime ASEM wraps every LLM call through a **pluggable backend abstraction**. The operator
selects either **LangChain / LangGraph** or **HuggingFace** at startup via config; all ASEM
stages remain unchanged.

### Backend selection

```yaml
# configs/default.yaml
inference:
  backend: "langchain"   # "langchain" | "huggingface"

  # --- LangChain / LangGraph options ---
  langchain:
    provider: "openai"           # openai | anthropic | huggingface_hub | ollama | …
    model: "gpt-4o"
    temperature: 0.0
    use_langgraph: true          # wrap pipeline as a LangGraph StateGraph

  # --- HuggingFace options ---
  huggingface:
    model_name_or_path: "Qwen/Qwen2.5-7B-Instruct"
    device_map: "auto"
    load_in_4bit: false
    pipeline_task: "text-generation"
    max_new_tokens: 512
    temperature: 0.0
```

### Backend contract

Every backend must implement the same interface:

```python
class InferenceBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Single-turn text generation."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a dense vector for the given text."""
```

### LangChain / LangGraph backend

- **LLM calls** routed through `langchain_core.language_models.BaseChatModel`
  (swappable provider: OpenAI, Anthropic, HuggingFace Hub, Ollama, etc.)
- **Embeddings** via `langchain_core.embeddings.Embeddings`
  (e.g. `OpenAIEmbeddings`, `HuggingFaceEmbeddings`)
- **Optional LangGraph integration:** the full ASEM pipeline (write path + read path + update
  path) can be expressed as a `langgraph.graph.StateGraph` where each ASEM stage is a node.
  This enables streaming, checkpointing, human-in-the-loop pauses, and parallel stage execution.

```python
# asem/backends/langchain_backend.py
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

class LangChainBackend(InferenceBackend):
    def __init__(self, llm: BaseChatModel, embedder: Embeddings): ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...

# asem/backends/langgraph_pipeline.py  (optional LangGraph wiring)
from langgraph.graph import StateGraph, END
# nodes: note_construct → write_op → link_evolve → retrieve → distil → utility_update
```

### HuggingFace backend

- **LLM calls** via `transformers.pipeline("text-generation", model=..., device_map="auto")`
- **Embeddings** via `sentence_transformers.SentenceTransformer`
- Supports 4-bit / 8-bit quantisation through `bitsandbytes` for GPU-constrained deployments
- GRPO training (Stages 2 & 4) **always uses the HuggingFace backend** regardless of
  inference backend selection (training requires direct model access)

```python
# asem/backends/huggingface_backend.py
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

class HuggingFaceBackend(InferenceBackend):
    def __init__(self, model_name: str, embedder_name: str, **kwargs): ...
    def generate(self, prompt: str, **kwargs) -> str: ...
    def embed(self, text: str) -> np.ndarray: ...
```

### Backend factory

```python
# asem/backends/__init__.py
def build_backend(config: dict) -> InferenceBackend:
    if config["backend"] == "langchain":
        return LangChainBackend.from_config(config["langchain"])
    elif config["backend"] == "huggingface":
        return HuggingFaceBackend.from_config(config["huggingface"])
    raise ValueError(f"Unknown backend: {config['backend']}")
```

All ASEM components receive the backend via constructor injection — no stage calls a model
library directly.

### Backend comparison

| Capability | LangChain / LangGraph | HuggingFace |
|---|---|---|
| Provider flexibility | ✓ (OpenAI, Anthropic, Ollama, …) | ✗ (local models only) |
| Streaming output | ✓ (native) | ✓ (via `streamer`) |
| LangGraph orchestration | ✓ (StateGraph, checkpoints) | ✗ |
| GRPO training | ✗ (inference only) | ✓ (required for MM + AA) |
| 4-bit quantisation | ✗ | ✓ (`bitsandbytes`) |
| Offline / air-gapped | ✗ (provider API needed) | ✓ |
| Unified embedding API | ✓ | ✓ (`sentence-transformers`) |

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      ASEM Pipeline                       │
│                                                          │
│  New Interaction c                                       │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐    Stage 1: Note Construction          │
│  │  NoteSchema │  ← LLM generates K, G, X; encoder → e │
│  └──────┬──────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐    Stage 2: RL Write Operations        │
│  │   MM (πθ)  │  ← GRPO-trained: ADD/UPDATE/DELETE/NOOP│
│  └──────┬──────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐    Stage 3: Dynamic Linking & Evolution│
│  │  Link+Evolve│  ← k-NN neighbours → link gen + evolve│
│  └──────┬──────┘                                        │
│         │                                                │
│  Query q│                                               │
│         ▼                                                │
│  ┌─────────────┐    Stage 4: Two-Phase Retrieval        │
│  │  Phase A+B  │  ← Sim filter → Q-value re-rank        │
│  │  + Distill  │  ← Answer Agent distils noise          │
│  └──────┬──────┘                                        │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐    Stage 5: Utility Update             │
│  │  EMA Update │  ← q ← q + α(r − q)                   │
│  └──────┬──────┘                                        │
│         │                                                │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │            InferenceBackend (pluggable)          │    │
│  │  ┌─────────────────┐  ┌────────────────────┐    │    │
│  │  │ LangChain /     │  │  HuggingFace        │    │    │
│  │  │ LangGraph       │  │  (transformers +    │    │    │
│  │  │ (multi-provider)│  │   sentence-trans.)  │    │    │
│  │  └─────────────────┘  └────────────────────┘    │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Memory Note Schema
Each note `m_i` is a 9-tuple:
```
m_i = (c_i, t_i, K_i, G_i, X_i, e_i, L_i, z_i, q_i)
```

| Field | Type | Description |
|---|---|---|
| `c` | str | Raw interaction content |
| `t` | datetime | Timestamp |
| `K` | list[str] | LLM-generated keywords |
| `G` | list[str] | Categorical tags |
| `X` | str | Contextual description |
| `e` | vector(d) | Dense embedding of all text fields |
| `L` | list[note_id] | Bidirectional semantic link set |
| `z` | vector(d) | Intent embedding (from raw `c`) |
| `q` | float | Learned utility / Q-value |

### Memory Bank
```
M_t = { m_i }  for i in 1..N_t
```
Stored in a vector database with support for both ANN search (by `e`) and metadata filtering
(by `K`, `G`).

---

## Stage-by-Stage Specification

### Stage 1 — Multi-Attribute Atomic Note Construction
- **Input:** raw content `c`, timestamp `t`
- **Process:**
  1. Call backbone LLM with prompt `P1` → generate `K`, `G`, `X`
  2. Encode `concat(c, K, G, X)` with sentence encoder → `e`
  3. Encode `c` alone → intent embedding `z`
  4. Initialise utility `q = q0` (default 0.5)
- **Output:** fully populated note `m_new`

### Stage 2 — RL-Driven Memory Write Operations
- **Input:** extracted information `x` from `c`, top-k neighbours `M_old`
- **Process:**
  1. Memory Manager `π_θ` selects `(op, target)` from `{ADD, UPDATE, DELETE, NOOP}`
  2. Apply operation to `M`
- **Training:** GRPO with group size G; reward = exact-match score of frozen Answer Agent
  ```
  J(θ) = E[ (1/G) Σ ρ_i · A_i  −  β · KL(π_θ ∥ π_ref) ]
  A_i  = (r_i − mean(r)) / std(r)
  ```
- **Key behaviour:** learns UPDATE (consolidation) over ADD+DELETE (fragmentation)

### Stage 3 — Dynamic Linking & Bidirectional Memory Evolution
- **Input:** `m_new`, memory bank `M`
- **Process:**
  1. Find k nearest neighbours by cosine similarity on `e` → `M_near`
  2. LLM with prompt `P2` generates free-form relationship descriptions → populate `L_new`
  3. For each `m_j ∈ M_near`: LLM with prompt `P3` revises `m_j` attributes in light of `m_new`
     → replace `m_j` with updated `m_j*` in `M`
- **Note:** links are schema-free (semantic, causal, temporal — whatever the LLM identifies)

### Stage 4 — Two-Phase Hybrid Retrieval with Memory Distillation
- **Input:** query `q`, memory bank `M`
- **Phase A (Recall):**
  ```
  C(q) = TopK_{k1}({ m_i | sim(e_q, e_i) > δ }, by sim)
  ```
  Fallback to backbone-only if `C(q) = ∅`
- **Phase B (Value-aware re-rank):**
  ```
  score(q, m_i) = (1 − λ) · sim_norm(e_q, e_i)  +  λ · q_norm(z_i, m_i)
  M_ret = TopK_{k2}(C(q), by score)
  ```
  Both components z-score normalised within `C(q)` to prevent magnitude dominance.
- **Distillation:**
  ```
  (M_distil, ŷ) = π_ϕ(q, M_ret)
  ```
  Answer Agent selects minimal sufficient subset before reasoning.

### Stage 5 — Non-Parametric Runtime Utility Update
- **Input:** reward `r ∈ [−1, 1]`, distilled notes `M_distil`
- **Update rule (EMA):**
  ```
  q_i ← q_i + α · (r − q_i)    for each m_i ∈ M_distil
  ```
- **Experience consolidation:** LLM summarises trajectory `(q, ŷ, r)` → append new note
  `(z_q, e_new, q0)` to `M`
- **Convergence guarantee:** `E[q_t] → β(z, m)` at rate `(1−α)^t`; asymptotic variance ≤ `ασ²/(2−α)`

---

## Hyperparameters (reference defaults from paper)

| Parameter | Value | Role |
|---|---|---|
| `k1` | 20 | Phase A candidate pool size |
| `k2` | 5 | Phase B final retrieved count |
| `k` | 5 | Linking neighbourhood size |
| `δ` | 0.30 | Phase A similarity threshold |
| `λ` | 0.40 | Utility weight in composite score |
| `α` | 0.10 | EMA learning rate |
| `q0` | 0.50 | Initial utility prior |

---

## Component Interfaces

```python
class NoteConstructor:
    def build(self, content: str, timestamp: datetime) -> Note: ...

class MemoryManager:          # π_θ  — GRPO-trained
    def select_op(self, x: str, M_old: list[Note]) -> tuple[Op, Note | None]: ...

class LinkEvolver:
    def link_and_evolve(self, m_new: Note, M: MemoryBank) -> None: ...

class HybridRetriever:
    def retrieve(self, query: str, M: MemoryBank) -> list[Note]: ...

class AnswerAgent:             # π_ϕ  — GRPO-trained
    def distil_and_answer(self, query: str, candidates: list[Note]) -> tuple[list[Note], str]: ...

class UtilityUpdater:
    def update(self, reward: float, used_notes: list[Note]) -> None: ...

class MemoryBank:
    def add(self, note: Note) -> None: ...
    def update(self, note_id: str, delta: dict) -> None: ...
    def delete(self, note_id: str) -> None: ...
    def ann_search(self, vector: np.ndarray, k: int) -> list[Note]: ...
```

---

## Training Pipeline

### Memory Manager Training
1. Collect trajectories: present sequential facts to a vanilla LLM manager → capture
   fragmentation failures as negative examples
2. Fine-tune with GRPO: sample G=8 candidate operations per state; reward from Answer Agent
   exact-match; KL penalty against reference policy
3. Evaluate: consolidation rate (UPDATE vs ADD+DELETE ratio on complementary fact pairs)

### Answer Agent Training
1. Collect retrieval sets `M_ret` and gold answers
2. Fine-tune with GRPO: reward = exact-match between `ŷ` and `y_gold`
3. Both agents share the same base model initialisation (Qwen2.5-7B-Instruct in the paper)

---

## Evaluation

### Benchmarks
- **LongMemEval** — 500 multi-session conversations, up to 50 turns; single/two/multi-hop splits
- **LoCoMo** — 10-session dialogues, ~300 turns; cross-session synthesis + contradiction resolution
- **PersonalMemBench** — 30-day simulated assistant; preference tracking + factual consistency

### Metrics
- Primary: Exact Match (EM) accuracy
- Secondary: ROUGE-L, BERTScore-F1, blind human evaluation (100-question subset)
- All runs: 5 random seeds, report mean ± std

### Baselines to implement
1. No-Memory (backbone only)
2. Full-Context (oracle — all history in context)
3. Sim-Retrieval (flat dense retrieval)
4. Atomic-Linking (notes + links, no RL ops or utility retrieval)
5. RL-Manager-Only (RL write ops, similarity retrieval, flat entries)
6. Value-Retrieval-Only (Q-value retrieval, no rich notes or linking)

---

## Repository Structure (suggested)

```
asem/
├── data/
│   ├── benchmarks/          # LongMemEval, LoCoMo, PersonalMemBench
│   └── prompts/             # P1 (note construction), P2 (link gen), P3 (evolution)
├── asem/
│   ├── backends/
│   │   ├── __init__.py      # build_backend() factory
│   │   ├── base.py          # InferenceBackend ABC
│   │   ├── langchain_backend.py   # LangChain/LangGraph implementation
│   │   └── huggingface_backend.py # HuggingFace implementation
│   ├── note.py              # Note dataclass + NoteConstructor
│   ├── memory_bank.py       # MemoryBank (vector DB wrapper)
│   ├── memory_manager.py    # MemoryManager (π_θ) + Op enum
│   ├── link_evolver.py      # LinkEvolver (Stage 3)
│   ├── retriever.py         # HybridRetriever (Stages 4a + 4b)
│   ├── answer_agent.py      # AnswerAgent (π_ϕ, distillation + answer)
│   ├── utility_updater.py   # UtilityUpdater (Stage 5 EMA)
│   └── pipeline.py          # Full ASEM pipeline (Algorithm 1)
├── training/
│   ├── grpo.py              # GRPO trainer (shared by MM and AA)
│   ├── train_manager.py     # Memory Manager training loop
│   └── train_answer.py      # Answer Agent training loop
├── eval/
│   ├── evaluate.py          # Benchmark runner
│   └── baselines.py         # All 6 baseline implementations
├── configs/
│   └── default.yaml         # Hyperparameters + backend selection
└── tests/
    ├── test_backends.py     # Backend contract compliance tests
    └── ...
```

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Frozen backbone LLM | Prevents catastrophic forgetting; all adaptation is non-parametric |
| Pluggable inference backend | Decouples ASEM logic from model library; supports LangChain/LangGraph for multi-provider flexibility and HuggingFace for local/offline deployment |
| LangGraph optional wiring | Enables streaming, checkpointing, and parallel stage execution without changing ASEM stage code |
| HuggingFace always used for training | GRPO requires direct model weight access; inference backend does not affect training path |
| Free-form link schema | Allows novel relationship types at runtime; no schema migration required |
| Z-score normalisation in Phase B | Prevents magnitude dominance between sim and Q-value components |
| EMA utility update (not TD) | Simpler Monte Carlo backup; handles non-stationary reward distributions |
| GRPO (not PPO) | No explicit value baseline needed; advantages normalised within group |
| Joint encoding `concat(c, K, G, X)` for `e` | Captures full semantic footprint including implicit structure |