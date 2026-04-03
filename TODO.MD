# ASEM Implementation — TODO List

Track progress by checking off tasks. Ordered by dependency (earlier tasks unblock later ones).

---

## Phase 0 — Project Setup

- [x] Initialise repository with the directory structure from `plan.md`
- [x] Set up `pyproject.toml` / `requirements.txt`:
  - `torch`, `transformers`, `sentence-transformers`
  - `faiss-cpu` (or `faiss-gpu`) for ANN search
  - `trl` for GRPO training
  - `datasets`, `evaluate` (HuggingFace) for benchmark loading
  - `pydantic` for Note schema validation
  - `pytest` for tests
  - **LangChain / LangGraph backend (optional):**
    `langchain`, `langchain-core`, `langchain-community`, `langgraph`
    + provider package matching chosen LLM (e.g. `langchain-openai`, `langchain-anthropic`,
    `langchain-huggingface`)
  - **HuggingFace backend:** `transformers`, `sentence-transformers`, `accelerate`
    + `bitsandbytes` (optional, for 4-bit/8-bit quantisation)
- [x] Configure logging and experiment tracking (e.g. wandb or mlflow)
- [x] Write `configs/default.yaml` with all hyperparameters and backend selection:
  `k1=20, k2=5, k=5, δ=0.30, λ=0.40, α=0.10, q0=0.50`
  `inference.backend: "langchain"` or `"huggingface"` (and sub-config for each)
- [x] Download / prepare base model: `Qwen2.5-7B-Instruct`

---

## Phase 0.5 — Inference Backend Abstraction

> **Must be completed before Phase 1.** All ASEM components receive the backend via constructor
> injection and must never import a model library directly.

- [x] Define `InferenceBackend` ABC in `asem/backends/base.py`
  ```python
  class InferenceBackend(ABC):
      def generate(self, prompt: str, **kwargs) -> str: ...
      def embed(self, text: str) -> np.ndarray: ...
  ```
- [x] Implement `build_backend(config: dict) -> InferenceBackend` factory in
  `asem/backends/__init__.py`; reads `inference.backend` key from config

### LangChain / LangGraph backend (`asem/backends/langchain_backend.py`)
- [x] Implement `LangChainBackend(InferenceBackend)`
  - `generate()` → `BaseChatModel.invoke(HumanMessage(prompt)).content`
  - `embed()` → `Embeddings.embed_query(text)` → `np.ndarray`
- [x] Add `LangChainBackend.from_config(cfg)` classmethod:
  - Instantiate correct LLM class from `cfg.provider`
    (`ChatOpenAI`, `ChatAnthropic`, `ChatHuggingFace`, `ChatOllama`, …)
  - Instantiate correct embedder
    (`OpenAIEmbeddings`, `HuggingFaceEmbeddings`, …)
- [ ] **(Optional)** Implement `LangGraphPipeline` in `asem/backends/langgraph_pipeline.py`
  - Define `ASEMState` TypedDict (holds interaction, memory bank ref, query, answer, reward)
  - Wire ASEM stages as LangGraph nodes:
    `note_construct → write_op → link_evolve → retrieve → distil → utility_update`
  - Add conditional edge: if `C(q) = ∅` → skip to backbone-only node
  - Enable checkpointing via `MemorySaver` for multi-turn session persistence
  - Expose `pipeline.stream(state)` for streaming token output

### HuggingFace backend (`asem/backends/huggingface_backend.py`)
- [x] Implement `HuggingFaceBackend(InferenceBackend)`
  - `generate()` → `transformers.pipeline("text-generation", …)(prompt)[0]["generated_text"]`
  - `embed()` → `SentenceTransformer.encode(text)` → `np.ndarray`
- [x] Add `HuggingFaceBackend.from_config(cfg)` classmethod:
  - Load model with `device_map="auto"`, optional `load_in_4bit=True` via `bitsandbytes`
  - Load embedder from `cfg.embedder_name` (default: `"sentence-transformers/all-MiniLM-L6-v2"`)
- [x] Confirm: **GRPO training always uses HuggingFace backend** — training code must
  instantiate `HuggingFaceBackend` directly, independent of `configs/default.yaml`

### Contract compliance tests (`tests/test_backends.py`)
- [x] Test that both backends satisfy the ABC contract (generate returns str, embed returns
  `np.ndarray` of correct shape)
- [x] Test `build_backend` factory resolves correct class for each config value
- [x] Test LangChain backend with a mock `BaseChatModel` (no real API key required)
- [x] Test HuggingFace backend with a tiny model (e.g. `"sshleifer/tiny-gpt2"`) to keep CI fast

---

## Phase 1 — Note Schema & Memory Bank (Stage 1 + Storage)

- [x] Define `Note` dataclass in `asem/note.py`
  - Fields: `id`, `c`, `t`, `K`, `G`, `X`, `e`, `L`, `z`, `q`
  - Include serialisation / deserialisation (JSON + numpy array)
- [x] Write prompts file `data/prompts/P1_note_construction.txt`
  - Must instruct LLM to output structured JSON with keys: `keywords`, `tags`, `description`
- [x] Implement `NoteConstructor.build(content, timestamp) -> Note`
  - Accept `backend: InferenceBackend` in constructor (injected, not imported directly)
  - Call `backend.generate(P1_prompt)` → parse `K`, `G`, `X`
  - Call `backend.embed(concat(c, K, G, X))` → `e`
  - Call `backend.embed(c)` → `z`
  - Set `L = []`, `q = q0`
- [x] Implement `MemoryBank` in `asem/memory_bank.py`
  - Backend: FAISS index for ANN search on `e`
  - Metadata store (SQLite or dict) for `K`, `G`, `q`, `z`, `L`
  - Methods: `add(note)`, `update(note_id, delta)`, `delete(note_id)`, `ann_search(vec, k)`
  - Persistence: `save(path)` and `load(path)`
- [x] Unit tests: construct a note from sample text; verify all fields populated; save/load bank

---

## Phase 2 — RL Write Operations — Memory Manager (Stage 2)

- [x] Define `Op` enum: `ADD`, `UPDATE`, `DELETE`, `NOOP`
- [x] Implement `MemoryManager.select_op(x, M_old) -> (Op, Note | None)`
  - At inference: call fine-tuned model with structured prompt → parse op + target
  - Include a heuristic fallback (vanilla LLM prompt) for use before training
- [x] Build training data pipeline in `training/train_manager.py`
  - Input: sequential fact pairs (complementary, contradictory, novel, redundant)
  - Sample G=8 candidate `(op, target)` pairs per state via current policy
  - Call frozen Answer Agent to score each resulting memory bank → exact-match reward `r`
  - Compute group-relative advantage: `A_i = (r_i − mean(r)) / std(r)`
- [x] Implement GRPO trainer in `training/grpo.py`
  - Loss: `J(θ) = E[(1/G) Σ ρ_i · A_i  −  β · KL(π_θ ∥ π_ref)]`
  - Use `trl.GRPOTrainer` or implement custom training loop
  - Log consolidation rate (UPDATE ratio) as training diagnostic
- [ ] Train Memory Manager; save checkpoint to `checkpoints/memory_manager/`
- [ ] Ablation test: compare RL manager vs vanilla LLM manager on complementary-fact pairs;
  confirm RL manager issues UPDATE (not ADD+DELETE)

---

## Phase 3 — Dynamic Linking & Bidirectional Evolution (Stage 3)

- [x] Write prompts:
  - `data/prompts/P2_link_generation.txt` — instructs LLM to identify semantic/causal/temporal
    relationships between `m_new` and each neighbour; output list of `{source, target, relation}`
  - `data/prompts/P3_memory_evolution.txt` — instructs LLM to revise `m_j`'s `K`, `G`, `X`
    fields in light of `m_new`; output updated JSON
- [x] Implement `LinkEvolver.link_and_evolve(m_new, M)` in `asem/link_evolver.py`
  - Step 1: `M_near = M.ann_search(m_new.e, k=k)` → top-k neighbours
  - Step 2: call LLM with P2 → parse relationships → update `m_new.L` and each neighbour's `L`
  - Step 3: for each `m_j ∈ M_near`, call LLM with P3 → parse revised attributes → `M.update(m_j.id, delta)`
- [x] Unit tests:
  - Add two related notes; verify bidirectional links created in both
  - Evolve a note; verify `K`, `G`, `X` of neighbour are updated
- [ ] Ablation: run retrieval with and without evolution; confirm multi-hop accuracy impact
  (paper reports −11.3% multi-hop / −2.1% single-hop when removed)

---

## Phase 4 — Two-Phase Hybrid Retrieval + Distillation (Stage 4)

- [x] Implement `HybridRetriever.retrieve(query, M)` in `asem/retriever.py`
  - Phase A: encode query → `e_q`; `C(q) = TopK_{k1}({m_i | sim(e_q, e_i) > δ}, by sim)`
  - Handle empty `C(q)` → return `[]` (pipeline falls back to backbone-only)
  - Phase B: z-score normalise `sim` scores within `C(q)`; z-score normalise `q` values;
    `score = (1−λ)·sim_norm + λ·q_norm`; `M_ret = TopK_{k2}(C(q), by score)`
- [x] Implement `AnswerAgent.distil_and_answer(query, candidates)` in `asem/answer_agent.py`
  - At inference: call fine-tuned model → select minimal subset + generate answer
  - Include vanilla LLM baseline version for pre-training use
- [x] Build Answer Agent training pipeline in `training/train_answer.py`
  - Input: `(query, M_ret, gold_answer)` triples
  - Reward: exact-match between `ŷ` and `y_gold`
  - Use same GRPO objective as Memory Manager training
- [ ] Train Answer Agent; save checkpoint to `checkpoints/answer_agent/`
- [x] Unit tests: retrieve from a small bank; verify Phase A respects `δ`; verify Phase B
  re-ranks by composite score; verify z-score normalisation is per-pool

---

## Phase 5 — Utility Update (Stage 5)

- [x] Implement `UtilityUpdater.update(reward, used_notes)` in `asem/utility_updater.py`
  - EMA rule: `q_i ← q_i + α · (r − q_i)` for each `m_i ∈ M_distil`
  - Call `M.update(m_i.id, {'q': new_q})`
- [x] Implement experience consolidation:
  - LLM summarises `(query, ŷ, r)` → `e_new` string
  - Append new note `(z_q, e_new, q0)` to `M`
- [x] Unit tests: simulate 10 interactions with known rewards; verify `q` converges toward
  `E[r]` as predicted by Theorem 5.1; verify variance stays within `ασ²/(2−α)` bound

---

## Phase 6 — Full Pipeline Integration (Algorithm 1)

- [x] Implement `ASEMPipeline` in `asem/pipeline.py` combining all stages:
  ```
  write_path(c):   Stage1 → Stage2 → Stage3
  read_path(q):    Stage4a → Stage4b → Stage4c → return ŷ
  update_path(r):  Stage5
  ```
- [x] Add full pipeline integration test with a 5-turn simulated conversation
- [x] Verify correct stage ordering matches Algorithm 1 in the paper
- [x] Profile memory and latency per stage; identify bottlenecks
  (Stage 3 evolution is O(k) LLM calls — most expensive)

---

## Phase 7 — Baselines

- [x] Implement all 6 baselines in `eval/baselines.py` behind a common interface
  `baseline.answer(query, history) -> str`:
  - [x] `NoMemory` — backbone-only, no history
  - [x] `FullContext` — all prior turns concatenated as prefix context
  - [x] `SimRetrieval` — flat FAISS retrieval, top-k by cosine similarity
  - [x] `AtomicLinking` — Stages 1+3 only (notes + links, no RL, no Q-value)
  - [x] `RLManagerOnly` — Stages 1+2 only (RL write ops, similarity retrieval, flat entries)
  - [x] `ValueRetrievalOnly` — Stage 4+5 on IEU triplets (no rich notes, no linking)

---

                                                                                                                   ## Phase 8 — Evaluation

- [x] Set up benchmark loaders in `eval/evaluate.py`:
  - [x] LongMemEval (500 conversations, single/two/multi-hop splits)
  - [x] LoCoMo (10-session dialogues)
  - [x] PersonalMemBench (30-day simulated assistant)
- [x] Implement metrics:
  - [x] Exact Match (EM) — primary
  - [x] ROUGE-L
  - [x] BERTScore-F1
  - [x] Human evaluation setup (100-question blind annotation interface)
- [ ] Run full evaluation: ASEM vs all 6 baselines × 3 benchmarks × 5 seeds
- [ ] Generate results table matching Table 1 format from paper
- [ ] Run ablations:
  - [ ] Remove Stage 3 (link+evolve) → measure multi-hop vs single-hop delta
  - [ ] Remove z-score normalisation in Phase B → measure score instability
  - [ ] Vary `λ` from 0 to 1 → measure utility weight sensitivity
  - [ ] Vary `α` (0.01, 0.05, 0.10, 0.30) → measure variance vs convergence trade-off

---

## Phase 9 — Documentation & Cleanup

- [x] Write docstrings for all public classes and methods
- [x] Add `README.md` with quickstart, training instructions, evaluation commands
- [x] Add `THEORY.md` summarising Theorem 5.1 and Proposition 5.1 with proofs
- [ ] Ensure all unit tests pass (`pytest tests/`)
- [ ] Add type annotations throughout (`mypy` clean)

---

## Dependency Graph

```
Phase 0
  └─ Phase 0.5 (InferenceBackend abstraction — blocks everything downstream)
       └─ Phase 1 (Note + MemoryBank)
            ├─ Phase 2 (Memory Manager training requires MemoryBank + AnswerAgent inference)
            ├─ Phase 3 (LinkEvolver requires MemoryBank)
            ├─ Phase 4 (Retriever requires MemoryBank; AnswerAgent training requires Retriever)
            └─ Phase 5 (UtilityUpdater requires MemoryBank)
                 └─ Phase 6 (Full Pipeline — all stages)
                      ├─ Phase 7 (Baselines)
                      └─ Phase 8 (Evaluation — requires Pipeline + Baselines)
                           └─ Phase 9 (Docs)
```

> **Note for agents:** Phases 2, 3, 4, and 5 can be developed in parallel once Phase 1 is complete.
> Phase 6 requires all of 2–5. Do not start evaluation (Phase 8) before the full pipeline passes
> integration tests (Phase 6).
>
> **Backend note:** The LangGraph pipeline wiring (Phase 0.5 optional task) can be deferred until
> after Phase 6, as it wraps the completed pipeline rather than replacing it. The core
> `InferenceBackend` ABC and both concrete implementations must be done before Phase 1.