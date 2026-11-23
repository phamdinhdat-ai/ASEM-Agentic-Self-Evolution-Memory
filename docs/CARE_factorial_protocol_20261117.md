# Pre-registered protocol: CARE factorial ablation and baselines (17/11/2026)

Written **before** any kernel of this run was launched. Results must be reported
against this file; any deviation is listed in the results document with its reason.

## Code under test

- Kaggle bundle SHA-256: `713d8ee816c4f48ff48f77449d03363a5a5ba6537889e2e028b89e0e3f2cd166`
  (built by `python -m scripts.prepare_care_factorial_kaggle`).
- Key files (SHA-256 prefix): `asem/care/hybrid.py` d99a3a1f6499fafc,
  `scripts/run_care_factorial.py` fe0bf46ecc324951, `scripts/embed_care_turns.py` 1d12e1ad4af61f09.
- Tests: `tests/test_care_hybrid.py` (lexical repair is call-for-call identical to
  CareBridge on 160 LoCoMo questions), `tests/test_care_factorial.py` (arms, call accounting, resume).

## Fixed setting (all arms)

Reader Qwen3-8B (pinned revision, vLLM, fp16, 2×T4, 8,192-token window, temperature 0,
non-thinking, ≤224 output tokens, `--max-num-seqs 1`). Encoder Qwen3-Embedding-0.6B (pinned,
fp16, last-token pooling, instruction-prefixed queries, docs ≤1,024 tokens), run on GPU inside
the worker. Evidence cap 10,000 characters per call; first bundle 8 turns; at most one repair.

## Arms (2 × 3 factorial + baselines)

| Arm | First bundle | Repair |
|---|---|---|
| `bm25_first` | BM25 top-8 | none |
| `bm25_lex` (= CARE, reference) | BM25 top-8 | lexical (CARE) |
| `bm25_hyb` | BM25 top-8 | hybrid (lexical + dense top-6 for the question and each search) |
| `dense_first` | dense top-8 | none |
| `dense_lex` | dense top-8 | lexical |
| **`dense_hyb` (proposed system)** | dense top-8 | hybrid |
| `rag_dense_top18` | plain RAG prompt, dense top-18, one call | — |
| `no_memory`, `full_context_8k` | LongMemEval only (LoCoMo reuses the earlier runs of the same reader) | — |

The proposed system is fixed here, before any result, as `dense_hyb`. All arms are reported.

## Data

- **Development: LoCoMo**, all 1,986 questions (used to develop CARE; results are development results).
- **Test: LongMemEval cleaned oracle**, all 500 questions, run once with this bundle.
  No parameter may be changed after LongMemEval results are seen.

## Primary metric

Mean per-question score over all questions of a benchmark: on LoCoMo the benchmark scorer
(aggregate); on LongMemEval the diagnostic score (normalized token F1 on answerable questions,
phrase-based abstention on the 30 `_abs` questions). Not the official judged LongMemEval accuracy.

## Primary hypotheses (on the test set, LongMemEval)

- **H1** `dense_hyb` − `bm25_lex` > 0 (the proposed system beats CARE).
- **H2** `dense_hyb` − `dense_lex` > 0 (hybrid repair helps given a dense first bundle).
- **H3** `dense_first` − `bm25_first` > 0 (the dense first bundle helps without repair).

Paired question bootstrap (10,000 draws). With three primary tests, a hypothesis is supported
only if the Bonferroni-adjusted 98.33% interval excludes zero; 95% intervals are also reported.
LoCoMo results are reported the same way with conversation-cluster bootstrap and per-conversation
win counts, as development evidence only.

## Secondary (reported, not used for claims of superiority)

Answerable score, correct refusals, refused answerable questions, calls, prompt and completion
tokens, annotated-evidence recall, per-category / per-type scores, comparison with
`rag_dense_top18`, `full_context_8k`, `no_memory`.

## Reproducibility checks

- `bm25_lex` on LoCoMo is compared answer-by-answer with the saved CARE run; on LongMemEval
  with the saved 500-question run. Agreement rate is reported.
- `dense_top8` from the GPU encoder is compared with the frozen rankings
  (`build/care/qwen3_0p6b_dense_rankings.json`).

## Known deviations from earlier runs (declared in advance)

- vLLM `--gpu-memory-utilization` is 0.80 (earlier 0.88) to fit the encoder on the GPU.
  With `--max-num-seqs 1` and greedy decoding this should not change outputs; the
  reproducibility check above measures it.
- The NoMemory prompt asks for a best guess (as in the earlier LoCoMo baseline), which
  penalizes its abstention score.
- The ASEM-Evolve write path is not part of this run.

---

## Amendment 1 (17/11/2026, before any full-run result; pilot outputs not inspected)

Reason: the user asked for graph knowledge (graph construction, layers, communities, graph
querying, influence). The first protocol had none. This amendment adds one graph arm and one
hypothesis. Nothing else changes: arms, metrics, test set and H1–H3 are as above.

- **Code under test is now** bundle SHA-256
  `87959f7f7656b5a6e0ecfc96eca9357b1efe1b6d15775d6b21f1d19cf9df55fc`
  (`asem/care/hybrid.py` 3e7368181093d3d5, `asem/care/graph_memory.py` 0029da74569fe05d,
  `scripts/run_care_factorial.py` 426528659f5392af, `scripts/run_care_kaggle.py` 3c513d734e4209c3).
  The v1 pilots ran the previous bundle and only test infrastructure.
- **Layered memory graph** (`asem/care/graph_memory.py`, built per history without an LLM):
  L0 turns with `next` and semantic 5-NN edges (Qwen3-Embedding-0.6B), L1 entities
  (capitalized tokens in 2 to 5% of turns), L2 sessions, L3 Louvain communities (seed 20261117).
  Query = personalized PageRank (restart 0.5, 5 dense seeds); a turn's PPR mass is its influence;
  ranking = 0.7 · z(dense) + 0.3 · z(PPR).
- **New arm `dense_graph`**: dense top-8 first bundle; repair = lexical expansion plus the top-6
  graph-ranked turns for the question and for each search (identical to `dense_hyb` except that
  the graph ranker replaces the dense ranker).
- **Choice of the graph setting, on LoCoMo (development) only** (`scripts/eval_graph_retrieval.py`;
  `analysis/CARE_graph_retrieval_locomo_dev*_20261117.json`). Recall of annotated evidence:
  at k=8, dense 0.6172, PPR 0.6050, mix λ=0.3/0.5/0.7 0.6137/0.6168/0.6184, community round-robin 0.4151;
  at k=18, dense 0.7246, PPR 0.6951, mix 0.5 0.7351, mix 0.7 0.7357. The graph adds almost nothing
  at k=8, so it is used in repair (wider pool), with λ=0.7 (best at both k). LongMemEval graph
  recall has not been computed and will be computed once, with this fixed setting.
- **H4** `dense_graph` − `dense_hyb` > 0 (graph knowledge adds value beyond plain dense additions).
- With four primary hypotheses, a hypothesis is supported only if the Bonferroni-adjusted
  **98.75%** interval excludes zero (replaces 98.33%).
- `networkx==3.4.2` is installed into the Kaggle runtime for Louvain.

---

## Amendment 2 (18/11/2026, before lme-s3 and lme-s4 were re-run)

Reason: `lme-s3` failed with `torch.OutOfMemoryError` at the start of its worker, while the
dense index for one long history was being encoded. `locomo-s4` failed the same way (same error, 48 MiB
request, 10 MiB free) under the old bundle; it was re-run under the new one. vLLM held most of the T4 memory
(`gpu_memory_utilization` 0.80), leaving too little for the encoder with an 8,192-token batch
budget. The failure happened before any answer of that shard was saved (2 events written). Shards
`lme-s0..s2` and `locomo-s0..s3` completed normally under the old bundle.

- **Change:** `scripts/run_care_factorial.py` builds the encoder with `token_budget=2048` instead of
  8,192. Nothing else in the worker, the arms, the prompts, the models or the scoring changes.
- **Code under test is now** bundle SHA-256 `bceca0e5ab4ca734116a30c9755b7e79aa701a4067cd7bbaf83360196ed5da80`;
  `scripts/run_care_factorial.py` f4d7570d3e7a5a18 (previously 426528659f5392af), other key files unchanged
  (`asem/care/hybrid.py` 3e7368181093d3d5, `asem/care/graph_memory.py` 0029da74569fe05d,
  `scripts/embed_care_turns.py` 1d12e1ad4af61f09).
- **Mixed code across shards:** `lme-s0..s2`, `locomo-s0..s3` ran bundle `87959f7f…` (token budget
  8,192); `lme-s3`, `lme-s4`, `locomo-s4` run bundle `bceca0e5…` (token budget 2,048). Combined
  `code_sha256` written by the worker for the new code: `837e0b259a638ed1`. Each report's manifest records
  its own `code_sha256`; the summary must show per-shard hashes.
- **Assumption, to be checked and not assumed:** a smaller padded batch changes only fp16 rounding in
  the encoder, so dense rankings should be unchanged except for rare ties. The check is the same as before:
  `bm25_lex` must reproduce the saved CARE answer on every question of the new shards, and the dense
  top-8 must match the frozen rankings for LoCoMo shards (LongMemEval has no frozen rankings; its dense
  top-8 is reported in the manifest).
- **Not changed:** the protocol's arms, hypotheses H1–H4, the development/test split, the metric and the
  test set.
- **Fallback if the new budget also fails:** lower the vLLM utilization to 0.70 in a further amendment
  before any new run.
