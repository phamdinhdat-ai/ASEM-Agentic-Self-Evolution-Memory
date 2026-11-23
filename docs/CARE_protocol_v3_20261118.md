# Protocol v3: confirmatory run of the dev-selected arm on LongMemEval-S (18/11/2026)

Written **before** any v3 kernel was launched. It supersedes the test role of the factorial
protocol (`CARE_factorial_protocol_20261117.md`, amendments 1–2), whose LongMemEval oracle results were
already observed. Nothing in this file may be changed after the first v3 result is seen.

## Why this run, and what it can and cannot show

- The factorial study chose `bm25_hyb` on LoCoMo (dev), where it scored 0.5405 against 0.5256 for
  CARE (`bm25_lex`). That choice was made on development data only.
- The LongMemEval **oracle** results were seen and used to discuss the design. They cannot confirm the
  choice, so they are not reused here.
- **LongMemEval-S contains the same 500 questions as the oracle file**, with each question's full
  haystack (median 48 sessions, 491 turns, range 396–616 turns) instead of the relevant sessions only.
  The questions and answers are therefore *not new*. What is new is the setting: retrieval must find
  the evidence among the whole history. The claim this run can support is therefore a retrieval-setting
  claim on the same questions, not a claim about unseen questions. A fully independent test requires a
  different benchmark, which is outside this protocol.

## Data

- File `datasets/longmemeval/longmemeval_s_cleaned.json`, SHA-256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`, 277,383,467 bytes.
  Checked against the pinned source
  `xiaowu0162/longmemeval-cleaned@98d7416c24c778c2fee6e6f3006e7a073259d48f`; added to
  `kaggle/vllm-experiment-lock.json` with the same hash, so the kernel verifies it on download.
- 500 questions, 30 abstention (`_abs`), the same IDs as the oracle file.
- Shards are the existing hash split of `question_id` (sizes 98, 110, 100, 97, 95).

## Code under test

- Bundle SHA-256 `1459b45ac907794c283cbf76f17946a1cdb84e97b1245880a80f9780c72a8144`
  (`python -m scripts.prepare_care_v3_kaggle`). The only change from bundle `bceca0e5…` is the added
  lock entry; the worker, arms and prompts are identical to amendment 2 of the factorial protocol.
- Encoder batch: 2,048 padded tokens (amendment 2). Reader vLLM utilization 0.80.

## Fixed setting

As in the factorial protocol: Qwen3-8B reader (pinned, vLLM, 8,192-token window, greedy, non-thinking,
≤224 output tokens), Qwen3-Embedding-0.6B encoder (pinned, fp16, on GPU), 10,000-character evidence cap,
first bundle 8 turns, at most one repair. Gold is used only after inference.

## Arms

All arms of the factorial worker run, because the worker computes them together. Reporting uses:

| Role | Arm |
|---|---|
| **Confirmatory comparator** | `bm25_lex` (CARE) |
| **Confirmatory system** | `bm25_hyb` (chosen on LoCoMo dev) |
| Secondary | `dense_hyb`, `dense_graph`, `rag_dense_top18`, `bm25_first`, `dense_first` |
| Reference | `no_memory`, `full_context_8k` (truncated to 8K: keeps the first 5 turns and the most recent suffix; not a full-context system, and reported as such) |

## Primary hypothesis and decision rule (fixed now)

- **H-main:** `bm25_hyb` − `bm25_lex` > 0 on the 500 LongMemEval-S questions, using the diagnostic
  score (normalized token F1 on answerable questions, phrase-based abstention on the 30 `_abs`
  questions), as in the factorial protocol.
- Interval: question-level paired bootstrap, 10,000 draws, 95% two-sided.
- **Confirmed** if the 95% interval lies entirely above 0. **Not confirmed** if it contains 0.
  **Reversed** if it lies entirely below 0.
- No other contrast is a basis for a claim. Secondary contrasts are reported with 95% intervals and
  labelled exploratory.

## Reproducibility checks (fixed now)

- `bm25_lex` must reproduce the saved CARE answers on LongMemEval-S. These answers are not saved, so
  the check is instead: `bm25_lex` on the oracle shards must still match exactly (already verified,
  500/500), and each v3 shard must record its code hash in its manifest.
- Dense top-8 on LongMemEval-S has no frozen reference; its recall against gold is reported.

## Deviations (declared before the run)

- The reader sees the same prompts as before; no prompt change.
- The graph arm is reported as is. On the oracle it changed the bundle in only 9 of 500 questions;
  H-main does not depend on it.
- Pilot first: the 4-question pilot must complete with no error and with `bm25_lex` answers recorded
  before any full shard is launched. A pilot failure stops the run and is reported.
