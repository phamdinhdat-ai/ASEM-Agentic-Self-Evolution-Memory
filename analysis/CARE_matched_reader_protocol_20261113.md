# CARE matched reader ablation protocol

Frozen before the new dense-reader results are inspected. The 1,986 LoCoMo
questions, Qwen3-8B checkpoint, 8,192-token server window, temperature 0,
224-token answer cap, source-turn adapter, and benchmark scorer follow the
complete CARE comparison.

## Question

How much of the measured CARE result is associated with first-bundle
selection, and how does the conditional second pass affect each selection
strategy when the reader prompt and source-binding checks are held fixed?

## Arms

| Arm | Initial source bundle | Reader | Second pass |
|---|---|---|---|
| BM25-8 first | Original BM25 top 8 | CARE | Never; reconstructed from saved CARE first-pass trace |
| BM25-8 conditional | Original BM25 top 8 | CARE | Original CARE gate and expansion; saved complete run |
| Dense-8 first | Top 8 of frozen MiniLM raw-turn rankings | CARE | Never; saved first output of new run |
| Dense-8 conditional | Same Dense-8 first output | CARE | Original CARE gate and expansion; new run |

The new run uses the exact CARE reader method for every dense output. Each
question's dense first and conditional arms share one physical first call;
their reported calls and tokens are *virtual arm costs* and must not be added
together to estimate execution expenditure. Sparse scores are replayed from
the existing complete CARE logs, without another model call. The dense source
rankings come from the completed raw-turn SimRetrieval run and contain source
IDs only, never answers or gold labels. The second-pass expansion is the same
CARE implementation for both retrievers, although first-pass search requests
may differ.

## Analysis fixed in advance

Report LoCoMo aggregate, answerable score, category-5 accepted-refusal
accuracy, annotated-source recall, model calls, answer-model tokens, and
per-category results. Use 10,000 conversation-cluster bootstrap draws for
paired score differences. Main contrasts: Dense-8 first vs BM25-8 first;
BM25-8 conditional vs BM25-8 first; Dense-8 conditional vs Dense-8 first;
and Dense-8 conditional vs BM25-8 conditional. Report both positive and
negative score changes per question. The first contrast isolates the initial
retrieval choice under the same prompt and top-8 budget. The conditional
contrasts measure each pipeline's second-pass selection effect; they do not
separately identify expanded evidence from correction-prompt effects.

This is an in-dataset ablation, because the public ten histories were used
while developing CARE. It is not a held-out generalization result. The
category-5 metric detects accepted refusal phrases, not all unsupported
claims. BERTScore is complementary text similarity, not entailment.

## Additional validation required before a broad novelty claim

Run the same read policy on untouched LongMemEval histories and a second
backbone; evaluate claim-level source support on an independently judged
sample. Run a full-history reader only when the model window can contain all
turns, and compare quality against total token and wall-time costs. These
future results must not be inferred from the present 8K window baseline.
