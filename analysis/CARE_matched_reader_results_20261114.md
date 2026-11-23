# CARE matched-reader ablation: complete LoCoMo result

All five Kaggle shards completed on the same 1,986 LoCoMo questions. The
pre-registered protocol is in `CARE_matched_reader_protocol_20261113.md`;
machine-readable results are in `CARE_matched_reader_results_20261113.json`.
The dataset, Qwen3-8B checkpoint, 8,192-token window, 224-token answer cap,
reader prompt, source-binding checks and top-eight first-bundle size are
fixed. First-pass BM25 outputs come from saved CARE traces. Dense first and
conditional outputs share one physical first call per question; virtual
per-arm resource totals count the calls each arm would require alone.

| Initial selector / policy | Aggregate | Answerable F1 | Correct refusals | Annotated source recall | Model calls | Answer-model tokens |
|---|---:|---:|---:|---:|---:|---:|
| BM25-8, first pass | 0.4588 | 0.3489 | 374/446 | 0.5353 | 1,986 | 1,418,914 |
| Dense-8, first pass | 0.4168 | 0.2908 | 380/446 | 0.4576 | 1,986 | 1,420,283 |
| BM25-8, conditional repair | **0.5256** | **0.4648** | 328/446 | **0.7033** | 2,990 | 3,647,127 |
| Dense-8, conditional repair | 0.4994 | 0.4415 | 312/446 | 0.6919 | 3,056 | 3,813,955 |

Paired, conversation-cluster bootstrap (10,000 draws, 10 clusters):

| Contrast | Mean score change | 95% interval | Improved / worsened / tied questions |
|---|---:|---:|---:|
| BM25-8 first − Dense-8 first | +0.0420 | [+0.0193, +0.0656] | 381 / 284 / 1,321 |
| BM25-8 conditional − BM25-8 first | +0.0667 | [+0.0503, +0.0848] | 279 / 46 / 1,661 |
| Dense-8 conditional − Dense-8 first | +0.0826 | [+0.0659, +0.1040] | 344 / 68 / 1,574 |
| BM25-8 conditional − Dense-8 conditional | +0.0261 | [+0.0153, +0.0377] | 286 / 229 / 1,471 |

The first-pass comparison isolates initial retrieval under the same reader
and evidence-size rule. After the conditional second pass, both retrieval
strategies improve their aggregate score, while their correct-refusal counts
fall by 46 and 68. The first-pass answer was checked before being saved, so
these changes measure which checked output is selected. They do not separate
expanded evidence from the correction prompt. The second-stage expansion is
the same algorithm for both selectors, although the first reader can request
different searches.

The earlier CARE versus raw-turn SimRetrieval difference of +0.1187 is a
**system-level** comparison: that control used a different prompt, top-18
selection and evidence serialization. The matched-reader difference of
+0.0261 is the appropriate estimate for the initial selector under the
current reader and repair policy. The model-token table excludes MiniLM
embedding, index construction, model loading and server startup; it cannot
establish a complete end-to-end latency ranking.

This is an in-dataset ablation. CARE was developed using these ten public
histories, so the cluster intervals do not imply independent generalization.
The LoCoMo aggregate mixes answer token F1 and accepted-refusal accuracy;
neither source-ID recall nor a visible citation proves semantic entailment.
