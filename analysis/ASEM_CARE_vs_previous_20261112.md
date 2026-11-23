# CARE Bridge v2 versus previous LoCoMo methods

Paired comparison on exact question IDs, same dataset SHA and same pinned Qwen3-8B answer-model revision. All old predictions were rescored with the CARE LoCoMo category-specific scorer; original predictions were not edited.
Only completed old graph shards and the interrupted paired BM25 subset are included.

| Previous method | Group | n | Old score | CARE score | Delta | 95% cluster bootstrap | W/L/T |
|---|---|---:|---:|---:|---:|---|---|
| hybrid_attributed | all | 746 | 0.3467 | 0.5188 | +0.1721 | [+0.1295, +0.2077] | 332/118/296 |
| hybrid_attributed | answerable | 590 | 0.2079 | 0.4679 | +0.2600 | [+0.2160, +0.2990] | 322/83/185 |
| hybrid_attributed | adversarial | 156 | 0.8718 | 0.7115 | -0.1603 | [-0.2139, -0.1073] | 10/35/111 |
| graph_memory | all | 746 | 0.3786 | 0.5188 | +0.1402 | [+0.0970, +0.1826] | 314/164/268 |
| graph_memory | answerable | 590 | 0.2635 | 0.4679 | +0.2044 | [+0.1525, +0.2498] | 296/130/164 |
| graph_memory | adversarial | 156 | 0.8141 | 0.7115 | -0.1026 | [-0.1571, -0.0441] | 18/34/104 |
| bm25 | all | 1128 | 0.1372 | 0.4912 | +0.3540 | [+0.3103, +0.3987] | 680/86/362 |
| bm25 | answerable | 954 | 0.1622 | 0.4477 | +0.2855 | [+0.2450, +0.3175] | 553/86/315 |
| bm25 | adversarial | 174 | 0.0000 | 0.7299 | +0.7299 | [+0.6759, +0.8247] | 127/0/47 |

## Per-category LoCoMo score

| Previous method | Category | n | Old | CARE | Delta |
|---|---:|---:|---:|---:|---:|
| hybrid_attributed | 1 | 102 | 0.1446 | 0.2418 | +0.0972 |
| hybrid_attributed | 2 | 121 | 0.2597 | 0.4615 | +0.2018 |
| hybrid_attributed | 3 | 33 | 0.0351 | 0.1815 | +0.1464 |
| hybrid_attributed | 4 | 334 | 0.2255 | 0.5675 | +0.3420 |
| hybrid_attributed | 5 | 156 | 0.8718 | 0.7115 | -0.1603 |
| graph_memory | 1 | 102 | 0.2616 | 0.2418 | -0.0198 |
| graph_memory | 2 | 121 | 0.2900 | 0.4615 | +0.1715 |
| graph_memory | 3 | 33 | 0.0383 | 0.1815 | +0.1432 |
| graph_memory | 4 | 334 | 0.2767 | 0.5675 | +0.2908 |
| graph_memory | 5 | 156 | 0.8141 | 0.7115 | -0.1026 |
| bm25 | 1 | 200 | 0.0584 | 0.2561 | +0.1977 |
| bm25 | 2 | 232 | 0.1155 | 0.4594 | +0.3439 |
| bm25 | 3 | 71 | 0.0126 | 0.1720 | +0.1594 |
| bm25 | 4 | 451 | 0.2558 | 0.5700 | +0.3142 |
| bm25 | 5 | 174 | 0.0000 | 0.7299 | +0.7299 |

## Limits

- FullContext in the earlier Qwen3-8B CEG run exceeded the 4,096-token evidence allowance and did not produce usable answers. Its zero score is a runtime-budget failure, not a quality comparison.
- The earlier Qwen3-4B FullContext/SimRetrieval runs use a different backbone and protocol, so their reported scores cannot isolate a CARE gain.
- These are public development questions. Old graph and BM25 completion was interrupted; the overlapping cohorts are not an independent full benchmark.
- The comparison measures whole pipelines. It does not isolate the contribution of CARE memory, retrieval, answer prompt or refusal phrase.

Machine-readable rows: `build/care/care_previous_comparison.json`.
