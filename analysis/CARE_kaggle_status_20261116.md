# CARE Kaggle status and result comparison (16 November 2026)

Checked with `kaggle kernels list/status/output` (CLI 2.2.5, account `phamdinhdatai`). Downloaded outputs: `build/care/kaggle_check_20261116/`.

## 1. Kaggle run status (35 CARE kernels)

| Group | Kernels | Status |
|---|---|---|
| LoCoMo relation-full (paper's CARE), shards 0-4 | 5 | COMPLETE |
| LoCoMo context baselines (NoMemory, FullContext-8K), shards 0-4 + pilot | 6 | COMPLETE |
| LoCoMo SimRetrieval, shards 0-4 + pilot + BERTScore | 7 | COMPLETE |
| BERTScore rescoring (previous, context, full-LoCoMo) | 3 | COMPLETE |
| Matched-reader (BM25-8 vs Dense-8), shards 0-4 + pilot | 6 | COMPLETE |
| Development pilots (bridge, source-first, temporal-relations) | 3 | COMPLETE |
| LongMemEval cleaned-oracle: pilot, shards 0, 1, 2 | 4 | COMPLETE |
| **Full-history 40K pilot** | 1 | **ERROR** |

Nothing is running or queued on Kaggle.

### Differences from `CARE_followup_status_20261114.md`
- **40K pilot did not stay queued; it ran and failed** (~10 min). `vllm.log`: *"The model's max seq len (40960) is larger than the maximum number of tokens that can be stored in KV cache (30432)"* at `gpu_memory_utilization=0.88`, tensor-parallel 2 on T4 (14.56 GiB each; weights 7.64 GiB, 2.09 GiB left for KV cache). It is a configuration error, not a quota problem. Fixes: lower `max_model_len` to <=30,432 (too small for the 36.2K target), raise utilization, or quantize/offload KV.
- **The scheduler chain is dead.** The PC rebooted 15 Nov 03:55 and no Python process survives. Status files froze at 14 Nov 20:53 UTC: LongMemEval shards **3 and 4 were never submitted** (shard 2 finished afterwards, unrecorded), and the Qwen3-4B chain never started (`NOT_SUBMITTED`).

## 2. CARE on LoCoMo: paper vs Kaggle (1,986 questions, Qwen3-8B)

Every number below checks against `paper_asem/asem_v2.tex` (0.5256, 0.4994, 0.4588, 0.4168, 0.4069, 0.1961, 0.0774, 2,990 calls, 1,004 repairs, 0.7033, 0.0261, 0.0153 all present). No discrepancy found.

| System | LoCoMo score | Calls | Tokens/question |
|---|---:|---:|---:|
| NoMemory | 0.0774 | 1,986 | 77.7 |
| FullContext-8K | 0.1961 | 1,986 | 7,887.9 |
| SimRetrieval (top-18, MiniLM) | 0.4069 | 1,986 | 1,123.9 |
| **CARE** (BM25-8 + conditional repair) | **0.5256** | 2,990 | 1,836.4 |

Matched reader (same reader, same repair policy):

| Selector / policy | Aggregate | Answerable F1 | Correct refusals /446 | Source recall |
|---|---:|---:|---:|---:|
| BM25-8 first pass | 0.4588 | 0.3489 | 374 | 0.5353 |
| Dense-8 first pass | 0.4168 | 0.2908 | 380 | 0.4576 |
| BM25-8 conditional | **0.5256** | **0.4648** | 328 | **0.7033** |
| Dense-8 conditional | 0.4994 | 0.4415 | 312 | 0.6919 |

- BM25 vs Dense after repair: +0.0261, 95% cluster CI [+0.0153, +0.0377].
- Repair vs first pass (BM25): +0.0667 [+0.0503, +0.0848], but 46 questions worsened (correct refusals fall 374 -> 328).
- Caveat in the paper: CARE was developed on these 10 histories, so intervals are not independent generalization.

## 3. New since the paper: LongMemEval cleaned-oracle (308 of 500 questions)

Not in the paper (it lists LongMemEval as future work). Shards 0, 1, 2 = 98 + 110 + 100 questions; shards 3 and 4 (97 + 95) missing. Dataset SHA-256 `821a2034...20e38620c` matches across shards. Diagnostic metrics only, not official LLM-judged LongMemEval accuracy; oracle setting supplies only annotated relevant sessions.

| Metric | First pass | CARE conditional | Paired change (95% question-bootstrap) |
|---|---:|---:|---|
| Answerable F1 (n=294) | 0.3728 | 0.4059 | **+0.0331 [+0.0151, +0.0540]**, 13 better / 0 worse |
| Answerable EM | 0.2857 | 0.3129 | -- |
| Diagnostic score, all 308 | 0.3948 | 0.4135 | +0.0187 [-0.0037, +0.0424] (CI includes 0) |
| Abstention correct (`_abs`, n=14) | 12 | 8 | fewer correct refusals |
| Source-session recall | 0.969 | 0.927 | **-0.0416 [-0.0601, -0.0242]**, 5 better / 32 worse |
| Model calls | 308 | 440 (1.43/question) | 42.9% escalated |

By question type (answerable F1, first -> conditional): single-session-user 0.846 -> 0.888; single-session-assistant 0.663 -> 0.763; knowledge-update 0.526 -> 0.548; temporal-reasoning 0.233 -> 0.270; multi-session 0.083 -> 0.090; single-session-preference 0.005 -> 0.009.

Reading: repair helps answerable F1 here too, matching the LoCoMo direction, and again costs correct refusals (12 -> 8 of 14, small n). Multi-session and preference questions stay near zero. Recall drops after repair in the oracle setting because first-pass recall is already ~0.97 and the rebuilt bundle can displace annotated sessions.

## 4. Not yet run
- LongMemEval shards 3 and 4 (192 questions) -> needed for the preregistered 500-question merge.
- Full-history 40K pilot + 5 shards (needs config fix).
- Qwen3-4B second-backbone pilot + 50-question cohort.
- Claim-level source-support audit (100-answer queue frozen, no labels).
