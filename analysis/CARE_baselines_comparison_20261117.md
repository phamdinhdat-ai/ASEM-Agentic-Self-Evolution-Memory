# End-to-end comparison on all 1,986 LoCoMo questions (17/11/2026)

Same reader (Qwen3-8B, pinned, 8K window, non-thinking), same CARE prompt, checks and repair gate.
New run: matched reader with frozen Qwen3-Embedding-0.6B top-8 first bundle, kernels `asem-care-mr-qwen3emb-shard-{0..4}`
(output `kaggle_outputs/mr_qwen3emb/`, rankings `build/care/qwen3_0p6b_dense_rankings.json`, SHA 29073f9e…).
Script: `python -m scripts.compare_care_baselines` (reproduces every previously published number exactly). JSON: `CARE_baselines_comparison_20261117.json`.

| System | Aggregate | Answerable F1 | Correct refusals | Refused answerable | Calls |
|---|---|---|---|---|---|
| NoMemory | 0.0774 | 0.0998 | 0/446 | 2/1540 | 1986 |
| FullContext-8K | 0.1961 | 0.2406 | 19/446 | 40/1540 | 1986 |
| SimRetrieval (MiniLM, top-18) | 0.4069 | 0.3124 | 327/446 | 387/1540 | 1986 |
| MiniLM-8 first | 0.4168 | 0.2908 | 380/446 | 691/1540 | 1986 |
| MiniLM-8 repair | 0.4994 | 0.4415 | 312/446 | 245/1540 | 3056 |
| BM25-8 first | 0.4588 | 0.3489 | 374/446 | 630/1540 | 1986 |
| **BM25-8 repair (CARE)** | **0.5256** | **0.4648** | **328/446** | 257/1540 | 2990 |
| Qwen3-Emb-8 first | 0.4730 | 0.3775 | 358/446 | 469/1540 | 1986 |
| Qwen3-Emb-8 repair | 0.5169 | 0.4640 | 312/446 | **219/1540** | **2813** |

Paired differences (conversation-cluster bootstrap, 10,000 draws; conversations better out of 10):

| Contrast | Mean | 95% CI | Conv. better |
|---|---|---|---|
| Qwen3-Emb first − BM25 first | +0.0142 | [+0.0062, +0.0228] | 9/10 |
| Qwen3-Emb first − MiniLM first | +0.0562 | [+0.0405, +0.0728] | 10/10 |
| Qwen3-Emb repair − MiniLM repair | +0.0175 | [+0.0088, +0.0266] | 9/10 |
| Qwen3-Emb repair − Qwen3-Emb first | +0.0439 | [+0.0316, +0.0573] | 10/10 |
| Qwen3-Emb repair − CARE (BM25 repair) | −0.0087 | [−0.0183, +0.0013] | 3/10 |

By category (multi-hop / temporal / open-domain / single-hop / adversarial):
- CARE (BM25 repair): 0.2609 / 0.4415 / 0.1647 / 0.5763 / 0.7354
- Qwen3-Emb repair: **0.3005** / 0.4404 / **0.1813** / 0.5601 / 0.6996

Reading:
- The better encoder wins without repair: the dense first pass beats BM25 (+0.0142, 9/10 conversations) and MiniLM (+0.0562, 10/10).
- With repair, Qwen3-Emb does not beat CARE; the difference is −0.0087 and its interval includes zero.
- Answerable F1 is essentially equal (0.4640 vs 0.4648). The gap is refusals: 312 vs 328 correct, 16 questions = 0.0081 of the 0.0087.
- Repair adds less on top of a dense bundle (+0.0439) than on top of BM25 (+0.0667) or MiniLM (+0.0826). The repair expansion is lexical (unstemmed/stemmed BM25 + neighbours), so it complements a lexical first bundle more. Hypothesis, not isolated.
- Qwen3-Emb repair uses the fewest calls of the repair systems (2,813) and refuses the fewest answerable questions (219).
- Dense helps most where several pieces of evidence are needed: multi-hop +0.040, open-domain +0.017 over CARE.
