# CARE v3: LongMemEval-S (500 questions, full haystacks)

Diagnostic token-F1 metric, not the official judged accuracy. H-main is the only confirmatory test.

**H-main** bm25_hyb - bm25_lex = +0.0031, 95% CI [-0.0039, +0.0106] (5 better / 2 worse): **not confirmed**.

| Arm | Score | Answerable | Correct refusals | Refused ans. | Calls | Tokens/q | Evidence recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| no_memory | 0.1076 | 0.1144 | 0/30 | 0 | 500 | 85 | -- |
| full_context_8k | 0.1520 | 0.1617 | 0/30 | 5 | 500 | 7726 | -- |
| rag_dense_top18 | 0.4086 | 0.3708 | 30/30 | 122 | 500 | 2129 | 0.948 |
| bm25_first | 0.3641 | 0.3320 | 26/30 | 217 | 500 | 2056 | 0.753 |
| bm25_lex | 0.3816 | 0.3527 | 25/30 | 185 | 743 | 3326 | 0.689 |
| bm25_hyb | 0.3847 | 0.3560 | 25/30 | 185 | 743 | 3333 | 0.693 |
| dense_first | 0.3804 | 0.3557 | 23/30 | 186 | 500 | 1298 | 0.870 |
| dense_lex | 0.3953 | 0.3738 | 22/30 | 168 | 709 | 2392 | 0.736 |
| dense_hyb | 0.3941 | 0.3724 | 22/30 | 173 | 709 | 2399 | 0.733 |
| dense_graph | 0.3948 | 0.3732 | 22/30 | 169 | 709 | 2398 | 0.732 |

Exploratory contrasts (95% paired bootstrap, not used for claims):

- dense_first_vs_bm25_first: +0.0163 [-0.0092, +0.0416] (44 better / 38 worse)
- dense_hyb_vs_bm25_lex: +0.0125 [-0.0116, +0.0368] (40 better / 37 worse)
- bm25_lex_vs_rag: -0.0270 [-0.0590, +0.0047] (86 better / 136 worse)
- bm25_hyb_vs_rag: -0.0239 [-0.0557, +0.0084] (88 better / 133 worse)
- dense_hyb_vs_rag: -0.0145 [-0.0466, +0.0172] (89 better / 129 worse)
- bm25_lex_vs_full_context_8k: +0.2296 [+0.1862, +0.2726] (183 better / 92 worse)
- dense_graph_vs_dense_hyb: +0.0007 [-0.0040, +0.0060] (1 better / 1 worse)
