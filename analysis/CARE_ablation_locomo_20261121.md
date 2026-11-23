# CARE component ablation: locomo (1986 questions)

LoCoMo is development data.

Reproducibility: `care` arm identical to saved CARE answers on 1985/1986 (FAIL). Physical model calls: 6545.

| Arm | Score | Answerable | Correct refusals | Refused ans. | Repairs | Calls (alone) | Tokens/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_pass | 0.4588 | 0.3489 | 374/446 | 629 | 0 | 1986 | 714 |
| care | 0.5256 | 0.4648 | 328/446 | 257 | 1003 | 2989 | 1835 |
| no_cite_req | 0.5143 | 0.4678 | 301/446 | 245 | 998 | 2984 | 1829 |
| no_binding | 0.4276 | 0.4722 | 122/446 | 226 | 872 | 2858 | 1699 |
| no_event | 0.5246 | 0.4648 | 326/446 | 257 | 1002 | 2988 | 1834 |
| no_checks | 0.4266 | 0.4722 | 120/446 | 226 | 871 | 2857 | 1697 |
| gate_always | 0.5229 | 0.4484 | 348/446 | 329 | 1986 | 3972 | 2854 |
| repair_expand_only | 0.5289 | 0.4723 | 323/446 | 233 | 1003 | 2989 | 1826 |
| repair_retry_only | 0.4587 | 0.3499 | 372/446 | 622 | 1003 | 2989 | 1083 |
| repair_no_search | 0.5262 | 0.4650 | 329/446 | 258 | 1003 | 2989 | 1762 |

Contrast with `care` (paired bootstrap; Bonferroni 99.375% over 8 arms; margin +/-0.01):

- **no_cite_req**: -0.0112, 95% [-0.0168, -0.0061], adjusted [-0.0186, -0.0043] (7 better / 28 worse, 0/10 clusters better): contributes (removing it lowers the score)
- **no_binding**: -0.0980, 95% [-0.1127, -0.0825], adjusted [-0.1189, -0.0761] (18 better / 211 worse, 0/10 clusters better): contributes (removing it lowers the score)
- **no_event**: -0.0010, 95% [-0.0024, +0.0000], adjusted [-0.0029, +0.0000] (0 better / 2 worse, 0/10 clusters better): not needed (within +/-0.01)
- **no_checks**: -0.0990, 95% [-0.1144, -0.0829], adjusted [-0.1208, -0.0764] (18 better / 213 worse, 0/10 clusters better): contributes (removing it lowers the score)
- **gate_always**: -0.0026, 95% [-0.0134, +0.0070], adjusted [-0.0181, +0.0101] (203 better / 206 worse, 5/10 clusters better): inconclusive
- **repair_expand_only**: +0.0033, 95% [-0.0032, +0.0089], adjusted [-0.0063, +0.0107] (45 better / 24 worse, 7/10 clusters better): inconclusive
- **repair_retry_only**: -0.0669, 95% [-0.0862, -0.0494], adjusted [-0.0926, -0.0445] (47 better / 281 worse, 0/10 clusters better): contributes (removing it lowers the score)
- **repair_no_search**: +0.0007, 95% [-0.0038, +0.0049], adjusted [-0.0058, +0.0063] (31 better / 28 worse, 5/10 clusters better): not needed (within +/-0.01)

Repair decomposition (share of the first-pass -> care gain):

- repair_retry_only: 0.4587 (share -0.00)
- repair_expand_only: 0.5289 (share 1.05)
- care: 0.5256 (share 1.00)
