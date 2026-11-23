# CARE component ablation: lme (500 questions)

LongMemEval cleaned oracle; diagnostic metric; results were seen before design choices.

Reproducibility: `care` arm identical to saved CARE answers on 500/500 (PASS). Physical model calls: 1446.

| Arm | Score | Answerable | Correct refusals | Refused ans. | Repairs | Calls (alone) | Tokens/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| first_pass | 0.3689 | 0.3478 | 21/30 | 195 | 0 | 500 | 2106 |
| care | 0.3832 | 0.3736 | 16/30 | 156 | 216 | 716 | 3208 |
| no_cite_req | 0.3872 | 0.3779 | 16/30 | 146 | 196 | 696 | 3119 |
| no_binding | 0.3832 | 0.3736 | 16/30 | 156 | 216 | 716 | 3208 |
| no_event | 0.3852 | 0.3757 | 16/30 | 155 | 215 | 715 | 3203 |
| no_checks | 0.3852 | 0.3757 | 16/30 | 155 | 215 | 715 | 3203 |
| gate_always | 0.3095 | 0.2909 | 18/30 | 225 | 500 | 1000 | 4559 |
| repair_expand_only | 0.3798 | 0.3700 | 16/30 | 153 | 216 | 716 | 3201 |
| repair_retry_only | 0.3721 | 0.3511 | 21/30 | 188 | 216 | 716 | 3048 |
| repair_no_search | 0.3852 | 0.3736 | 17/30 | 156 | 216 | 716 | 3207 |

Contrast with `care` (paired bootstrap; Bonferroni 99.375% over 8 arms; margin +/-0.01):

- **no_cite_req**: +0.0040, 95% [-0.0004, +0.0104], adjusted [-0.0012, +0.0137] (7 better / 1 worse): inconclusive
- **no_binding**: +0.0000, 95% [+0.0000, +0.0000], adjusted [+0.0000, +0.0000] (0 better / 0 worse): not needed (within +/-0.01)
- **no_event**: +0.0020, 95% [+0.0000, +0.0060], adjusted [+0.0000, +0.0080] (1 better / 0 worse): not needed (within +/-0.01)
- **no_checks**: +0.0020, 95% [+0.0000, +0.0060], adjusted [+0.0000, +0.0080] (1 better / 0 worse): not needed (within +/-0.01)
- **gate_always**: -0.0737, 95% [-0.0981, -0.0508], adjusted [-0.1086, -0.0425] (12 better / 67 worse): contributes (removing it lowers the score)
- **repair_expand_only**: -0.0034, 95% [-0.0098, +0.0011], adjusted [-0.0130, +0.0018] (3 better / 3 worse): inconclusive
- **repair_retry_only**: -0.0111, 95% [-0.0262, +0.0030], adjusted [-0.0318, +0.0084] (5 better / 18 worse): inconclusive
- **repair_no_search**: +0.0020, 95% [+0.0000, +0.0060], adjusted [+0.0000, +0.0080] (1 better / 0 worse): not needed (within +/-0.01)

Repair decomposition (share of the first-pass -> care gain):

- repair_retry_only: 0.3721 (share 0.22)
- repair_expand_only: 0.3798 (share 0.76)
- care: 0.3832 (share 1.00)
