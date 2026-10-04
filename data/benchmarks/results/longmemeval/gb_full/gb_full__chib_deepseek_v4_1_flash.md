# LongMemEval evaluation — tag `gb_full`, model `chib_deepseek_v4_1_flash`

- Config: `configs/models/deepseek_v41_flash_thg_v1.yaml`
- Questions: 500
- Metrics: em, f1, rougeL, bertscore_f1, judge

> Recall columns are `n/a` for systems that do not emit retrieval telemetry (only `ASEM-THG` does); `n/a` means *not measured*, not 0.

## Overall

| System | n | em | f1 | rougeL | bertscore_f1 | judge | sess_recall | sess_hit | turn_recall | abst_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| ASEM-THG | 500 | 0.2740 | 0.4166 | 0.4092 | 0.8833 | 0.5880 | n/a | n/a | n/a | 0.8840 |

## Per question type

### ASEM-THG

| question_type | n | em | f1 | rougeL | bertscore_f1 | judge | sess_recall | turn_recall |
|---|---|---|---|---|---|---|---|---|
| knowledge-update | 78 | 0.3333 | 0.4886 | 0.4790 | 0.8996 | 0.7051 | n/a | n/a |
| multi-session | 133 | 0.1729 | 0.2913 | 0.2840 | 0.8679 | 0.4586 | n/a | n/a |
| single-session-assistant | 56 | 0.4107 | 0.5224 | 0.5136 | 0.8850 | 0.6071 | n/a | n/a |
| single-session-preference | 30 | 0.0000 | 0.1297 | 0.1082 | 0.8436 | 0.4000 | n/a | n/a |
| single-session-user | 70 | 0.6714 | 0.8184 | 0.8162 | 0.9505 | 0.9571 | n/a | n/a |
| temporal-reasoning | 133 | 0.1353 | 0.3083 | 0.3032 | 0.8622 | 0.4887 | n/a | n/a |

## Abstention

| System | abstention questions | refusal rate | abstention accuracy |
|---|---|---|---|
| ASEM-THG | 30 | 0.1360 | 0.8840 |
