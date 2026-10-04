# LongMemEval evaluation — tag `gb_full_legacy`, model `chib_deepseek_v4_1_flash`

- Config: `configs/models/deepseek_v41_flash.yaml`
- Questions: 500
- Metrics: em, f1, rougeL, bertscore_f1, judge

> Recall columns are `n/a` for systems that do not emit retrieval telemetry (only `ASEM-THG` does); `n/a` means *not measured*, not 0.

## Overall

| System | n | em | f1 | rougeL | bertscore_f1 | judge | sess_recall | sess_hit | turn_recall | abst_acc |
|---|---|---|---|---|---|---|---|---|---|---|
| ASEM-THG | 500 | 0.2680 | 0.4232 | 0.4131 | 0.8860 | 0.6140 | n/a | n/a | n/a | 0.8920 |

## Per question type

### ASEM-THG

| question_type | n | em | f1 | rougeL | bertscore_f1 | judge | sess_recall | turn_recall |
|---|---|---|---|---|---|---|---|---|
| knowledge-update | 78 | 0.3205 | 0.5046 | 0.4945 | 0.8947 | 0.7821 | n/a | n/a |
| multi-session | 133 | 0.1955 | 0.2966 | 0.2872 | 0.8722 | 0.4887 | n/a | n/a |
| single-session-assistant | 56 | 0.4643 | 0.6468 | 0.6277 | 0.9094 | 0.7500 | n/a | n/a |
| single-session-preference | 30 | 0.0000 | 0.1424 | 0.1087 | 0.8515 | 0.5333 | n/a | n/a |
| single-session-user | 70 | 0.6143 | 0.7750 | 0.7750 | 0.9462 | 0.9286 | n/a | n/a |
| temporal-reasoning | 133 | 0.1053 | 0.2862 | 0.2792 | 0.8610 | 0.4361 | n/a | n/a |

## Abstention

| System | abstention questions | refusal rate | abstention accuracy |
|---|---|---|---|
| ASEM-THG | 30 | 0.1320 | 0.8920 |
