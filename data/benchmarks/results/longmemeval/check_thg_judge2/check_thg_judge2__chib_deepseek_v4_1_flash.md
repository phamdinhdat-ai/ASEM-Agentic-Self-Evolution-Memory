# LongMemEval evaluation — tag `check_thg_judge2`, model `chib_deepseek_v4_1_flash`

- Config: `configs/models/deepseek_v41_flash.yaml`
- Questions: 10
- Metrics: em, f1, rougeL, judge

> Recall columns are `n/a` for systems that do not emit retrieval telemetry (only `ASEM-THG` does); `n/a` means *not measured*, not 0.

## Overall

| System | n | em | f1 | rougeL | judge | sess_recall | sess_hit | turn_recall | abst_acc |
|---|---|---|---|---|---|---|---|---|---|
| ASEM-THG | 10 | 0.0000 | 0.2335 | 0.2335 | 0.4000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 |

## Per question type

### ASEM-THG

| question_type | n | em | f1 | rougeL | judge | sess_recall | turn_recall |
|---|---|---|---|---|---|---|---|
| temporal-reasoning | 10 | 0.0000 | 0.2335 | 0.2335 | 0.4000 | 1.0000 | 0.0000 |

## Abstention

| System | abstention questions | refusal rate | abstention accuracy |
|---|---|---|---|
| ASEM-THG | 0 | 0.0000 | 1.0000 |
