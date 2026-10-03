# LongMemEval evaluation — tag `thg_w1fix`, model `chib_deepseek_v4_1_flash`

- Config: `configs/models/deepseek_v41_flash.yaml`
- Questions: 60
- Metrics: em, f1, judge

> Recall columns are `n/a` for systems that do not emit retrieval telemetry (only `ASEM-THG` does); `n/a` means *not measured*, not 0.

## Overall

| System | n | em | f1 | judge | sess_recall | sess_hit | turn_recall | abst_acc |
|---|---|---|---|---|---|---|---|---|
| ASEM-THG | 60 | 0.1667 | 0.3577 | 0.6167 | n/a | n/a | n/a | 0.9500 |

## Per question type

### ASEM-THG

| question_type | n | em | f1 | judge | sess_recall | turn_recall |
|---|---|---|---|---|---|---|
| temporal-reasoning | 60 | 0.1667 | 0.3577 | 0.6167 | n/a | n/a |

## Abstention

| System | abstention questions | refusal rate | abstention accuracy |
|---|---|---|---|
| ASEM-THG | 6 | 0.0833 | 0.9500 |
