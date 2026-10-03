# LongMemEval evaluation — tag `snap_test`, model `chib_deepseek_v4_1_flash`

- Config: `configs/models/deepseek_v41_flash.yaml`
- Questions: 3
- Metrics: em, f1

> Recall columns are `n/a` for systems that do not emit retrieval telemetry (only `ASEM-THG` does); `n/a` means *not measured*, not 0.

## Overall

| System | n | em | f1 | sess_recall | sess_hit | turn_recall | abst_acc |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 3 | 0.3333 | 0.6825 | n/a | n/a | n/a | 1.0000 |

## Per question type

### ASEM-THG

| question_type | n | em | f1 | sess_recall | turn_recall |
|---|---|---|---|---|---|
| temporal-reasoning | 3 | 0.3333 | 0.6825 | n/a | n/a |

## Abstention

| System | abstention questions | refusal rate | abstention accuracy |
|---|---|---|---|
| ASEM-THG | 0 | 0.0000 | 1.0000 |
