# LongMemEval evaluation — tag `smoke`, model `deepseek_v4_flash`

- Config: `configs/models/deepseek_api.yaml`
- Questions: 5
- Metrics: em, f1, rougeL

## Overall

| System | n | em | f1 | rougeL | sess_recall | sess_hit | turn_recall | abst_acc |
|---|---|---|---|---|---|---|---|---|
| NoMemory | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FullContext | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SimRetrieval | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Per question type

### NoMemory

| question_type | n | em | f1 | rougeL | sess_recall | turn_recall |
|---|---|---|---|---|---|---|
| temporal-reasoning | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### FullContext

| question_type | n | em | f1 | rougeL | sess_recall | turn_recall |
|---|---|---|---|---|---|---|
| temporal-reasoning | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Abstention

| System | abstention questions | refusal rate | abstention accuracy |
|---|---|---|---|
| NoMemory | 0 | 1.0000 | 0.0000 |
| FullContext | 0 | 1.0000 | 0.0000 |
| SimRetrieval | 0 | 0.0000 | 0.0000 |
