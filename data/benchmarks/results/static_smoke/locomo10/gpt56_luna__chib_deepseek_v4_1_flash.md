# Static-bank evaluation — tag `gpt56_luna`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/gpt56_luna`
- config: `configs/models/deepseek_v41_flash_4k.yaml`
- judge: `configs/models/deepseek_v41_flash_4k.yaml`
- metrics: `em, f1, rougeL`
- conversations: 1 (locomo_0000)
- QA pairs: 12 | elapsed: 115.8s

| System | n | EM | F1 | ROUGE-L | Errors |
|---|---|---|---|---|---|
| ASEM-THG | 12 | 0.2500 | 0.4694 | 0.4694 | 0 |


**multi_hop**
| System | n | EM | F1 | ROUGE-L |
|---|---|---|---|---|
| ASEM-THG | 4 | 0.0000 | 0.3000 | 0.3000 |

**open_domain**
| System | n | EM | F1 | ROUGE-L |
|---|---|---|---|---|
| ASEM-THG | 1 | 0.0000 | 0.1000 | 0.1000 |

**temporal**
| System | n | EM | F1 | ROUGE-L |
|---|---|---|---|---|
| ASEM-THG | 7 | 0.4286 | 0.6190 | 0.6190 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L |
|---|---|---|---|---|
| locomo_0000 | 12 | 0.2500 | 0.4694 | 0.4694 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.2500 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4694 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4694 |
