# Static-bank evaluation — tag `thg_v2`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/thg_v2`
- config: `configs/models/deepseek_v41_flash.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 1 (locomo_0000)
- QA pairs: 20 | elapsed: 111.8s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 20 | 0.3000 | 0.5675 | 0.5342 | 0.9006 | 0.6500 | 0 |


**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 8 | 0.1250 | 0.4648 | 0.3815 | 0.8694 | 0.5000 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 2 | 0.0000 | 0.1577 | 0.1577 | 0.8393 | 0.5000 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 10 | 0.5000 | 0.7316 | 0.7316 | 0.9378 | 0.8000 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 20 | 0.3000 | 0.5675 | 0.5342 | 0.9006 | 0.6500 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3000 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5675 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5342 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.9006 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.6500 |
