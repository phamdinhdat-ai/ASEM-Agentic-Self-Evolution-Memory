# Static-bank evaluation — tag `thg_v2_postfix`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/thg_v2_postfix`
- config: `configs/models/deepseek_v41_flash_thg_v1.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 1 (locomo_0000)
- QA pairs: 20 | elapsed: 115.2s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 20 | 0.3500 | 0.5543 | 0.5209 | 0.8925 | 0.6500 | 0 |


**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 8 | 0.1250 | 0.4265 | 0.3432 | 0.8625 | 0.3750 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 2 | 0.0000 | 0.1829 | 0.1829 | 0.8228 | 0.5000 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 10 | 0.6000 | 0.7308 | 0.7308 | 0.9304 | 0.9000 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 20 | 0.3500 | 0.5543 | 0.5209 | 0.8925 | 0.6500 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3500 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5543 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5209 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8925 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.6500 |
