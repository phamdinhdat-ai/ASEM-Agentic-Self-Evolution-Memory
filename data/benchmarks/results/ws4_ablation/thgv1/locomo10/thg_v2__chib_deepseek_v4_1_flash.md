# Static-bank evaluation — tag `thg_v2`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/thg_v2`
- config: `configs/models/deepseek_v41_flash_thg_v1.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 1 (locomo_0000)
- QA pairs: 20 | elapsed: 101.9s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 20 | 0.3000 | 0.5762 | 0.5429 | 0.9090 | 0.7000 | 0 |


**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 8 | 0.0000 | 0.4390 | 0.3557 | 0.8836 | 0.5000 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 2 | 0.0000 | 0.2481 | 0.2481 | 0.8416 | 1.0000 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 10 | 0.6000 | 0.7516 | 0.7516 | 0.9429 | 0.8000 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 20 | 0.3000 | 0.5762 | 0.5429 | 0.9090 | 0.7000 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3000 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5762 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5429 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.9090 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.7000 |
