# Static-bank evaluation — tag `thg_v2_postfix`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/thg_v2_postfix`
- config: `configs/models/deepseek_v41_flash.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 1 (locomo_0000)
- QA pairs: 20 | elapsed: 113.8s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 20 | 0.3500 | 0.5093 | 0.4759 | 0.8937 | 0.7000 | 0 |


**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 8 | 0.1250 | 0.3247 | 0.2414 | 0.8630 | 0.5000 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 2 | 0.0000 | 0.1829 | 0.1829 | 0.8278 | 0.5000 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 10 | 0.6000 | 0.7222 | 0.7222 | 0.9314 | 0.9000 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 20 | 0.3500 | 0.5093 | 0.4759 | 0.8937 | 0.7000 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3500 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5093 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4759 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8937 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.7000 |
