# Static-bank evaluation — tag `thg_v2_postfix`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/thg_v2_postfix`
- config: `configs/models/deepseek_v41_flash_thg_v1.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 1 (locomo_0000)
- QA pairs: 199 | elapsed: 1027.5s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 199 | 0.3065 | 0.4851 | 0.4717 | 0.8702 | 0.6181 | 0 |


**adversarial**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 47 | 0.5532 | 0.5731 | 0.5731 | 0.7961 | 0.5532 |

**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 32 | 0.0625 | 0.2585 | 0.2235 | 0.8463 | 0.2500 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 13 | 0.0000 | 0.1824 | 0.1824 | 0.8534 | 0.4615 |

**single_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 70 | 0.2286 | 0.4813 | 0.4591 | 0.9016 | 0.7286 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 37 | 0.4595 | 0.6830 | 0.6830 | 0.9316 | 0.8649 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.3065 | 0.4851 | 0.4717 | 0.8702 | 0.6181 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3065 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4851 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4717 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8702 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.6181 |
