# Static-bank evaluation — tag `thg_v2_postfix`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/thg_v2_postfix`
- config: `configs/models/deepseek_v41_flash.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 1 (locomo_0000)
- QA pairs: 199 | elapsed: 1166.9s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 199 | 0.3065 | 0.4935 | 0.4836 | 0.8685 | 0.6533 | 0 |


**adversarial**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 47 | 0.5957 | 0.6278 | 0.6278 | 0.7974 | 0.6596 |

**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 32 | 0.0625 | 0.2445 | 0.2336 | 0.8470 | 0.3438 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 13 | 0.0000 | 0.1783 | 0.1783 | 0.8544 | 0.6154 |

**single_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 70 | 0.2429 | 0.4877 | 0.4645 | 0.8989 | 0.7000 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 37 | 0.3784 | 0.6598 | 0.6598 | 0.9249 | 0.8378 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.3065 | 0.4935 | 0.4836 | 0.8685 | 0.6533 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3065 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4935 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4836 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8685 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.6533 |
