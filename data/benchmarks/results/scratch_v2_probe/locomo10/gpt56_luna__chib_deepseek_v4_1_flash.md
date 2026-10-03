# Static-bank evaluation — tag `gpt56_luna`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/gpt56_luna`
- config: `configs/models/deepseek_v41_flash_4k_v2.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 1 (locomo_0000)
- QA pairs: 199 | elapsed: 1493.4s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 199 | 0.3467 | 0.5318 | 0.5172 | 0.9029 | 0.6834 | 0 |


**adversarial**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 47 | 0.8085 | 0.8156 | 0.8156 | 0.9458 | 0.8085 |

**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 32 | 0.0938 | 0.3363 | 0.2836 | 0.8581 | 0.3438 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 13 | 0.0000 | 0.2017 | 0.2017 | 0.8565 | 0.5385 |

**single_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 70 | 0.2000 | 0.4293 | 0.4119 | 0.8906 | 0.7143 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 37 | 0.3784 | 0.6503 | 0.6503 | 0.9268 | 0.8108 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.3467 | 0.5318 | 0.5172 | 0.9029 | 0.6834 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3467 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5318 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5172 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.9029 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.6834 |
