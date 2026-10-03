# Static-bank evaluation — tag `gpt56_luna`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/gpt56_luna`
- config: `configs/models/deepseek_v41_flash_4k_v2.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 10 (locomo_0000, locomo_0001, locomo_0002, locomo_0003, locomo_0004, locomo_0005, locomo_0006, locomo_0007, locomo_0008, locomo_0009)
- QA pairs: 1986 | elapsed: 21468.5s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 1986 | 0.3364 | 0.5186 | 0.5032 | 0.8968 | 0.6334 | 1 |


**adversarial**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 446 | 0.8677 | 0.8688 | 0.8688 | 0.9534 | 0.8655 |

**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 282 | 0.0461 | 0.2542 | 0.2294 | 0.8584 | 0.3014 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 96 | 0.0312 | 0.1370 | 0.1307 | 0.8411 | 0.2917 |

**single_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 841 | 0.2176 | 0.4357 | 0.4214 | 0.8868 | 0.6302 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 321 | 0.2555 | 0.5960 | 0.5616 | 0.8945 | 0.7134 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.3518 | 0.5317 | 0.5188 | 0.9033 | 0.6734 |
| locomo_0001 | 105 | 0.3905 | 0.5493 | 0.5449 | 0.8979 | 0.7238 |
| locomo_0002 | 193 | 0.3264 | 0.5475 | 0.5247 | 0.9051 | 0.6477 |
| locomo_0003 | 260 | 0.3154 | 0.4761 | 0.4640 | 0.8934 | 0.6000 |
| locomo_0004 | 242 | 0.3678 | 0.5308 | 0.5208 | 0.8999 | 0.6116 |
| locomo_0005 | 158 | 0.3291 | 0.5042 | 0.4954 | 0.8928 | 0.5759 |
| locomo_0006 | 190 | 0.3211 | 0.5400 | 0.4990 | 0.8922 | 0.5947 |
| locomo_0007 | 239 | 0.3305 | 0.5155 | 0.5092 | 0.8890 | 0.6527 |
| locomo_0008 | 196 | 0.3316 | 0.5079 | 0.4903 | 0.8967 | 0.6276 |
| locomo_0009 | 204 | 0.3235 | 0.5079 | 0.4905 | 0.8990 | 0.6667 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3518 |
| locomo_0001 | 0.3905 |
| locomo_0002 | 0.3264 |
| locomo_0003 | 0.3154 |
| locomo_0004 | 0.3678 |
| locomo_0005 | 0.3291 |
| locomo_0006 | 0.3211 |
| locomo_0007 | 0.3305 |
| locomo_0008 | 0.3316 |
| locomo_0009 | 0.3235 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5317 |
| locomo_0001 | 0.5493 |
| locomo_0002 | 0.5475 |
| locomo_0003 | 0.4761 |
| locomo_0004 | 0.5308 |
| locomo_0005 | 0.5042 |
| locomo_0006 | 0.5400 |
| locomo_0007 | 0.5155 |
| locomo_0008 | 0.5079 |
| locomo_0009 | 0.5079 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.5188 |
| locomo_0001 | 0.5449 |
| locomo_0002 | 0.5247 |
| locomo_0003 | 0.4640 |
| locomo_0004 | 0.5208 |
| locomo_0005 | 0.4954 |
| locomo_0006 | 0.4990 |
| locomo_0007 | 0.5092 |
| locomo_0008 | 0.4903 |
| locomo_0009 | 0.4905 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.9033 |
| locomo_0001 | 0.8979 |
| locomo_0002 | 0.9051 |
| locomo_0003 | 0.8934 |
| locomo_0004 | 0.8999 |
| locomo_0005 | 0.8928 |
| locomo_0006 | 0.8922 |
| locomo_0007 | 0.8890 |
| locomo_0008 | 0.8967 |
| locomo_0009 | 0.8990 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.6734 |
| locomo_0001 | 0.7238 |
| locomo_0002 | 0.6477 |
| locomo_0003 | 0.6000 |
| locomo_0004 | 0.6116 |
| locomo_0005 | 0.5759 |
| locomo_0006 | 0.5947 |
| locomo_0007 | 0.6527 |
| locomo_0008 | 0.6276 |
| locomo_0009 | 0.6667 |
