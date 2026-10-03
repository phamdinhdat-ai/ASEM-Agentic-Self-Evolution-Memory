# Static-bank evaluation — tag `gpt56_luna`, model `chib_deepseek_v4_1_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/gpt56_luna`
- config: `configs/models/deepseek_v41_flash_4k.yaml`
- judge: `configs/models/judge_v41_flash.yaml`
- metrics: `bertscore_f1, judge`
- conversations: 10 (locomo_0000, locomo_0001, locomo_0002, locomo_0003, locomo_0004, locomo_0005, locomo_0006, locomo_0007, locomo_0008, locomo_0009)
- QA pairs: 1986 | elapsed: 2680.7s

| System | n | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|
| ASEM-THG | 1986 | 0.8564 | 0.6103 | 0 |


**adversarial**
| System | n | BERTScore-F1 | Judge |
|---|---|---|---|
| ASEM-THG | 446 | 0.7982 | 0.6839 |

**multi_hop**
| System | n | BERTScore-F1 | Judge |
|---|---|---|---|
| ASEM-THG | 282 | 0.8486 | 0.3050 |

**open_domain**
| System | n | BERTScore-F1 | Judge |
|---|---|---|---|
| ASEM-THG | 96 | 0.8216 | 0.3438 |

**single_hop**
| System | n | BERTScore-F1 | Judge |
|---|---|---|---|
| ASEM-THG | 841 | 0.8809 | 0.6516 |

**temporal**
| System | n | BERTScore-F1 | Judge |
|---|---|---|---|
| ASEM-THG | 321 | 0.8906 | 0.7477 |

## Per-conversation

**ASEM-THG**
| Conversation | n | BERTScore-F1 | Judge |
|---|---|---|---|
| locomo_0000 | 199 | 0.8683 | 0.6633 |
| locomo_0001 | 105 | 0.8577 | 0.6476 |
| locomo_0002 | 193 | 0.8634 | 0.6166 |
| locomo_0003 | 260 | 0.8461 | 0.6077 |
| locomo_0004 | 242 | 0.8571 | 0.5579 |
| locomo_0005 | 158 | 0.8474 | 0.5823 |
| locomo_0006 | 190 | 0.8518 | 0.5579 |
| locomo_0007 | 239 | 0.8522 | 0.6444 |
| locomo_0008 | 196 | 0.8600 | 0.5765 |
| locomo_0009 | 204 | 0.8629 | 0.6618 |

## Per-conversation matrices

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8683 |
| locomo_0001 | 0.8577 |
| locomo_0002 | 0.8634 |
| locomo_0003 | 0.8461 |
| locomo_0004 | 0.8571 |
| locomo_0005 | 0.8474 |
| locomo_0006 | 0.8518 |
| locomo_0007 | 0.8522 |
| locomo_0008 | 0.8600 |
| locomo_0009 | 0.8629 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.6633 |
| locomo_0001 | 0.6476 |
| locomo_0002 | 0.6166 |
| locomo_0003 | 0.6077 |
| locomo_0004 | 0.5579 |
| locomo_0005 | 0.5823 |
| locomo_0006 | 0.5579 |
| locomo_0007 | 0.6444 |
| locomo_0008 | 0.5765 |
| locomo_0009 | 0.6618 |
