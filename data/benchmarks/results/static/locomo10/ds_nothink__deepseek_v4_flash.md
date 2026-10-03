# Static-bank evaluation — tag `ds_nothink`, model `deepseek_v4_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/ds_nothink`
- config: `configs/models/deepseek_openai.yaml`
- judge: `configs/models/judge_api.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 9 (locomo_0000, locomo_0001, locomo_0002, locomo_0003, locomo_0004, locomo_0005, locomo_0006, locomo_0007, locomo_0009)
- QA pairs: 1790 | elapsed: 56858.3s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM | 1790 | 0.0291 | 0.2189 | 0.2087 | 0.8335 | 0.4172 | 0 |
| FastASEM | 207 | 0.1981 | 0.3533 | 0.3448 | 0.8305 | 0.5000 | 8 |
| FullContext | 28 | 0.1429 | 0.3241 | 0.3122 | 0.6236 | 0.6000 | 8 |
| NoMemory | 28 | 0.0000 | 0.0000 | 0.0000 | 0.5608 | 0.0000 | 8 |


**adversarial**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM | 406 | 0.0197 | 0.1583 | 0.1512 | 0.8187 | 0.2178 |
| FastASEM | 47 | 0.0638 | 0.1052 | 0.1028 | 0.8108 | 0.0000 |

**commonsense**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM | 83 | 0.0000 | 0.0724 | 0.0701 | 0.8064 | 0.2410 |
| FastASEM | 13 | 0.0000 | 0.1175 | 0.1175 | 0.7946 | 0.2000 |
| FullContext | 4 | 0.0000 | 0.2381 | 0.2381 | 0.4248 | 0.5000 |
| NoMemory | 4 | 0.0000 | 0.0000 | 0.0000 | 0.4287 | 0.0000 |

**conversational**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM | 768 | 0.0443 | 0.2784 | 0.2686 | 0.8517 | 0.5833 |
| FastASEM | 72 | 0.2639 | 0.4628 | 0.4495 | 0.8674 | 0.0000 |

**single_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM | 245 | 0.0041 | 0.1760 | 0.1534 | 0.8310 | 0.2245 |
| FastASEM | 34 | 0.0588 | 0.2023 | 0.1818 | 0.7710 | 0.1667 |
| FullContext | 10 | 0.3000 | 0.5261 | 0.4927 | 0.7373 | 0.6250 |
| NoMemory | 10 | 0.0000 | 0.0000 | 0.0000 | 0.6399 | 0.0000 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM | 288 | 0.0312 | 0.2247 | 0.2172 | 0.8160 | 0.4688 |
| FastASEM | 41 | 0.4146 | 0.6456 | 0.6456 | 0.8490 | 0.7895 |
| FullContext | 14 | 0.0714 | 0.2044 | 0.2044 | 0.5991 | 0.6000 |
| NoMemory | 14 | 0.0000 | 0.0000 | 0.0000 | 0.5419 | 0.0000 |

## Per-conversation

**ASEM**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.0302 | 0.2358 | 0.2204 | 0.8424 | 0.4171 |
| locomo_0001 | 105 | 0.0381 | 0.1795 | 0.1712 | 0.8245 | 0.4381 |
| locomo_0002 | 193 | 0.0207 | 0.2047 | 0.1965 | 0.8358 | 0.3717 |
| locomo_0003 | 260 | 0.0231 | 0.2064 | 0.1992 | 0.8273 | 0.3462 |
| locomo_0004 | 242 | 0.0496 | 0.2439 | 0.2330 | 0.8437 | 0.4587 |
| locomo_0005 | 158 | 0.0063 | 0.1709 | 0.1628 | 0.8237 | 0.3671 |
| locomo_0006 | 190 | 0.0211 | 0.2383 | 0.2261 | 0.8327 | 0.4842 |
| locomo_0007 | 239 | 0.0460 | 0.2526 | 0.2420 | 0.8334 | 0.4686 |
| locomo_0009 | 204 | 0.0196 | 0.2026 | 0.1918 | 0.8317 | 0.4069 |

**FastASEM**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.2060 | 0.3675 | 0.3587 | 0.8639 | 0.5000 |
| locomo_0001 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**FullContext**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 28 | 0.1429 | 0.3241 | 0.3122 | 0.6236 | 0.6000 |

**NoMemory**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 28 | 0.0000 | 0.0000 | 0.0000 | 0.5608 | 0.0000 |

## Per-conversation matrices

**EM**
| Conversation | ASEM | FastASEM | FullContext | NoMemory |
|---|---|---|---|---|
| locomo_0000 | 0.0302 | 0.2060 | 0.1429 | 0.0000 |
| locomo_0001 | 0.0381 | 0.0000 | — | — |
| locomo_0002 | 0.0207 | — | — | — |
| locomo_0003 | 0.0231 | — | — | — |
| locomo_0004 | 0.0496 | — | — | — |
| locomo_0005 | 0.0063 | — | — | — |
| locomo_0006 | 0.0211 | — | — | — |
| locomo_0007 | 0.0460 | — | — | — |
| locomo_0009 | 0.0196 | — | — | — |

**F1**
| Conversation | ASEM | FastASEM | FullContext | NoMemory |
|---|---|---|---|---|
| locomo_0000 | 0.2358 | 0.3675 | 0.3241 | 0.0000 |
| locomo_0001 | 0.1795 | 0.0000 | — | — |
| locomo_0002 | 0.2047 | — | — | — |
| locomo_0003 | 0.2064 | — | — | — |
| locomo_0004 | 0.2439 | — | — | — |
| locomo_0005 | 0.1709 | — | — | — |
| locomo_0006 | 0.2383 | — | — | — |
| locomo_0007 | 0.2526 | — | — | — |
| locomo_0009 | 0.2026 | — | — | — |

**ROUGE-L**
| Conversation | ASEM | FastASEM | FullContext | NoMemory |
|---|---|---|---|---|
| locomo_0000 | 0.2204 | 0.3587 | 0.3122 | 0.0000 |
| locomo_0001 | 0.1712 | 0.0000 | — | — |
| locomo_0002 | 0.1965 | — | — | — |
| locomo_0003 | 0.1992 | — | — | — |
| locomo_0004 | 0.2330 | — | — | — |
| locomo_0005 | 0.1628 | — | — | — |
| locomo_0006 | 0.2261 | — | — | — |
| locomo_0007 | 0.2420 | — | — | — |
| locomo_0009 | 0.1918 | — | — | — |

**BERTScore-F1**
| Conversation | ASEM | FastASEM | FullContext | NoMemory |
|---|---|---|---|---|
| locomo_0000 | 0.8424 | 0.8639 | 0.6236 | 0.5608 |
| locomo_0001 | 0.8245 | 0.0000 | — | — |
| locomo_0002 | 0.8358 | — | — | — |
| locomo_0003 | 0.8273 | — | — | — |
| locomo_0004 | 0.8437 | — | — | — |
| locomo_0005 | 0.8237 | — | — | — |
| locomo_0006 | 0.8327 | — | — | — |
| locomo_0007 | 0.8334 | — | — | — |
| locomo_0009 | 0.8317 | — | — | — |

**Judge**
| Conversation | ASEM | FastASEM | FullContext | NoMemory |
|---|---|---|---|---|
| locomo_0000 | 0.4171 | 0.5000 | 0.6000 | 0.0000 |
| locomo_0001 | 0.4381 | 0.0000 | — | — |
| locomo_0002 | 0.3717 | — | — | — |
| locomo_0003 | 0.3462 | — | — | — |
| locomo_0004 | 0.4587 | — | — | — |
| locomo_0005 | 0.3671 | — | — | — |
| locomo_0006 | 0.4842 | — | — | — |
| locomo_0007 | 0.4686 | — | — | — |
| locomo_0009 | 0.4069 | — | — | — |
