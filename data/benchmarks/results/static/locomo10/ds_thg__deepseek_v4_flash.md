# Static-bank evaluation — tag `ds_thg`, model `deepseek_v4_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/ds_thg`
- config: `configs/models/deepseek_openai.yaml`
- judge: `configs/models/deepseek_openai.yaml`
- metrics: `em, rougeL, bertscore_f1`
- conversations: 1 (locomo_0000)
- QA pairs: 199 | elapsed: 877.6s

| System | n | EM | ROUGE-L | BERTScore-F1 | Errors |
|---|---|---|---|---|---|
| ASEM-THG | 199 | 0.1608 | 0.3624 | 0.8831 | 0 |


**adversarial**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 47 | 0.0000 | 0.0775 | 0.8257 |

**multi_hop**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 32 | 0.0938 | 0.2640 | 0.8673 |

**open_domain**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 13 | 0.0000 | 0.1796 | 0.8586 |

**single_hop**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 70 | 0.1857 | 0.4587 | 0.9029 |

**temporal**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 37 | 0.4324 | 0.6915 | 0.9409 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| locomo_0000 | 199 | 0.1608 | 0.3624 | 0.8831 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.1608 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3624 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8831 |
