# Static-bank evaluation — tag `ds_thg`, model `deepseek_v4_flash`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/ds_thg`
- config: `configs/models/deepseek_openai.yaml`
- judge: `configs/models/deepseek_openai.yaml`
- metrics: `em, rougeL, bertscore_f1`
- conversations: 1 (locomo_0000)
- QA pairs: 199 | elapsed: 830.3s

| System | n | EM | ROUGE-L | BERTScore-F1 | Errors |
|---|---|---|---|---|---|
| ASEM-THG | 199 | 0.0754 | 0.2923 | 0.8679 | 0 |


**adversarial**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 47 | 0.0000 | 0.1169 | 0.8334 |

**multi_hop**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 32 | 0.0312 | 0.1587 | 0.8383 |

**open_domain**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 13 | 0.0000 | 0.1059 | 0.8452 |

**single_hop**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 70 | 0.0857 | 0.3576 | 0.8864 |

**temporal**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 37 | 0.2162 | 0.5728 | 0.9101 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| locomo_0000 | 199 | 0.0754 | 0.2923 | 0.8679 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.0754 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.2923 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8679 |
