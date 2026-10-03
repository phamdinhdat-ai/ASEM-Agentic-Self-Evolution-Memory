# Static-bank evaluation — tag `ds_thg`, model `hotrovlg_vult_gpt_5_6_sol`

- dataset: `locomo10`
- bank root: `static/memory_banks\locomo10/ds_thg`
- config: `configs/models/vult55_openai.yaml`
- judge: `configs/models/vult55_openai.yaml`
- metrics: `em, rougeL, bertscore_f1`
- conversations: 1 (locomo_0000)
- QA pairs: 199 | elapsed: 2629.4s

| System | n | EM | ROUGE-L | BERTScore-F1 | Errors |
|---|---|---|---|---|---|
| ASEM-THG | 175 | 0.2343 | 0.4524 | 0.8552 | 8 |


**adversarial**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 23 | 0.4348 | 0.4386 | 0.6039 |

**multi_hop**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 32 | 0.0625 | 0.2917 | 0.8578 |

**open_domain**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 13 | 0.0000 | 0.1960 | 0.8514 |

**single_hop**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 70 | 0.2429 | 0.4677 | 0.9011 |

**temporal**
| System | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| ASEM-THG | 37 | 0.3243 | 0.6610 | 0.9237 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | ROUGE-L | BERTScore-F1 |
|---|---|---|---|---|
| locomo_0000 | 175 | 0.2343 | 0.4524 | 0.8552 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.2343 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.4524 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8552 |
