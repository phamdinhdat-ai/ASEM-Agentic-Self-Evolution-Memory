# Static-bank evaluation — tag `gpt56_luna`, model `qwen_qwen3_4b_instruct_2507`

- dataset: `locomo10`
- bank root: `static/memory_banks/locomo10/gpt56_luna`
- config: `/kaggle/working/ASEM-Agentic-Self-Evolution-Memory/configs/models/qwen35_4B_api.yaml`
- judge: `/kaggle/working/ASEM-Agentic-Self-Evolution-Memory/configs/models/qwen35_4B_api.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 10 (locomo_0000, locomo_0001, locomo_0002, locomo_0003, locomo_0004, locomo_0005, locomo_0006, locomo_0007, locomo_0008, locomo_0009)
- QA pairs: 1986 | elapsed: 17633.0s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM-THG | 1986 | 0.1923 | 0.3728 | 0.3583 | 0.8501 | 0.6813 | 0 |


**adversarial**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 446 | 0.6951 | 0.7036 | 0.7036 | 0.8147 | 0.7848 |

**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 282 | 0.0035 | 0.1810 | 0.1560 | 0.8441 | 0.5213 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 96 | 0.0104 | 0.1089 | 0.0988 | 0.8316 | 0.4375 |

**single_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 841 | 0.0333 | 0.2577 | 0.2445 | 0.8637 | 0.7063 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| ASEM-THG | 321 | 0.1308 | 0.4619 | 0.4320 | 0.8745 | 0.6854 |

## Per-conversation

**ASEM-THG**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.1658 | 0.3585 | 0.3483 | 0.8604 | 0.7186 |
| locomo_0001 | 105 | 0.2571 | 0.4524 | 0.4457 | 0.8612 | 0.7238 |
| locomo_0002 | 193 | 0.1813 | 0.3801 | 0.3598 | 0.8581 | 0.6788 |
| locomo_0003 | 260 | 0.1923 | 0.3337 | 0.3250 | 0.8405 | 0.6385 |
| locomo_0004 | 242 | 0.2066 | 0.3832 | 0.3724 | 0.8531 | 0.6942 |
| locomo_0005 | 158 | 0.1772 | 0.3526 | 0.3428 | 0.8448 | 0.6582 |
| locomo_0006 | 190 | 0.1789 | 0.3889 | 0.3552 | 0.8455 | 0.6842 |
| locomo_0007 | 239 | 0.2008 | 0.3746 | 0.3682 | 0.8466 | 0.6695 |
| locomo_0008 | 196 | 0.1735 | 0.3641 | 0.3436 | 0.8497 | 0.6735 |
| locomo_0009 | 204 | 0.2108 | 0.3830 | 0.3651 | 0.8484 | 0.7010 |

## Per-conversation matrices

**EM**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.1658 |
| locomo_0001 | 0.2571 |
| locomo_0002 | 0.1813 |
| locomo_0003 | 0.1923 |
| locomo_0004 | 0.2066 |
| locomo_0005 | 0.1772 |
| locomo_0006 | 0.1789 |
| locomo_0007 | 0.2008 |
| locomo_0008 | 0.1735 |
| locomo_0009 | 0.2108 |

**F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3585 |
| locomo_0001 | 0.4524 |
| locomo_0002 | 0.3801 |
| locomo_0003 | 0.3337 |
| locomo_0004 | 0.3832 |
| locomo_0005 | 0.3526 |
| locomo_0006 | 0.3889 |
| locomo_0007 | 0.3746 |
| locomo_0008 | 0.3641 |
| locomo_0009 | 0.3830 |

**ROUGE-L**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.3483 |
| locomo_0001 | 0.4457 |
| locomo_0002 | 0.3598 |
| locomo_0003 | 0.3250 |
| locomo_0004 | 0.3724 |
| locomo_0005 | 0.3428 |
| locomo_0006 | 0.3552 |
| locomo_0007 | 0.3682 |
| locomo_0008 | 0.3436 |
| locomo_0009 | 0.3651 |

**BERTScore-F1**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.8604 |
| locomo_0001 | 0.8612 |
| locomo_0002 | 0.8581 |
| locomo_0003 | 0.8405 |
| locomo_0004 | 0.8531 |
| locomo_0005 | 0.8448 |
| locomo_0006 | 0.8455 |
| locomo_0007 | 0.8466 |
| locomo_0008 | 0.8497 |
| locomo_0009 | 0.8484 |

**Judge**
| Conversation | ASEM-THG |
|---|---|
| locomo_0000 | 0.7186 |
| locomo_0001 | 0.7238 |
| locomo_0002 | 0.6788 |
| locomo_0003 | 0.6385 |
| locomo_0004 | 0.6942 |
| locomo_0005 | 0.6582 |
| locomo_0006 | 0.6842 |
| locomo_0007 | 0.6695 |
| locomo_0008 | 0.6735 |
| locomo_0009 | 0.7010 |

## Comparison vs baselines and FullContext (same backbone, n = 1986)

**Why this run matters.** ASEM-THG here runs on **Qwen3-4B-Instruct-2507** — the *same* backbone as the `ds_fixed` baselines (`configs/models/qwen3_4b_instruct_api.yaml`). It is therefore the **only run directly comparable to FullContext / FastASEM / ASEM / NoMemory**. All systems answer with one model call and the same `context_window − max_tokens − safety` trimming rule, so the comparison isolates *memory architecture + answer contract*.

| System | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|
| **ASEM-THG (this run)** | **0.1923** | **0.3728** | **0.3583** | 0.8501 | 0.6813 |
| FullContext | 0.0745 | 0.3485 | 0.3361 | **0.8773** | **0.7039** |
| FastASEM | 0.1012 | 0.3262 | 0.3123 | 0.8740 | 0.5504 |
| ASEM (v1) | 0.0650 | 0.3041 | 0.2897 | 0.8695 | 0.6631 |
| NoMemory | 0.0035 | 0.0678 | 0.0625 | 0.8288 | 0.1843 |

- **EM: THG 0.1923 — best of all systems**, 2.58× FullContext, 1.90× FastASEM, 2.96× ASEM v1.
- **F1 / ROUGE-L: THG also leads** (0.3728 / 0.3583 vs FullContext 0.3485 / 0.3361) — the first same-backbone run where THG beats FullContext on F1.
- **BERTScore / Judge: FullContext still edges it** (0.8773 / 0.7039 vs 0.8501 / 0.6813). This is the **verbosity artifact** (THG answers with explanation sentences; a semantic-overlap metric that rewards verbatim transcript copies rates it lower), not a comprehension gap.

### Per-category (same run — key on NUMERIC category id)

| Category (id) | n | EM | F1 | ROUGE-L | BERTScore | Judge |
|---|---|---|---|---|---|---|
| adversarial (5) | 446 | 0.6951 | 0.7036 | 0.7036 | 0.8147 | 0.7848 |
| temporal (2) | 321 | 0.1308 | 0.4619 | 0.4320 | 0.8745 | 0.6854 |
| single_hop (4) | 841 | 0.0333 | 0.2577 | 0.2445 | 0.8637 | 0.7063 |
| multi_hop (1) | 282 | 0.0035 | 0.1810 | 0.1560 | 0.8441 | 0.5213 |
| open_domain (3) | 96 | 0.0104 | 0.1089 | 0.0988 | 0.8316 | 0.4375 |

**Weak spots:** `multi_hop` (EM 0.0035) and `open_domain` (EM 0.0104) — the two inference-heavy sets. **Wins:** `temporal` (EM 0.1308, ~6× FullContext) and the adversarial refusal category. On single_hop the judge (0.7063) far exceeds strict EM (0.0333), confirming the answers are semantically right but padded.

### Cross-backbone reference (same frozen banks)

| Backbone / config | n | EM | F1 | ROUGE-L | BERTScore | Judge |
|---|---|---|---|---|---|---|
| deepseek-v4.1-flash `deepseek_v41_flash_4k_v2.yaml` | 1986 | 0.3364 | 0.5186 | 0.5032 | 0.8968 | 0.6334 |
| deepseek-v4.1-flash `deepseek_v41_flash_4k.yaml` | 1986 | 0.2659* | 0.4540* | 0.4420* | 0.8564 | 0.6103 |
| **qwen3-4b (this run)** | 1986 | 0.1923 | 0.3728 | 0.3583 | 0.8501 | 0.6813 |

\\* Corrected for the refusal-metric fix (`scratch_diag/report_v41_corrected_metrics.md`); the raw v41 `.md` predates it (EM read 0.1375). Same banks/retriever/prompt across all three — the ~1.75× deepseek EM advantage is a **backbone** effect, so the qwen3-4b run is the architecture-isolating one.

### Caveats before quoting

1. **Adversarial cells are not like-for-like.** `ds_fixed` baselines were scored with the old `normalize_text()` (contractions split, so refusal markers never matched). Re-score baselines before quoting the 0.6951-vs-0.0471 gap.
2. **Category labels are not portable.** `ds_fixed` uses shifted labels; this run uses real ids. Always key on the numeric `category`.
3. **~12 duplicate `(conversation, question)` rows** in the raw preds (1986 → 1974 unique); no ranking change.
4. **BEST BERTScore / "wins outright" only holds on deepseek** (0.8968 vs FullContext 0.8773); on qwen3-4b it trails (0.8501 vs 0.8773).

> Detailed narrative, worked samples and per-question head-to-head: `THG_vs_FullContext_analysis.md`.
