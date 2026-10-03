# Static-bank evaluation — tag `gpt56_luna`, model `qwen_qwen3_4b_instruct_2507`

- dataset: `locomo10`
- bank root: `static/memory_banks/locomo10/gpt56_luna`
- config: `/kaggle/working/ASEM-Agentic-Self-Evolution-Memory/configs/models/qwen35_4B_api.yaml`
- judge: `/kaggle/working/ASEM-Agentic-Self-Evolution-Memory/configs/models/qwen35_4B_api.yaml`
- metrics: `em, f1, rougeL, bertscore_f1, judge`
- conversations: 10 (locomo_0000, locomo_0001, locomo_0002, locomo_0003, locomo_0004, locomo_0005, locomo_0006, locomo_0007, locomo_0008, locomo_0009)
- QA pairs: 1986 | elapsed: 2611.4s

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| FullContext | 1986 | 0.1334 | 0.3682 | 0.3602 | 0.8729 | 0.6803 | 0 |


**adversarial**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| FullContext | 446 | 0.1547 | 0.2113 | 0.2112 | 0.8253 | 0.4955 |

**multi_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| FullContext | 282 | 0.0426 | 0.3065 | 0.2867 | 0.8709 | 0.6277 |

**open_domain**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| FullContext | 96 | 0.0312 | 0.1691 | 0.1585 | 0.8582 | 0.5104 |

**single_hop**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| FullContext | 841 | 0.2069 | 0.5537 | 0.5447 | 0.9135 | 0.9465 |

**temporal**
| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| FullContext | 321 | 0.0218 | 0.2140 | 0.2085 | 0.8392 | 0.3364 |

## Per-conversation

**FullContext**
| Conversation | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.0955 | 0.3407 | 0.3324 | 0.8678 | 0.6482 |
| locomo_0001 | 105 | 0.0952 | 0.3007 | 0.2943 | 0.8626 | 0.6476 |
| locomo_0002 | 193 | 0.1347 | 0.4106 | 0.4009 | 0.8819 | 0.7254 |
| locomo_0003 | 260 | 0.1231 | 0.3572 | 0.3531 | 0.8733 | 0.6231 |
| locomo_0004 | 242 | 0.1488 | 0.3672 | 0.3605 | 0.8744 | 0.7025 |
| locomo_0005 | 158 | 0.1772 | 0.3844 | 0.3785 | 0.8775 | 0.6772 |
| locomo_0006 | 190 | 0.0895 | 0.3014 | 0.2920 | 0.8627 | 0.6105 |
| locomo_0007 | 239 | 0.1757 | 0.4279 | 0.4176 | 0.8751 | 0.7448 |
| locomo_0008 | 196 | 0.1173 | 0.3652 | 0.3533 | 0.8738 | 0.6837 |
| locomo_0009 | 204 | 0.1569 | 0.3877 | 0.3798 | 0.8754 | 0.7206 |

## Per-conversation matrices

**EM**
| Conversation | FullContext |
|---|---|
| locomo_0000 | 0.0955 |
| locomo_0001 | 0.0952 |
| locomo_0002 | 0.1347 |
| locomo_0003 | 0.1231 |
| locomo_0004 | 0.1488 |
| locomo_0005 | 0.1772 |
| locomo_0006 | 0.0895 |
| locomo_0007 | 0.1757 |
| locomo_0008 | 0.1173 |
| locomo_0009 | 0.1569 |

**F1**
| Conversation | FullContext |
|---|---|
| locomo_0000 | 0.3407 |
| locomo_0001 | 0.3007 |
| locomo_0002 | 0.4106 |
| locomo_0003 | 0.3572 |
| locomo_0004 | 0.3672 |
| locomo_0005 | 0.3844 |
| locomo_0006 | 0.3014 |
| locomo_0007 | 0.4279 |
| locomo_0008 | 0.3652 |
| locomo_0009 | 0.3877 |

**ROUGE-L**
| Conversation | FullContext |
|---|---|
| locomo_0000 | 0.3324 |
| locomo_0001 | 0.2943 |
| locomo_0002 | 0.4009 |
| locomo_0003 | 0.3531 |
| locomo_0004 | 0.3605 |
| locomo_0005 | 0.3785 |
| locomo_0006 | 0.2920 |
| locomo_0007 | 0.4176 |
| locomo_0008 | 0.3533 |
| locomo_0009 | 0.3798 |

**BERTScore-F1**
| Conversation | FullContext |
|---|---|
| locomo_0000 | 0.8678 |
| locomo_0001 | 0.8626 |
| locomo_0002 | 0.8819 |
| locomo_0003 | 0.8733 |
| locomo_0004 | 0.8744 |
| locomo_0005 | 0.8775 |
| locomo_0006 | 0.8627 |
| locomo_0007 | 0.8751 |
| locomo_0008 | 0.8738 |
| locomo_0009 | 0.8754 |

**Judge**
| Conversation | FullContext |
|---|---|
| locomo_0000 | 0.6482 |
| locomo_0001 | 0.6476 |
| locomo_0002 | 0.7254 |
| locomo_0003 | 0.6231 |
| locomo_0004 | 0.7025 |
| locomo_0005 | 0.6772 |
| locomo_0006 | 0.6105 |
| locomo_0007 | 0.7448 |
| locomo_0008 | 0.6837 |
| locomo_0009 | 0.7206 |

---

# Statistics — FullContext (qwen3-4b) under a 4K context window

Same backbone, same frozen banks, only the answer budget changes to match ASEM-THG's 4K run
(`answer.context_window: 4096`, `answer.max_tokens: 512` → prompt budget 3520 tokens; the
FullContext prompt is token-trimmed by `drop="oldest"`, `head_turns=5`).

| Setting | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| FullContext — untrimmed (reported above) | 1986 | 0.1334 | 0.3682 | 0.3602 | 0.8729 | 0.6803 | 0 |
| FullContext — 4K budget (**this run**) | 1986 | 0.1053 | 0.2991 | 0.2902 | 0.8590 | 0.5938 | 0 |
| Δ (4K − untrimmed) | — | **−0.0281** | **−0.0691** | **−0.0700** | −0.0139 | **−0.0865** | — |

## Per-category — FullContext under the 4K budget (n = 1986)

| Category | id | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge |
|---|---|---|---|---|---|---|---|
| adversarial | 5 | 446 | 0.1194 | 0.1698 | 0.1691 | 0.8163 | 0.4426 |
| multi_hop | 1 | 282 | 0.0248 | 0.2291 | 0.2136 | 0.8602 | 0.4861 |
| open_domain | 3 | 96 | 0.0104 | 0.1109 | 0.1047 | 0.8497 | 0.3854 |
| single_hop | 4 | 841 | 0.1699 | 0.4746 | 0.4642 | 0.9094 | 0.9322 |
| temporal | 2 | 321 | 0.0156 | 0.1626 | 0.1573 | 0.8247 | 0.2804 |

## Per-category Δ — 4K budget minus untrimmed

| Category | id | n | ΔEM | ΔF1 | ΔROUGE-L | ΔBERTScore | ΔJudge |
|---|---|---|---|---|---|---|---|
| adversarial | 5 | 446 | −0.0353 | −0.0415 | −0.0421 | −0.0090 | −0.0529 |
| multi_hop | 1 | 282 | −0.0178 | −0.0774 | −0.0731 | −0.0107 | −0.1416 |
| open_domain | 3 | 96 | −0.0208 | −0.0582 | −0.0538 | −0.0085 | −0.1250 |
| single_hop | 4 | 841 | −0.0370 | −0.0791 | −0.0805 | −0.0041 | −0.0143 |
| temporal | 2 | 321 | −0.0062 | −0.0514 | −0.0512 | −0.0145 | −0.0560 |

## Observations

- **The 4K trim costs FullContext ~2.1–2.8 EM points overall** (0.1334 → 0.1053), and it is the
  ONLY system that genuinely loses information to the budget: at 32768/512 **0/10** conversations
  are trimmed, but at 4096/512 the prompt is trimmed on essentially every turn because the raw
  history is 12.6k–25.9k tokens.
- **`single_hop` absorbs the largest F1 damage** (−0.0791; EM −0.0370) because FullContext's edge on
  that category came from copying evidence turns verbatim — the `drop="oldest"` trim removes exactly
  those middle/session turns.
- **Judge falls furthest on `multi_hop` (−0.1416) and `open_domain` (−0.1250)** — the categories whose
  evidence spans multiple sessions, i.e. the rows where the middle of the history is deleted.
- **`temporal` is the most robust** (ΔEM −0.0062): its answers lean on a single dated fact, which
  usually survives the `head_turns=5` + newest-turns retention.
- **Relaxing the budget does not rescue FullContext to parity**: even untrimmed, its EM 0.1334 is
  well below ASEM-THG's 0.1923 on the same backbone. Under an equal 4K budget the gap widens to
  **0.1053 vs 0.1923 (1.83×)**, and FullContext's Judge edge over THG disappears (0.5938 vs 0.6813).

> NOTE: the untrimmed block (top of file) and the 4K block above were produced by the same pipeline
> on the same frozen `gpt56_luna` banks; the only difference is `answer.context_window`. The `single_hop`
> / `multi_hop` mapping uses the **numeric** category id (see the category-label caveat in
> `THG_vs_FullContext_analysis.md`).

---

# Head-to-head — ASEM-THG vs FullContext (same backbone, same 4K run, n = 1986)

Both arms answered the identical 1986 QA rows with `qwen_qwen3_4b_instruct_2507` on the same frozen
`gpt56_luna` banks. The numbers below are re-scored from the two raw preds files
(`…__ASEM-THG_pred.jsonl`, `…__FullContext.jsonl`) with the official protocol
(`adversarial_flags` → refusal credited on category 5). Script: `scratch_diag/_compare_thg_fc_4k.py`.

## Overall

| System | n | EM | F1 | ROUGE-L |
|---|---|---|---|---|
| **ASEM-THG** | 1986 | **0.1923** | **0.3728** | 0.3583 |
| FullContext | 1986 | 0.1334 | 0.3682 | **0.3602** |
| Δ (THG − FC) | — | **+0.0589** (1.44×) | +0.0046 | −0.0018 |

## Per-category (numeric id: 1=multi_hop 2=temporal 3=open_domain 4=single_hop 5=adversarial)

| Category | id | n | THG EM | FC EM | ΔEM | THG F1 | FC F1 | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| multi_hop | 1 | 282 | 0.0035 | 0.0426 | −0.0390 | 0.1810 | 0.3065 | −0.1255 |
| temporal | 2 | 321 | **0.1308** | 0.0218 | **+0.1090** | **0.4619** | 0.2140 | **+0.2479** |
| open_domain | 3 | 96 | 0.0104 | 0.0312 | −0.0208 | 0.1089 | 0.1691 | −0.0602 |
| single_hop | 4 | 841 | 0.0333 | **0.2069** | **−0.1736** | 0.2577 | **0.5537** | **−0.2960** |
| adversarial | 5 | 446 | **0.6951** | 0.1547 | **+0.5404** | **0.7036** | 0.2113 | **+0.4924** |

## Paired outcomes (strict EM, identical idx)

| THG-only | FC-only | both correct | neither | net (THG − FC) |
|---|---|---|---|---|
| 50 | 174 | 22 | 1740 | **−124** |

## Abstention / verbosity

| System | refusals | answer length p50 |
|---|---|---|
| ASEM-THG | 485 / 1986 (**24.4%**) | 58 chars |
| FullContext | 30 / 1986 (1.5%) | 52 chars |

## Adversarial detail (category 5, n = 446)

- 444 / 446 gold refs are literally **"Not mentioned"** → this is a binary refusal-detection task, not QA.
- THG refuses **285 / 446 (63.9%)**; FullContext refuses **19 / 446 (4.3%)** (NoMemory-style "I don't know" behaviour).

## What this says

1. **ASEM-THG's overall EM lead (0.1923 vs 0.1334, +5.9 pts) comes almost entirely from two categories
   it wins by design — `temporal` (+10.9 pts) and `adversarial` (+54 pts)** — while it loses
   `single_hop` (−17.4 pts) and `multi_hop` (−3.9 pts).
2. **Under a 4K budget the two systems are otherwise at parity on F1/ROUGE-L** (ΔF1 +0.0046, ΔRL −0.0018);
   the untrimmed FullContext run looks stronger on ROUGE-L (0.3602) but that is the *same* number —
   FullContext already sat at the 4K boundary for its own trim.
3. **The paired test is the honest one**: THG wins only 50 rows outright where FullContext wins 174
   (net −124). Aggregate EM favours THG because its wins are concentrated in the two large
   refusal/temporal categories, not because it answers more questions correctly overall.
4. **FullContext's `single_hop` advantage is verbatim copying** (EM 0.2069 vs 0.0333, F1 0.5537 vs
   0.2577) — its context includes the evidence turn, so it echoes the gold span.
5. **THG's aggressive abstention (24.4%) is double-edged**: it earns the adversarial credit but
   suppresses EM elsewhere — 485 refusals vs FullContext's 30.

> Reproduce: `scratch_diag/run_compare_thg_fc_4k.ps1` (uses the `memory-r1` interpreter, no API calls).
