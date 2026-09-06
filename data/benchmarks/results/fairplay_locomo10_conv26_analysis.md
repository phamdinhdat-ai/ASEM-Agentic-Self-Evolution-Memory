# Fair-Play Benchmark Analysis — LoCoMo conv-26

**Date:** 2026-09-03 · **Run timestamp:** 2026-09-03T15:31:20
**Summary JSON:** `data/benchmarks/results/fairplay_locomo10_conv26.json`
**Per-question preds:** `data/benchmarks/results/preds/fairplay_{ASEMv2,SimRetrieval,FullContext}_conv26.jsonl`
**Full run log:** `logs/fairplay_full.log`

---

## 1. Executive summary

Four memory/retrieval methods were compared under **identical conditions** on LoCoMo
conversation `conv-26` (Caroline & Melanie): the same 19-session / 419-turn conversation,
the same 117 questions, and the same metric functions.

| Rank | Method | EM% | ROUGE-L% | BERT-F1% | Judge% |
|:---:|--------|:---:|:---:|:---:|:---:|
| 1 | **FastASEM** | **29.1** | **43.6** | **88.2** | 70.1 |
| 2 | FullContext | 29.1 | 37.7 | 87.1 | **76.1** |
| 3 | SimRetrieval | 27.4 | 36.5 | 87.0 | 70.1 |
| 4 | ASEMv2 | 17.9 | 19.3 | 82.9 | 65.8 |

**Headline conclusions**

1. **FastASEM is the strongest memory method** on this conversation: best EM, best
   ROUGE-L, best BERTScore, and a decisive lead on Temporal Reasoning (43.2 EM vs
   2.7–21.6 for the others).
2. **ASEMv2 underperforms due to a fixable temporal-grounding bug**, not a retrieval
   failure: its notes and answers keep *relative* time expressions ("last week",
   "next month") while references use *absolute* dates ("7 May 2023"). This single
   issue accounts for most of its 11-point overall EM gap.
3. **FullContext (no memory) is a strong baseline** — it wins Judge% and three of four
   categories on EM, confirming that a memory system must beat "read the whole
   transcript" to be worthwhile.
4. **BERTScore is compressed** (82.9–88.2 across all methods) and should be treated as
   a floor, not a differentiator; EM / ROUGE-L / Judge expose the real differences.

---

## 2. Experimental setup (fair-play protocol)

To make the comparison fair, every method was held to the same constraints:

| Aspect | Setting |
|---|---|
| Conversation | LoCoMo `conv-26` (index 0), 19 sessions, 419 turns |
| Question set | The canonical 117 questions (from FastASEM's official run) — every method answers the *exact same* queries |
| Categories | Temporal Reasoning (37), Conversational Context (35), Single-Hop (32), Multi-Hop/Commonsense (13) |
| Config | `configs/presets/sota_benchmark.yaml` — OpenAI backend, model `gpt-5.4`, temp 0.1, embedder `all-MiniLM-L6-v2`; k1=30, k2=8, k=6, δ=0.25, λ=0.35, α=0.10, q0=0.50 |
| Freshness | ASEMv2, SimRetrieval, FullContext were **reset and re-run from scratch** (fresh banks). FastASEM's existing predictions were **re-scored** with the identical metric functions (its run was kept as reference). |
| Metrics | EM (normalized substring match), ROUGE-L (LCS F1), BERTScore-F1 (roberta-base, CPU), LLM-as-a-Judge (binary correct/incorrect) |
| Null handling | 0 null predictions across all 4 methods × 117 questions |

**Ingestion API differences (handled by the runner):**
- ASEMv2: `ingest_conversation(all_turns)` — flat turn list; batch note construction + RL memory ops + link evolution.
- SimRetrieval: `ingest_conversation(session_batches)` — list of `(label, turns)`; similarity index only.
- FullContext: no ingestion; full history is passed to the LLM per query.

**Runner:** `scripts/run_fair_play.py` (flags used: `--judge`).

---

## 3. Overall results

| Method | n | EM% | ROUGE-L% | BERT-F1% | Judge% |
|--------|:---:|:---:|:---:|:---:|:---:|
| FastASEM | 117 | **29.1** | **43.6** | **88.2** | 70.1 |
| FullContext | 117 | 29.1 | 37.7 | 87.1 | **76.1** |
| SimRetrieval | 117 | 27.4 | 36.5 | 87.0 | 70.1 |
| ASEMv2 | 117 | 17.9 | 19.3 | 82.9 | 65.8 |

Reading the table:
- **EM & ROUGE-L** (format-sensitive, retrieval-oriented): FastASEM leads; ASEMv2 far behind.
- **Judge%** (semantic correctness, format-agnostic): FullContext leads; the spread is much tighter (65.8–76.1), showing that all methods produce *semantically plausible* answers even when they fail exact matching.
- **BERTScore-F1**: narrow band (82.9–88.2) — high topical overlap everywhere; low discriminative power.

---

## 4. Per-category results

| Category (n) | FastASEM EM/RL/BS | ASEMv2 EM/RL/BS | SimRetrieval EM/RL/BS | FullContext EM/RL/BS |
|---|---|---|---|---|
| **Temporal Reasoning** (37) | **43.2 / 60.6 / 91.5** | 2.7 / 8.7 / 77.8 | 21.6 / 40.5 / 86.6 | 2.7 / 15.8 / 82.4 |
| **Conversational Context** (35) | 31.4 / 49.5 / 89.2 | 42.9 / 38.6 / 87.2 | 48.6 / 48.3 / 89.7 | **57.1 / 69.0 / 92.4** |
| **Single-Hop** (32) | 18.8 / 32.7 / 86.2 | 12.5 / 14.1 / 83.6 | 15.6 / 25.6 / 85.5 | **31.2 / 33.1 / 87.5** |
| **Multi-Hop / Commonsense** (13) | 7.7 / 6.3 / 81.3 | 7.7 / 10.1 / 84.1 | 15.4 / 20.6 / 84.5 | **23.1 / 27.3 / 85.2** |

### Category observations

- **Temporal Reasoning is the decisive category.** FastASEM (43.2 EM) is 2× SimRetrieval
  and ~16× ASEMv2/FullContext. This is exactly where a well-grounded memory system
  should win — and FastASEM does.
- **Conversational Context & Single-Hop favor FullContext.** With the whole transcript in
  context, the LLM can quote/paraphrase directly; memory-based retrieval adds noise here.
- **Multi-Hop is hard for everyone** (best EM = 23.1, FullContext). Link traversal /
  multi-hop expansion helps ASEMv2's ROUGE-L (10.1 vs 6.3 for FastASEM) but not EM.
- **ASEMv2's only category strength is Conversational Context EM (42.9)** — still below
  SimRetrieval (48.6) and FullContext (57.1).

---

## 5. Root-cause diagnosis: ASEMv2 temporal grounding

Sampling ASEMv2's Temporal Reasoning predictions against references reveals a consistent
failure mode — **relative vs absolute time expressions**:

| Question | Reference | ASEMv2 prediction | FastASEM prediction |
|---|---|---|---|
| When did Caroline go to the LGBTQ support group? | 7 May 2023 | *"The notes do not specify when Caroline went to the support group, only that…"* | **7 May 2023** |
| When did Melanie paint a sunrise? | 2022 | "Last year." | **2022** |
| When did Melanie run a charity race? | The Sunday before 25 May 2023 | "Melanie ran a charity race for mental health last Saturday." | **20 May 2023** |
| When is Melanie planning on going camping? | June 2023 | "Melanie is planning to go camping next month, during her kids' summer break…" | **June 2023** |
| When did Caroline give a speech at a school? | The week before 9 June 2023 | "Caroline gave a speech at a school event last week." | **2 June 2023** |
| When did Caroline meet up with friends/family? | The week before 9 June 2023 | "They met up last week." | **2 June 2023** |

**Diagnosis:**
1. ASEMv2's note construction stores time as it appears in the utterance (relative:
   "last week", "next month", "last Saturday") and does **not resolve it against the
   session date** at ingestion time.
2. The answer agent then echoes the relative expression (or, when the note lacks a time
   cue, says "the notes do not specify").
3. References use absolute dates, so EM (substring) and ROUGE-L (LCS) both fail even when
   the answer is semantically correct — which is why ASEMv2's Judge% (65.8) is much
   closer to the leaders than its EM% (17.9) suggests.

**Why FastASEM succeeds:** its notes/answers carry resolved absolute dates
("7 May 2023", "20 May 2023"), matching the reference format.

**Impact estimate:** ASEMv2's Temporal EM is 2.7/37. If temporal answers were normalized
to absolute dates, even a partial recovery (e.g., to SimRetrieval's 21.6) would lift
overall EM from 17.9 to ~23; matching FastASEM's 43.2 would lift it to ~28.

---

## 6. Metric behavior notes

- **EM** (normalized exact/substring): most sensitive to format; rewards terse,
  reference-matching answers. Best for ranking retrieval-style systems.
- **ROUGE-L** (LCS F1): partially tolerant of wording; still format-sensitive.
- **BERTScore-F1** (roberta-base): compressed to 82.9–88.2 — measures topical/semantic
  overlap, not correctness. Use as a sanity floor only.
- **Judge%** (LLM binary): format-agnostic semantic correctness. The tight spread
  (65.8–76.1) shows all four methods are "mostly right" semantically; the large EM gaps
  are largely a *format/grounding* artifact, not a knowledge gap.
- **Known discrepancy:** FastASEM's original official JSON reported `em_score 34.0`;
  re-scoring with the shared `compute_em` gives 29.1. The fair-play table uses the
  consistent function for all four methods.

---

## 7. Limitations

- **Single conversation** (conv-26). Results may not generalize; LoCoMo has 10
  conversations with different topics/lengths.
- **Single LLM backend** (gpt-5.4, temp 0.1). Judge and answer generation share the
  provider; judge bias is possible.
- **FastASEM not re-run fresh** — its predictions come from a prior run and were only
  re-scored. If its ingestion config changed since, the comparison is slightly asymmetric
  (mitigated by using the same canonical question set).
- **BERTScore on CPU** with roberta-base — approximate but consistent across methods.

---

## 8. Recommendations / next steps

1. **Fix ASEMv2 temporal grounding (highest ROI).** Resolve relative→absolute time at
   ingestion (note construction has the session date available) or at answer time
   (answer agent resolves "last week" against the note's timestamp). Then re-run only
   ASEMv2 (`--systems ASEMv2`) and re-score.
2. **Extend to all 10 LoCoMo conversations** for a robust, publication-ready comparison
   (average EM/ROUGE-L/Judge across convs; report variance).
3. **Add a temporal-accuracy probe** as a dedicated metric (date-extraction F1) to
   separate grounding quality from retrieval quality.
4. **Investigate FullContext's Temporal weakness (2.7 EM)** — likely the same relative-date
   echo; a date-normalization post-processor could help all methods.
5. **Multi-hop focus:** all methods are weak here (≤23.1 EM); ASEMv2's link traversal
   shows promise on ROUGE-L — worth a targeted ablation.

---

## 9. Reproduction

```bash
conda activate memory-r1

# Full fair-play run (3 fresh systems + FastASEM re-score, with judge)
python scripts/run_fair_play.py --judge

# Smoke test (3 questions)
python scripts/run_fair_play.py --limit 3

# Re-score existing fairplay_*.jsonl without LLM runs
python scripts/run_fair_play.py --score-only

# Re-run a single system only
python scripts/run_fair_play.py --systems ASEMv2 --judge
```

Outputs:
- `data/benchmarks/results/fairplay_locomo10_conv26.json` — summary (overall + per-category)
- `data/benchmarks/results/preds/fairplay_{ASEMv2,SimRetrieval,FullContext}_conv26.jsonl` — per-question (idx, session_id, category, query, pred, ref, em, rouge_l, judge_correct)
- `logs/fairplay_full.log` — full run log
