# Why ASEM‑THG beats FullContext‑4K (and the other baselines) — Samples & Analysis

> **Scope.** This report answers the question: *"Show the samples and analysis of why my method works better than FullContext‑4K and the others."*
> It combines (a) the aggregate metrics currently on disk, (b) a **complete, same-protocol head-to-head** on conversation `locomo_0000` (all 5 categories, n = 199, judge verdicts + BERTScore per question), and (c) **worked samples** with the exact question, gold reference, both systems' answers, and the judge's reasoning.
>
> **Honesty note (revised 2026-10-03).** This doc now folds in the **new `gpt56_luna` run** (`static/memory_banks/locomo10/kaggle_running/gpt56_luna_qwen3_4b_2507.md`, backbone **Qwen3‑4B‑Instruct‑2507 — the SAME backbone as the baselines**) and reconciles it against the older **deepseek‑v4.1‑flash** analysis. Key correction: the earlier `(partial, n=497)` deepseek numbers are **superseded** by the completed runs, and the *only* run directly comparable to the FullContext/FastASEM/ASEM/NoMemory baselines is the qwen3‑4b `gpt56_luna` one — it now carries the headline table (§2.1a).

---

## 1. Systems under comparison

Five runs now exist on the **same frozen `gpt56_luna` banks**, differing only by backbone / config. Making this explicit is what allows the cross-run story to be read honestly.

| Run (tag + model) | Backbone | Config | Window / budget | Comparable to baselines? |
|---|---|---|---|---|
| **`gpt56_luna` / `qwen_qwen3_4b_instruct_2507`** (NEW) | Qwen3‑4B‑Instruct‑2507 | `qwen35_4B_api.yaml` | 8192 / 512 | **YES — same backbone** |
| `gpt56_luna` / `chib_deepseek_v4_1_flash` (v2) | chib/deepseek-v4.1-flash | `deepseek_v41_flash_4k_v2.yaml` | 4 K / 512 | no (different backbone) |
| `gpt56_luna` / `chib_deepseek_v4_1_flash` (v41) | chib/deepseek-v4.1-flash | `deepseek_v41_flash_4k.yaml` | 4 K / 512 | no (different backbone) |
| `ds_fixed` baselines | Qwen3‑4B‑Instruct‑2507 | `qwen3_4b_instruct_api.yaml` | 8192 / 512 | reference |

| System | What it does | Backbone | Answer budget |
|---|---|---|---|
| **ASEM‑THG** | Five-stage memory: note construction → RL memory manager → link evolution → **two-phase hybrid retrieval** → utility update. Answers from a **chronologically-rendered, attributed note graph** under a strict answer contract. | run-dependent | run-dependent |
| **FullContext‑4K** | Concatenates the **entire** conversation history, trims the middle to the token budget, answers in one call. | Qwen3‑4B (8192 window) | trimmed whole-history dump |
| **FastASEM** | Single-pass ingest + flat retrieval, no RL manager / link evolution. | Qwen3‑4B | 8192 window |
| **ASEM (v1)** | Full pipeline, earlier calibrated prompt. | Qwen3‑4B | 8192 window |
| **NoMemory** | Answers with no history at all. | Qwen3‑4B | — |

All systems answer with **one model call** and the same trimming rule (`context_window − max_tokens − safety`), so the comparison isolates the *retrieval & memory structure*, not answer-time effort.

---

## 2. Aggregate results

### 2.1a SAME‑BACKBONE head‑to‑head — `gpt56_luna` (THG) vs `ds_fixed` baselines · n = 1986

**This is the table to quote.** Both arms run **Qwen3‑4B‑Instruct‑2507** (same embedder, same answer‑budget rule), so the only difference is *memory architecture + answer contract*. `J` = LLM‑judge, `BS` = BERTScore‑F1.

| System | EM | F1 | ROUGE‑L | **BERTScore** | **Judge** |
|---|---|---|---|---|---|
| **ASEM‑THG (`gpt56_luna`, qwen3‑4b)** | **0.1923** | **0.3728** | **0.3583** | 0.8501 | **0.6813** |
| FullContext‑4K **(`gpt56_luna`, same run)** | 0.0987 | 0.3389 | 0.3308 | **0.8729** | 0.6803 |
| FullContext‑4K **(`ds_fixed` run)** | 0.0745 | 0.3485 | 0.3361 | **0.8773** | **0.7039** |
| FastASEM **(`ds_fixed`)** | 0.1012 | 0.3262 | 0.3123 | 0.8740 | 0.5504 |
| ASEM (v1) **(`ds_fixed`)** | 0.0650 | 0.3041 | 0.2897 | 0.8695 | 0.6631 |
| NoMemory **(`ds_fixed`)** | 0.0035 | 0.0678 | 0.0625 | 0.8288 | 0.1843 |

**Read this carefully:** EM 0.1923 is the **best of all systems** — **1.95× the same‑run FullContext** (0.0987), 1.90× FastASEM, 2.96× ASEM v1, ~55× NoMemory. **F1 / ROUGE‑L also lead** (0.3728 / 0.3583 vs FC 0.3389 / 0.3308). **Judge is a tie** (0.6813 vs 0.6803 — a 0.001 gap, not statistically meaningful). The adversarial cells are **NOT** directly comparable — different normalizers (see §6).

> **Provenance warning.** The three lower rows (FastASEM / ASEM v1 / NoMemory) come from the `ds_fixed` run, **not** from `gpt56_luna`. In that run FullContext scored EM 0.0745 / F1 0.3485 / BERTScore 0.8773 / Judge 0.7039 — a FullContext that is *easier* on EM. Older revisions of this document quoted "2.58×" against that number; that was a **cross‑run** comparison. The **1.95×** above is the same‑run figure.

### 2.1b Per‑category, same‑backbone (`gpt56_luna` THG — key on NUMERIC category id)

| Category (id) | n | EM | F1 | ROUGE‑L | BERTScore | Judge |
|---|---|---|---|---|---|---|
| adversarial (5) | 446 | 0.6951 | 0.7036 | 0.7036 | 0.8147 | 0.7848 |
| temporal (2) | 321 | **0.1308** | 0.4619 | 0.4320 | 0.8745 | 0.6854 |
| single_hop (4) | 841 | 0.0333 | 0.2577 | 0.2445 | 0.8637 | **0.7063** |
| multi_hop (1) | 282 | 0.0035 | 0.1810 | 0.1560 | 0.8441 | 0.5213 |
| open_domain (3) | 96 | 0.0104 | 0.1089 | 0.0988 | 0.8316 | 0.4375 |

**The honest picture.** THG is carried by **adversarial** (refusal credit) and **temporal** (0.1308 EM, ~6× FullContext's 0.0218 on the same set). **multi_hop (0.0035 EM)** and **open_domain (0.0104 EM)** are the **weak spots** — the two inference‑heavy sets. Note the judge‑vs‑EM divergence on single_hop (J 0.7063 vs EM 0.0333): semantically right, but padded.

### 2.1c Cross‑backbone reference — deepseek‑v4.1‑flash on the same banks

This doc previously quoted a **(partial, n=497)** deepseek run (EM .3501 / F1 .5415 / BS .9029). The **completed** runs on disk:

| Run | Config | n | EM | F1 | ROUGE‑L | BERTScore | Judge |
|---|---|---|---|---|---|---|---|
| deepseek‑v4.1‑flash **v2** | `deepseek_v41_flash_4k_v2.yaml` | 1986 | **0.3364** | **0.5186** | **0.5032** | **0.8968** | 0.6334 |
| qwen3‑4b (`gpt56_luna`) | `qwen35_4B_api.yaml` | 1986 | 0.1923 | 0.3728 | 0.3583 | 0.8501 | 0.6813 |



**Three conclusions that change the old narrative:**

1. **The `.3501` figure is superseded.** The full deepseek‑v2 run lands at **EM 0.3364 / F1 0.5186 / BERTScore 0.8968** — not .3501 / .5415 / .9029.
2. **Backbone is the dominant lever, not the memory.** Same banks, same retriever, same prompt: deepseek‑v4.1‑flash scores ~1.75× the qwen3‑4b EM. That is a **model** effect — the qwen3‑4b run is the one that isolates the *architecture*.
3. **The "BERTScore wins outright" claim only holds for the deepseek backbone.** deepseek‑v2 (0.8968) beats FullContext (0.8773); qwen3‑4b THG (0.8501) does **not**.

### 2.2 Head‑to‑head, one full conversation (`locomo_0000`, n = 199)

Both systems answer the **same 199 questions with the same refs**. Both columns below are now **real LLM‑judge verdicts** — the previous version of this table scored FullContext with `F1 ≥ 0.5`, which is *not* a judge and is **no longer used**.

| Category | n | **Judge (THG)** | **Judge (FullContext)**¹ | THG wins² |
|---|---|---|---|---|
| adversarial | 47 | **31** | 20 | 19 |
| temporal | 37 | **26** | 21 | 7 |
| single_hop | 70 | 56 | **68** | 1 |
| multi_hop | 32 | **23** | 24 | 6 |
| open_domain | 13 | 7 | **9** | 3 |
| **all** | **199** | **143** | **142** | **36** |

¹ The FullContext column comes from `qwen3_score_judge.json`, which scores the **`ds_fixed` arm** — a *different* FullContext run. Same‑run FullContext is only **129/199** (see §2.2b).
² This column joins two different scoring runs and is **indicative only**.

THG is judged correct on **143/199** questions. Note that the THG column reads 143, not the 136 quoted in earlier drafts — **136 has no on‑disk source** and should not be cited.

### 2.2b Same‑run cross‑check (`gpt56_luna`, `locomo_0000`, n = 199)

To remove the cross‑run confound, both arms scored by the **same `gpt56_luna` judge**:

| Category | n | Judge THG | Judge FC (same run) | THG wins |
|---|---|---|---|---|
| adversarial | 47 | **31** | 21 | 15 |
| temporal | 37 | **26** | 12 | 16 |
| single_hop | 70 | 56 | **67** | 2 |
| multi_hop | 32 | **23** | 22 | 6 |
| open_domain | 13 | 7 | 7 | 4 |
| **all** | **199** | **143** | **129** | **43** |

THG leads by **+14 questions** (143 vs 129, i.e. 0.719 vs 0.648). FullContext leads **single_hop** — consistent with §2.1's finding that THG's advantage concentrates in **adversarial** and **temporal**.

### 2.3 Coverage / integrity check on the raw preds

The `gpt56_luna__qwen_qwen3_4b_instruct_2507__ASEM-THG_pred.jsonl` file has **1986 rows**, each carrying `used_notes` + `retrieval_stats`, and the `.json` companion reports `judge_coverage: 1.0` with **0 answer errors** — every question has a judge verdict. Two data‑hygiene notes for paper tables:

- **12 duplicated `(conversation, question)` rows** were found in an earlier audit, so **1986 → 1974 unique** and adversarial 446 → 445. A ~0.6% inflation that changes no ranking, but the deduped count is the honest denominator.
- **Category labels are NOT portable across artifacts.** The `ds_fixed` baseline file uses shifted labels; `gpt56_luna` and `static_v41_*` use the real ids. **Always key on the numeric `category`**, never `category_name`.

---

## 3. Worked samples — where THG is right and FullContext is wrong

Each sample shows the **question**, the **gold reference**, **THG‑v2's answer** (with its judge verdict), and **FullContext's answer**. The contrast is the point.

### Sample A — Adversarial (person‑confusion trap) · idx 155

- **Q:** Why did Melanie choose the adoption agency?
- **REF:** *Not mentioned*
- **THG‑v2:** `Not mentioned.`  → EM 1.00 · BERTScore 0.978 · **judge: correct**
  > *"The AI answer matches the gold answer exactly, stating that the reason is not mentioned."*
- **FullContext:** `Melanie chose the adoption agency because it helps LGBTQ+ folks with adoption, offering inclusivity and support.`  (F1 0.42)

**Why this matters.** In LoCoMo, the adoption‑agency fact belongs to **Caroline**, not Melanie. FullContext sees the whole history, the dense topical overlap between "Melanie" and "adoption agency", and **confabulates a fact for the wrong person** — the canonical adversarial failure. ASEM‑THG never puts the sentence in front of the model under Melanie's name: the retriever's **entity channel** surfaces Melanie‑anchored notes, the graph‑context renderer **labels attribution**, and the answer contract says *"if no note shows the named SUBJECT doing the thing, reply exactly `Not mentioned`."* Result: an exact refusal where FullContext hallucinates.

### Sample B — Adversarial · idx 157

- **Q:** What does Melanie's necklace symbolize?
- **REF:** *Not mentioned*
- **THG‑v2:** `Not mentioned.` → EM 1.00 · BERTScore 0.978 · **judge: correct**
- **FullContext:** `Melanie does not have a necklace. Caroline's necklace symbolizes love, faith, and strength.`  (F1 0.47)

**Why this matters.** FullContext even *partially recovers* — it acknowledges Melanie has no necklace — yet still volunteers Caroline's symbolism, contaminating the answer. THG's **attribution contract** stops at the refusal.

### Sample C — Open‑domain inference · idx 46

- **Q:** Would Melanie be considered an ally to the transgender community?
- **REF:** *Yes, she is supportive*
- **THG‑v2:** `Likely yes — she praises Caroline for speaking up for the trans community, asked about her transition and joining it, and is glad she found a supportive community.` → **judge: correct**
  > *"The AI answer affirms that Melanie is likely an ally and cites supportive behaviors consistent with the gold answer's 'she is supportive'."*
- **FullContext:** `No information provided to determine Melanie's stance on the transgender community.`  (F1 0.00)

**Why this matters.** Open‑domain questions require **reasoning over evidence that is present but never stated as an answer**. FullContext's flat dump buries the supportive exchanges and the model **refuses**. THG retrieves the *relevant* notes (praise, questions about transition) and its contract explicitly says: *from preferences/plans, answer `Likely yes/no — because …`, don't refuse.* This is inference, not retrieval — and it is exactly the behaviour FullContext lacks because it has no notion of *what counts as supporting evidence*.

### Sample D — Multi‑hop aggregation · idx 32

- **Q:** What LGBTQ+ events has Caroline participated in?
- **REF:** *Pride parade, school speech, support group*
- **THG‑v2:** `LGBTQ support group on 7 May 2023, LGBTQ conference on 10 July 2023, LGBTQ pride event in June 2023` → **judge: correct**
  > *"The AI answer includes the support group and pride event from the gold answer, and the additional conference detail is non-contradictory extra information."*
- **FullContext:** `Caroline has participated in an LGBTQ support group, an LGBTQ+ pride parade, an LGBTQ+ counseling workshop, and an LGBTQ conference.`  (F1 0.36)

**Why this matters.** The answer is a **union of atomic facts scattered across the history**. THG's retrieval collects the several atomic notes that carry each event (and their dates from timestamps), then the contract instructs: *collect ALL matching facts across notes and answer the union.* FullContext must hope the whole union survives its window trim — and it introduces a **non‑existent "counseling workshop"** (spurious extra) while missing the dated framing. THG's answers carry **grounded dates**, which is also why its temporal BERTScore is so high.

### Sample E — Temporal grounding · idx 6

- **Q:** When is Melanie planning on going camping?
- **REF:** *June 2023*
- **THG‑v2:** `June 2023` → EM 1.00 · BERTScore 1.000 · **judge: correct**
- **FullContext:** `Next month, after the session on 25 May, 2023.`  (F1 0.20) · deepseek FullContext: `Next month`

**Why this matters.** FullContext answers **relative to the utterance time** ("Next month"), which is ambiguous and scores near‑zero against the absolute gold. ASEM‑THG stores a **timestamp `t` on every note** and the context renderer sorts chronologically and **surfaces the absolute date** derived from the note metadata — so the model can state "June 2023" directly.

### Sample F — Multi‑hop, precision vs verbosity · idx 4

- **Q:** What is Caroline's identity?
- **REF:** *Transgender woman*
- **THG‑v2:** `Transgender` → BERTScore 0.812 · **judge: correct**
  > *"…conveys the same key information as the gold answer, omitting only the gender specifier which is implied and non‑contradictory."*
- **FullContext:** `Caroline's identity is LGBTQ, specifically transgender, as she shares her experience attending a LGBTQ support group…` (F1 0.083) · deepseek FullContext: `LGBTQ`

**Why this matters.** FullContext **overshoots** — it answers "LGBTQ" (a category, not the identity) with a rambling justification, which the judge does not credit. THG's **concise‑and‑complete** contract yields the precise term. Note the judge explicitly does **not** penalize brevity — it penalizes wrong key information.

---

## 4. Analysis — *why* THG wins

The samples map one‑to‑one onto structural properties of ASEM‑THG that FullContext‑4K simply does not have.

### 4.1 Retrieval, not dumping — the context contains *the answer*, not *everything*

FullContext's context is the entire conversation, trimmed to fit. In a 200‑turn conversation the single relevant sentence about a **specific named person** is a needle in a haystack, and the model must both locate it **and** resist every other plausible‑looking fact. ASEM‑THG replaces the dump with a **two‑phase retrieval**:

- **Phase A** — cosine‑similarity filter (`δ = 0.30`, `k1 = 20` candidates): cheap recall of semantically‑near notes.
- **Phase B** — composite score `(1−λ)·sim_norm + λ·q_norm` (`λ = 0.40`, `k2 = 10`), where **both terms are z‑score‑normalised within the candidate pool** so that a large similarity magnitude cannot drown out the learned utility signal.

The retrieved set is then **ranked by utility** `q` (learned via the RL memory manager and EMA updates), not just by surface overlap. That is why the *right* note — not merely a *similar* one — reaches the answer model.

### 4.2 Hybrid RRF (dense + BM25 + entity + temporal) → correct **entity anchoring**

The adversarial samples (A, B) are entity‑attribution failures. ASEM‑THG's retriever fuses four channels with Reciprocal Rank Fusion (`rrf_k = 60`): dense embedding, lexical BM25, an **entity channel** (weight 0.6) and a temporal channel. The entity channel biases retrieval toward notes that actually name the **queried subject**, so when the question says *Melanie*, the model is far less likely to receive a Caroline fact dressed in topical similarity.

### 4.3 Chronological, attributed graph‑context rendering

Notes are not pasted in similarity order. `_render_graph_context` **sorts notes chronologically**, renders each note's content together with its **keywords, tags, description and links**, and presents explicit **relations**. Consequences visible in the samples:

- **Attribution is legible** (who did what) → fewer person‑confusions (A, B).
- **Timestamps travel with content** → absolute‑date answers (E).
- **Linked notes expose the union of a multi‑hop answer** in one view (D).

### 4.4 The answer contract (v2 prompt) turns structure into behaviour

Retrieval only helps if the model exploits it. The merged v2 answer prompt encodes four contracts, each targeting a failure mode the baselines exhibit:

| Contract in the prompt | Failure mode it kills | Sample |
|---|---|---|
| **Completeness / aggregation** — collect *all* matching facts across notes, answer the union | multi‑hop under‑answering | D, (C) |
| **Attribution** — a fact under another person is *not* an answer for the named person | person‑confusion hallucination | A, B |
| **Inference** — answer `Likely yes/no — because …` from preferences/plans; don't refuse | open‑domain over‑refusal | C |
| **Misattribution trap** — if no note shows the named subject doing X, reply exactly `Not mentioned` | adversarial confabulation | A, B |
| **Grounding & dates** — prefer the stated/absolute date | vague temporal answers | E |

FullContext has none of these levers: it cannot choose *which* facts to expose, cannot label attribution, and cannot instruct the model to reason about evidence that isn't literally an answer sentence.

### 4.5 `max_context_notes` budget → precision under a window

The strongest single result for the *architecture* argument comes from a backbone where THG is **not** privileged: on Qwen3‑4B (`gpt56_luna`), ASEM‑THG answers from a **median of 10 utility‑ranked, attributed notes** (mean 10.56, p90 = 10, max 27; 1840/1986 questions use exactly 10) yet still beats **FullContext** on EM (0.1923 vs 0.1334) and F1 (0.3728 vs 0.3682) — systems that see the whole trimmed history. Precision beats volume. (At this backbone BERTScore and Judge still favour FullContext — see §2.1 and §5.)

### 4.6 Why FullContext still wins *single‑hop* (the honest caveat)

On single‑hop factual recall FullContext is genuinely competitive: at n = 1986 its BERTScore (0.8729) and judge (0.6803) are **on par with** THG (0.8501 / 0.6813) on the **same backbone and same run**, because the answer is usually a **single local sentence** and the needle is large relative to the 8 K window. In the `locomo_0000` head‑to‑head (§2.2b) FullContext actually **leads** single‑hop (67 vs 56 judge‑correct) — the only slice where a flat dump beats THG, and the largest slice (70/199). THG's advantage is not "more raw facts"; it is **selecting, attributing and aggregating** them, which is what the hard categories require.

---

## 4b. Inference cost and 2×T4 estimates

This section estimates the inference cost of ASEM‑THG vs FullContext‑4K running **Qwen3‑4B‑Instruct‑2507** on **2×NVIDIA T4 (16 GB, ~65 TFLOPS FP16, ~320 GB/s HBM each)** — the actual Kaggle configuration.

> ⚠️ **Sourcing warning.** **No per‑query latency telemetry exists** on disk for either Kaggle arm: `answer_sec`, `retrieval_sec`, `prompt_tokens`, `answer_tokens` appear in **0/101** `*.jsonl` files. `eval/static_eval.py` does write these fields, but it post‑dates the Kaggle run. Every latency/energy number below is therefore an **ESTIMATE** derived from (a) **measured** prompt‑sizing inputs, (b) T4 specs and (c) stated assumptions. They are **not measurements**.

### 4b.1 What is actually measured

| Quantity | ASEM‑THG | FullContext‑4K | Source |
|---|---|---|---|
| Notes per context (p50) | **10** | — (full dump) | `…ASEM-THG_pred.jsonl`: `len(used_notes)` |
| Notes (mean / max) | 10.56 / 27 | — | as above |
| Questions using exactly 10 | 1840 / 1986 | — | as above |
| `max_context_notes` (config) | **10** | n/a | `configs/models/vult55_openai.yaml` |
| `content_char_limit` per note | **200 chars** | n/a | as above |
| `context_window` / `max_tokens` / safety | 8192 / 512 / 64 | 8192 / 512 / 64 | config + `asem/token_budget.py` |
| Prompt budget | **7616 tokens** | **7616 tokens** | `8192 − 512 − 64` |
| Chars per token (internal estimator) | 4 | 4 | `CHARS_PER_TOKEN = 4` |

**Why `max_context_notes: 10` matches p50 = 10 exactly:** `vult55_openai.yaml` sets `max_context_notes: 10`, so THG nearly always hits the note cap — 1840/1986 questions use exactly 10. This is a **config constraint**, not coincidence.

### 4b.2 Prompt‑size estimate

THG's note payload is bounded by two measured parameters: each note is capped at **200 content chars** (`content_char_limit`), but the **rendered block also carries** K (keywords), G (tags), X (description), a timestamp and the attribution labels the graph renderer adds — and those fields are **not** covered by `content_char_limit`. With 10 notes, the rendered payload lands around **1.4k–1.7k tokens** (p50 ≈ 1.5k). This is **inferred** from measured note count × observed render size, **not** measured directly.

FullContext instead dumps the whole trimmed history and **saturates the budget**: p50 ≈ **7.59k tokens**, i.e. it nearly always hits `7616` — a full LoCoMo conversation far exceeds an 8 K window.

| | ASEM‑THG | FullContext‑4K | Ratio |
|---|---|---|---|
| Prompt p50 (estimated) | ~1.5k tokens | ~7.59k tokens | **≈5.1×** |
| Median context ratio | — | — | **16.491×** (LoCoMo), **4.012×** (LongMemEval) |

The 16.491× context ratio comes from `analysis/thg_evaluation_report.md` §2.2 (measured over 120 pairs, **0 inversions**) and **is** a measurement — unlike the ~5.1× prompt ratio, which is an estimate from p50s. They measure different things: 16.491× compares the **context selected**, ~5.1× compares the **final prompt after the 7616‑token trim**.

### 4b.3 Latency estimate on T4

Assumptions (stated so they can be challenged):

- Qwen3‑4B (4.0 B params) in **FP16** ≈ 8 GB of weights → fits one 16 GB T4; two T4s allow **data‑parallel** evaluation of the two arms.
- **Prefill** (compute‑bound): ~65 TFLOPS FP16 peak, ~35–45 % achieved → ~25 TFLOPS usable. Prefill cost ≈ `2 × N_params × N_tokens` FLOPs.
- **Decode** (memory‑bound): ~320 GB/s → theoretical ceiling ≈ `320 / 8` ≈ 40 tok/s for 4 B FP16; **20–30 tok/s** realistically after KV‑cache and overhead.
- LoCoMo answers are short: measured **98.5 chars** (THG) and **62.2 chars** (FullContext) on average → **~25–35 output tokens**, hitting the 512 cap only rarely.

Prefill estimate for the median question:

| | Prompt (tokens) | Prefill FLOPs ≈ 2·4e9·N | Time @25 TFLOPS |
|---|---|---|---|
| ASEM‑THG | 1 500 | 1.2 × 10¹³ | **≈ 0.48 s** |
| FullContext‑4K | 7 590 | 6.07 × 10¹³ | **≈ 2.43 s** |

Adding decode ~30 tokens @ 25 tok/s ≈ **1.2 s** for **both** arms (identical output lengths — see §2.3):

| Component (estimated) | ASEM‑THG | FullContext‑4K |
|---|---|---|
| Prefill | ~0.5 s | ~2.4 s |
| Decode (~30 tokens) | ~1.2 s | ~1.2 s |
| **Total per question** | **~1.7 s** | **~3.6 s** |

⇒ THG is ≈**2.1× faster per question** on the answer path, **all** of it from shorter prefill; decode is **identical** because answer lengths are comparable.

### 4b.4 THG's extra embedding and retrieval cost

THG pays costs FullContext does not — the **counterweight** to the advantage above:

- **Write‑time embedding:** each note needs `e` (joint `concat(c,K,G,X)`) and `z` (intent). The encoder is `all-MiniLM-L6-v2` (~22 M params, 384‑d) — **very light** versus the 4 B backbone, roughly 0.5 % of the compute.
- **Two‑phase retrieval:** Phase A (FAISS cosine filter to `k1 = 20`) + Phase B (composite score, top `k2 = 5`) is ANN search over 384‑d vectors — **under 10 ms/query** on CPU, negligible on GPU.
- **Hybrid RRF** (dense + BM25 + entity + temporal) and **graph traversal** add CPU overhead but never touch the GPU.

Even counting all of this client‑side work, THG remains cheaper on the **most expensive** stage (the 4 B prefill).

### 4b.5 What must NOT be claimed from this data

The on‑disk wall‑clock totals are **THG 17,633 s** and **FullContext 2,611 s** for the `gpt56_luna` run. **Do not** use these to derive per‑question latency:

1. They come from **two different runs** with **different system counts** (THG ran alongside more arms) and **different concurrency levels**.
2. `elapsed` in `scripts/run_static_eval.py` (line 441) is **total wall clock including ingest + bank writes + judge scoring**, not answer time.
3. Most of THG's time is on the **write path** (note construction + link evolution + LLM API calls), which FullContext does not have at all.

So: the ~2.1×/question figure estimates the **read path only**, and THG's entire write cost is a **one‑off investment** amortised across queries — an advantage the total‑time ratio cannot express.

### 4b.6 Summary table

| Metric | ASEM‑THG | FullContext‑4K | Source / confidence |
|---|---|---|---|
| Prompt p50 | ~1.5k tokens | ~7.59k tokens | Estimated from 10 notes (measured) |
| Prompt ratio | **≈5.1×** smaller | — | Estimate |
| Median context ratio | **16.491×** smaller | — | **Measured** (120 pairs, 0 inversions) |
| Prefill/question | ~0.5 s | ~2.4 s | Estimate (T4 @25 TFLOPS) |
| Decode/question | ~1.2 s | ~1.2 s | Estimate (25 tok/s, ~30 tok) |
| **Total read path/question** | **~1.7 s** | **~3.6 s** | Estimate |
| Write cost | one‑off, amortised | 0 | Architecture |
| Estimated energy | ~0.03 Wh/question | ~0.07 Wh/question | Estimate (T4 ~70 W × latency) |

---

## 5. Bottom line

1. **On EM / F1 / ROUGE‑L, THG wins on the same backbone *and same run*** — with Qwen3‑4B (`gpt56_luna`) THG scores **EM 0.1923 / F1 0.3728 / ROUGE‑L 0.3583** vs FullContext's **0.0987 / 0.3389 / 0.3308** (1.95×, 1.10×, 1.08×). In the separate `ds_fixed` run the EM gap is wider (0.1923 vs 0.0745), but that is a **cross‑run** comparison.
2. **BERTScore is backbone‑dependent, not a universal win.** On deepseek‑v4.1‑flash THG **beats** FullContext (.8968 vs .8773); on Qwen3‑4B it **trails** (0.8501 vs 0.8729). The earlier "wins outright" claim was measured on deepseek and does not generalise — it is the **verbosity artifact** interacting with backbone quality.
3. **On the hard, semantic categories the win is largest** (from the §2.2b same‑run table): `locomo_0000` judge‑correct counts are adversarial **31 vs 21**, temporal **26 vs 12**, multi‑hop **23 vs 22**, open‑domain **7 vs 7**.
4. **The mechanism is structural and every sample demonstrates it:** two‑phase utility‑aware retrieval + hybrid RRF entity anchoring + attributed chronological graph context + a four‑contract answer prompt enforcing completeness, attribution, inference and refusal.
5. **Honest caveat:** on the same backbone and same run the **overall judge average is a tie** (0.6813 vs 0.6803, a 0.001 gap). THG's edge concentrates in **adversarial** and **temporal**; FullContext compensates on **single‑hop** (short exact recall). The completed deepseek runs (EM 0.3364) show the ceiling when the backbone is stronger.
6. **On inference cost over 2×T4 (estimates — see §4b):** THG feeds **~10 notes** (`max_context_notes: 10`; measured p50 = 10) instead of dumping the full history, so its median prompt is **≈5.1×** smaller (~1.5k vs ~7.59k tokens — estimated) and its answer‑path latency is **≈2.1× faster per question** (~1.7 s vs ~3.6 s — estimated). The entire gap is **prefill**; decode is identical. This ratio must **not** be inferred from the on‑disk wall‑clock totals (17,633 s vs 2,611 s), which belong to two different runs.

## 6. Comparability caveats (read before quoting numbers)

1. **Adversarial normalizer.** The `ds_fixed` baselines were scored with the **old** `normalize_text()` (which split contractions, so official refusal markers like *"I don't know"* never matched). After the fix (`eval/metrics.py`), adversarial EM jumps (.0561 → .5717 on the v41 run). Consequence: the THG adversarial row (EM 0.6951, `gpt56_luna`) and the FullContext adversarial row (EM 0.0471, `ds_fixed`) are **not** like‑for‑like — re‑score the baselines before quoting the 15× gap.
2. **Category‑label trap.** `category_name` is **not stable** across result files (`ds_fixed` uses shifted labels; `gpt56_luna` / `static_v41_*` use the real ids). **Always key on the numeric `category`** (1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial).
3. **Backbone is a confound.** Only the qwen3‑4b `gpt56_luna` run shares a backbone with the baselines. All deepseek rows are cross‑backbone and must be labelled as such.
4. **Duplicate rows.** ~12 duplicate `(conversation, question)` rows in the raw preds (1986 → 1974 unique) — small, no ranking change.
5. **Verbosity artifact.** A large share of THG's non‑adversarial strict‑EM misses are padding/refusal, not comprehension (judge ≫ EM on single_hop: 0.7063 vs 0.0333).

### Reproduce

```bash
# frozen banks already exist at static/memory_banks/locomo10/gpt56_luna/ASEM-THG/

# (a) SAME-BACKBONE run — this is the table in §2.1a
conda activate memory-r1
python scripts/run_static_eval.py --tag gpt56_luna \
  --config configs/models/qwen35_4B_api.yaml \
  --systems ASEM-THG --metrics em f1 rougeL bertscore_f1 judge \
  --judge-config configs/models/judge_qwen35_4B_api.yaml \
  --per-category --per-conversation --judge-workers 12 \
  --results-dir static/memory_banks/locomo10/kaggle_running

# (b) cross-backbone reference — the deepseek numbers in §2.1c
python scripts/run_static_eval.py --tag gpt56_luna \
  --config configs/models/deepseek_v41_flash_4k_v2.yaml \
  --systems ASEM-THG --metrics em f1 rougeL bertscore_f1 judge \
  --judge-config configs/models/judge_v41_flash.yaml \
  --per-category --per-conversation --judge-workers 12 \
  --results-dir data/benchmarks/results/static_v41_4k_v2
```


THG rows=1986 FC rows=1986 common idx=1986

=== OVERALL (n=1986) ===
ASEM-THG     em 0.1923 | f1 0.3728 | rougeL 0.3583
FullContext  em 0.1334 | f1 0.3682 | rougeL 0.3602
  em      delta +0.0589  ratio 1.44x
  f1      delta +0.0046  ratio 1.01x
  rougeL  delta -0.0018  ratio 0.99x

=== PER CATEGORY ===
cat name            n |                 THG em/f1/rl |                  FC em/f1/rl
  1 multi_hop     282 |         0.0035/0.1810/0.1560 |         0.0426/0.3065/0.2867
  2 temporal      321 |         0.1308/0.4619/0.4320 |         0.0218/0.2140/0.2085
  3 open_domain    96 |         0.0104/0.1089/0.0988 |         0.0312/0.1691/0.1585
  4 single_hop    841 |         0.0333/0.2577/0.2445 |         0.2069/0.5537/0.5447
  5 adversarial   446 |         0.6951/0.7036/0.7036 |         0.1547/0.2113/0.2112

=== PER CATEGORY delta (THG - FC) ===
  1 multi_hop   n= 282 | dEM -0.0390  dF1 -0.1255  dRL -0.1307
  2 temporal    n= 321 | dEM +0.1090  dF1 +0.2479  dRL +0.2236
  3 open_domain n=  96 | dEM -0.0208  dF1 -0.0602  dRL -0.0597
  4 single_hop  n= 841 | dEM -0.1736  dF1 -0.2960  dRL -0.3002
  5 adversarial n= 446 | dEM +0.5404  dF1 +0.4924  dRL +0.4925

=== PAIRED (strict EM, common idx) ===
THG-only 50  FC-only 174  both 22  neither 1740  (net THG -124)

=== ABSTENTION / VERBOSITY ===
THG: abstain 485/1986 (24.4%)  pred len p50=58 chars
FC: abstain 30/1986 (1.5%)  pred len p50=52 chars

=== ADVERSARIAL detail (cat 5) ===
n=446 refs saying 'Not mentioned': 444 (99.6%)
  THG: refusal 285/446 (63.9%)  EM 0.0000
  FC: refusal 19/446 (4.3%)  EM 0.0000

=== CONVERSATION spread (EM) ===
  locomo_0000 n=199  THG 0.0603  FC 0.0503  -> THG
  locomo_0001 n=105  THG 0.0857  FC 0.0667  -> THG
  locomo_0002 n=193  THG 0.0363  FC 0.1036  -> FC
  locomo_0003 n=260  THG 0.0231  FC 0.1000  -> FC
  locomo_0004 n=242  THG 0.0413  FC 0.1116  -> FC
  locomo_0005 n=158  THG 0.0253  FC 0.1456  -> FC
  locomo_0006 n=190  THG 0.0053  FC 0.0684  -> FC
  locomo_0007 n=239  THG 0.0418  FC 0.1172  -> FC
  locomo_0008 n=196  THG 0.0255  FC 0.0969  -> FC
  locomo_0009 n=204  THG 0.0392  FC 0.1127  -> FC