# ASEM-THG graph-build report — LongMemEval + LoCoMo10

**Owner:** WS2 (`bank-runner`), shared task **task-3**
**Status:** LoCoMo10 complete (10/10). LongMemEval 500-question build in flight.
**Scope:** graph **construction** and **measurement** only. No file under `asem/`,
`eval/` or `scripts/` was modified by this workstream.

Every number below was **printed by a tool in `analysis/`**, not derived by
arithmetic on another number. Where two defensible counting rules exist, both are
given with the rule named, because they differ by construction and an unlabelled
number is not usable.

---

## 0. TL;DR — the five results that matter

1. **The entity channel is a hub, and it is nearly information-free.** Corpus-wide
   (10 LoCoMo conversations, 4,579 notes) **94.7% of `same-entity` edges connect
   two notes whose shared entities are ONLY the conversation's speaker names**
   (`hub_only_edge_fraction = 0.9474`). Of 22,565 same-entity records, 21,378 are
   hub-only and **1,187 are discriminative** — a **19:1 noise-to-signal ratio**.2. **The hub is a *class* of tokens, not one token.** On LongMemEval the hub is the
   generic role word `user` (df 67%). On LoCoMo it is the speaker names `caroline`
   (65.2%) and `melanie` (56.4%). **Both corpora land at ~65% median entity
   fan-out by different routes.**
3. **Capping cannot fix it — measured, not argued.** 46,084 implied pairs are
   compressed to 1,792 stored edges (a **25.7x** reduction) and the hub share moves
   from **99.94% to 99.27%** — removing 96.1% of pair volume changes the hub share
   by **0.67 percentage points**. Only re-weighting can work.
4. **`superseded_by` is semantically wrong, actively harmful, and universally
   exposed.** 22 of 166 pairs read in full; **22/22 are wrong**. The failure is not
   random: the mechanism repeatedly marks *still-true* facts obsolete and prefers
   the thinner statement. It is boosted at the **highest** relation weight (1.3×),
   and **166/166 (100%) of pairs have an endpoint reaching the retrieved context** —
   so the wrong signal is not merely present, it is promoted.
5. **Two further defect classes beyond the wrong pairs.** 18/166 pairs have
   direction-**inverted** validity intervals and 11/166 are **zero-width**
   (`valid_from == valid_to`) — **29/166 (17.5%) malformed**.

---

## 1. Provenance — what code built what

### 1.1 LoCoMo `thg_v2` — the **“mixed-window”** bank

`thg_v2` was built by a **single process started at 13:18:36**, while `graph-dev`
was concurrently editing the graph code. The artifact facts:

| | hyper_graph.py | note.py | single_pass_ingest.py | HEAD |
|---|---|---|---|---|
| **at build start** 13:18:36 | `A8335D1A4D5472A2…` | `3AE8D725E93A5E5E…` | `45A6829ECCF12C27…` | `593979c1` |
| **file mtime** | 13:51:22 | 13:19:39 | 13:19:47 | — |
| **at build end** 13:56:12 | `01982784CA9D5625…` | `D12480D238413A40…` | `086CB40206D2D4DB…` | `64f3d24a` |
| **at build end** 13:56:12 | `01982784CA9D5625B849CD97AFAA7826966B3A86A9F2473203D0FD150DC80213` | `D12480D238413A401145D493BA898CB978AE08EAA2DB1BD11A92483F66061132` | `086CB40206D2D4DB62B658992581DC9F1C00A71166E427B9F24EE8C8F39A04DE` | `64f3d24a99e6fb5dae38b00516c82acb0c713197` |

Build window **13:18:36 → 13:56:12** (2,216.85 s). `note.py` and
`single_pass_ingest.py` were rewritten ~21 s into the build (during
`locomo_0000`); `hyper_graph.py` was rewritten at 13:51:22, **during
`locomo_0008`**. So conversations 0000–0007 and 0008–0009 were ingested under
different on-disk code.

> **UNVERIFIED:** whether the edits were live *inside the running process*.
> Python had already imported these modules at process start, so it most likely
> used the start-time version — but that is not observable from the artifacts. The
> hashes and mtimes ARE the artifact; the in-memory import state is not. **This
> report does not assert either way, and `thg_v2` must not be described as a clean
> pre-fix baseline.** A clean post-fix re-ingest into tag `thg_v2_postfix` was
> commissioned by the Lead for the before/after comparison.

Also note `graph-dev`'s landed interval logic **differs** from what built this
bank (the Lead verified `_is_single_valued_predicate` on 16/16 cases and found a
remaining `_predicate_key("is happy") == "is"` collapse). Any before/after on
supersession therefore requires `thg_v2_postfix`, not a re-read of `thg_v2`.

### 1.2 LongMemEval `gb_full` — pre-fix, same caveat

Started 13:18:17, **never restarted**, so its process imported the pre-fix modules.
The identical caveat applies (import state unverifiable). It is the **aggregate
pre-fix baseline**; per-question graph detail comes from the post-fix run instead.

### 1.3 Manifest facts for `thg_v2`

```
model           chib/deepseek-v4.1-flash        (LLM: .env API)
embedder        sentence-transformers/all-MiniLM-L6-v2   (local)
input_sha256    79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
config_sha256   b8f688320dd8a1337e7864e759c1ccaf8eda3a2b9030b4f9a12c0e951648b79d
elapsed_sec     2216.85        (wall clock 2262.13 s incl. startup)
run             built 10, skipped 0, errors 0, notes 4579, discarded_partial 0
created_at      2026-10-03T13:19:16   updated_at 2026-10-03T13:56:12
```

---

## 2. Counting rules (read this before comparing any two numbers)

Three different quantities are all legitimately called "edges", and they differ by
construction. Every table below names which one it uses.

| Rule | Definition | locomo_0000 |
|---|---|---|
| **stored link records** | every `LinkRecord` written into a note's `L`; relations are written on **both** endpoints, and touched peers are re-persisted, so a pair may appear **more than twice** | 6,252 |
| **unique undirected pairs** | each unordered `{i,j}` counted once | **2,002** |
| **implied pairs — UNION** | unordered pairs sharing ≥1 entity, deduped: `46,084` | 46,084 |
| **implied pairs — PER-ENTITY** | `Σ_e C(df_e, 2)` — counts a pair once **per shared entity** | 49,346 |

The UNION and PER-ENTITY counts differ because a pair sharing `caroline` **and**
`melanie` counts twice under PER-ENTITY and once under UNION. The Lead's
intermediate figure of 92,114 for locomo_0000 was a double-count bug in a
throwaway probe and is **superseded** by 46,084 (UNION) / 49,346 (PER-ENTITY);
both were reproduced independently by two readers.

**Recommendation:** quote **unique undirected pairs** for any comparison against
the stored-graph size, and **UNION** for "implied pairs", because those two are
counts of the same kind of object (a pair).

---

## 3. LongMemEval (oracle)

### 3.1 Part-1 smoke (2 questions, tag `gb_smoke`)

Command per the task description; exit 0. **Wall clock 77.15 s for the whole
runner** (2 systems × 2 questions). 3 session ingests, 12 turns each, 10+14+9 = 33
facts extracted across the two questions.

| # | Requested number | Measured |
|---|---|---|
| 1 | notes in `data/benchmarks/longmemeval_banks/asem_thg.sqlite` | **27** — see 3.2, this is the LAST question only |
| 2 | `L` relation distribution | `same-entity` 174, `temporal` 48, `semantic` 15, **`superseded_by` 0** = **237 records / 123 unique undirected pairs** |
| 3 | notes with non-null `session_id` | **27/27**, ALL ordinals: `s1`×14, `s2`×10, `s3`×3. **No `answer_*` haystack id appears anywhere.** |
| 4 | unparseable session dates | **0** (`audit_temporal_grounding.py --corpus longmemeval`: 948/948 ok, bad 0; s_cleaned 24,367/24,367 ok) |
| 5 | throughput | 27 notes / 77.15 s ≈ **0.35 notes/s** end-to-end (includes 2 answer calls + embedder load) |

There is **no `split` relation** in this schema — the four relations are
`same-entity` / `temporal` / `semantic` / `superseded_by`.

**On `session_id` (the Lead's ADDITION 2) — CONFIRMED.** `SELECT DISTINCT
session_id` returns exactly `s1..sN`. `eval/phase_runner.py:114 sessions_to_fast`
assigns `session_id: f"s{s.num}"` and `eval/longmemeval.py:124 build_sessions`
numbers 1-based, so the bank stores **ordinals, never haystack ids**. The mapping
back is positional (`eval/longmemeval.py:282-300 normalize_session_id`), which is
correct only because the oracle haystack order happens to equal session order.
**The bank cannot cite a real haystack session id on its own.**

### 3.2 ⚠️ The LongMemEval bank is per-question scratch — not a dataset graph

`ASEMTHGSystem.reset()` (`eval/systems.py:698-700`) calls
`pipeline.memory_bank.clear()`, and the runner calls `reset()` at
`scripts/run_longmemeval_eval.py:440` before every question. Consequence:

- `data/benchmarks/longmemeval_banks/asem_thg.sqlite` **only ever holds the last
  question's graph.** The "27 notes" in 3.1 is question #2, not the corpus.
- Both smoke questions were ingested **from scratch** (3 then 2 session ingests —
  one per haystack session, matching idx0=3 sessions, idx1=2 sessions).
- **LongMemEval is 500 independent small graphs, not one 500-question graph.**
  Any per-question notes/edges/fan-out must be sampled *during* the run.

**Retraction of an intermediate claim.** An earlier reading of the smoke log
(5 ingest lines for 2 questions) was reported as *double ingestion* (2 passes per
question). **That was wrong and is retracted.** A direct non-invasive experiment
(`analysis/diagnose_double_ingest.py`, scratch bank, same call order as the
runner) shows the bank is untouched by `answer()`:

```
T0 after reset()                          n=0
T1 after explicit _ingest_question        n=34  null_ts=0  with_sid=34  sids=['s1','s2','s3']
T2 after answer() (the lazy path)         n=34  null_ts=0  with_sid=34  sids=['s1','s2','s3']  IDENTICAL
ratio 34/34 = 1.00
```

5 ingest lines = 3 sessions (idx 0) + 2 sessions (idx 1). `_ingested` is set by
the explicit path, so the guard at `eval/systems.py:690-691` correctly does
nothing. There is no cost bug and no undated duplicate layer.

### 3.3 Graph shape (smoke bank, 27 notes)

| metric | value |
|---|---|
| entity fan-out per note (notes sharing ≥1 entity) | **mean 11.33, median 17, p90 17, max 17** (of 26 possible) |
| distinct entities | **5** |
| notes with empty entity set | **8 / 27 = 29.6%** |
| implied same-entity pairs (PER-ENTITY) | 160 |
| notes rendered "date unknown" | **0** (timestamp_iso non-null on 27/27) |
| `superseded_by` | 0 **in the final bank** — but see 5.6: the mechanism demonstrably fires |

**The hub, precisely:** `user` appears on **18 of 27** notes (df 0.667). 18 notes
sharing one entity implies C(18,2) = 153 pairs against 160 implied — so **~92% of
the entity channel is the literal token `user`**. Top-10 df:

| entity | df | df/N |
|---|---|---|
| `user` | 18 | 0.667 |
| `tableau` | 3 | 0.111 |
| `power bi` | 3 | 0.111 |
| `shell` | 2 | 0.074 |
| `local community center` | 1 | 0.037 |
| *(only 5 distinct entities exist)* | | |

`max_entity_links=5` (`asem/hyper_graph.py:69,77`) caps **out-degree**, which is
why stored same-entity edges (174) < 2×160; it does not prevent the hub, because
the cap limits how many peers a note *links to*, not how many it *shares entities
with*.

### 3.4 Part-2 full run (`gb_full`) — status and coverage

Started 13:18:17, **no `--limit`, no `--clean`** (resumable). Progress at the time
of writing: **97/500**, rate **0.40 min/question → projected ≈3.3 h total**.
No crash, no traceback, process healthy.

> **`coverage`: `{n_measured: 0, n_total: 500, note: "per-question in-process
> capture was not active for this process — the run started before the hook
> existed. gb_full therefore yields AGGREGATE metrics only; per-question graph
> detail is null, NOT reconstructed."}`**
>
> Per the Lead's approved **(i)+partial** decision, the per-question graph index
> will be produced from the **post-fix** run on a new tag, which is the more useful
> artifact because the table's purpose is the before/after comparison. Nothing is
> backfilled or interpolated.

---

## 4. LoCoMo10 (`thg_v2`) — COMPLETE, 10/10

### 4.1 Per-conversation table

| conversation | sessions | turns | notes | link records | **unique pairs** | rec/note | **pairs/note** | fan-out med | fan-out p90 | disc. | sup. pairs | elapsed s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| locomo_0000 | 19 | 419 | 365 | 6,252 | 2,002 | 17.13 | 5.48 | 237 | 362 | 13 | 5 | 162.6 |
| locomo_0001 | 19 | 369 | 347 | 6,290 | 1,961 | 18.13 | 5.65 | 274 | 346 | 15 | 36 | 165.6 |
| locomo_0002 | 32 | 663 | 493 | 7,544 | 2,717 | 15.30 | 5.51 | 353 | 489 | 111 | 18 | 235.8 |
| locomo_0003 | 29 | 629 | 492 | 8,424 | 2,713 | 17.12 | 5.51 | 335 | 491 | 37 | 10 | 231.5 |
| locomo_0004 | 29 | 680 | 482 | 6,890 | 2,620 | 14.29 | 5.44 | 322 | 480 | 99 | 12 | 233.5 |
| locomo_0005 | 28 | 675 | 484 | 8,576 | 2,689 | 17.72 | 5.56 | 328 | 480 | 99 → **166** | 10 | 235.2 |
| locomo_0006 | 31 | 689 | 513 | 7,346 | 2,772 | 14.32 | 5.40 | 327 | 507 | 136 | 12 | 252.9 |
| locomo_0007 | 30 | 681 | 511 | 8,118 | 2,810 | 15.89 | 5.50 | 328 | 500 | 187 | 24 | 253.8 |
| locomo_0008 | 25 | 509 | 414 | 6,640 | 2,274 | 16.04 | 5.49 | 308 | 413 | 35 | 17 | 215.6 |
| locomo_0009 | 30 | 568 | 478 | 7,782 | 2,647 | 16.28 | 5.54 | 337 | 477 | 388 | 22 | 230.3 |
| **TOTAL** | **272** | **5,882** | **4,579** | **73,862** | **25,205** | — | — | — | — | **1,187** | **166** | **2,216.7** |

> **⚠️ `locomo_0005` discriminative = 166.** An earlier 6-conversation table
> reported 0005 as partial (50 notes, 14 discriminative); the completed value is
> **166 at 484 notes**. Use the value in this table. No partial conversation is
> included in any aggregate here.

**`pairs/note` is stable at 5.40–5.65 across all 10 conversations** (347–513
notes) while `records/note` ranges 14.29–18.13. The pair count is the more stable
quantity, and its stability is itself evidence that **the entity channel saturates
regardless of conversation size**.

Per-conversation edge histograms (records):

| conversation | same-entity | semantic | temporal | superseded_by | distinct entities | empty-entity notes | date unknown |
|---|---|---|---|---|---|---|---|
| locomo_0000 | 3,584 | 1,966 | 692 | 10 | 28 | 0 | 0 |
| locomo_0001 | 3,416 | 2,146 | 656 | 72 | 7 | 0 | 0 |
| locomo_0002 | 4,858 | 1,728 | 922 | 36 | 32 | 0 | 0 |
| locomo_0003 | 4,864 | 2,614 | 926 | 20 | 18 | 0 | 0 |
| locomo_0004 | 4,760 | 1,200 | 906 | 24 | 49 | 0 | 0 |
| locomo_0005 | 4,768 | 2,876 | 912 | 20 | 16 | 1 | 0 |
| locomo_0006 | 5,046 | 1,312 | 964 | 24 | 37 | 0 | 0 |
| locomo_0007 | 5,020 | 2,088 | 962 | 48 | 33 | 0 | 0 |
| locomo_0008 | 4,082 | 1,746 | 778 | 34 | 16 | 0 | 0 |
| locomo_0009 | 4,732 | 2,110 | 896 | 44 | 25 | 0 | 0 |
| **TOTAL** | **45,130** | **19,786** | **8,614** | **332** | — | **1** | **0** |

**Notes rendered "date unknown": 0** across all 4,579 notes — the W1 date-parsing
fix holds on LoCoMo end-to-end. `same-entity` is 57.3% of records; `superseded_by`
332 records = **166 pairs** (each stored on both endpoints).

### 4.2 The hub: `hub_only_edge_fraction`

**Definition.** Of the stored `same-entity` edges, the fraction whose two notes'
entity intersection contains **only hub entities**. Hub = entities with
df ≥ 50% of notes, plus the top-2 by df. On LoCoMo that is `caroline` + `melanie`.
An edge like this carries zero discriminative information: it says only "both notes
are about the conversation's speakers", which is true of most of the bank. Example
row: `shared=['caroline']  A=['caroline']  B=['caroline']`.

| conversation | notes | same-entity edges | hub-only | discriminative | **hub_only fraction** |
|---|---|---|---|---|---|
| locomo_0000 | 365 | 1,792 | 1,779 | **13** | 0.9927 |
| locomo_0001 | 347 | 1,708 | 1,693 | 15 | 0.9912 |
| locomo_0002 | 493 | 2,429 | 2,318 | 111 | 0.9543 |
| locomo_0003 | 492 | 2,432 | 2,395 | 37 | 0.9848 |
| locomo_0004 | 482 | 2,380 | 2,281 | 99 | 0.9584 |
| locomo_0005 | 484 | 2,384 | 2,218 | **166** | 0.9304 |
| locomo_0006 | 513 | 2,523 | 2,387 | 136 | 0.9461 |
| locomo_0007 | 511 | 2,510 | 2,323 | 187 | 0.9255 |
| locomo_0008 | 414 | 2,041 | 2,006 | 35 | 0.9829 |
| locomo_0009 | 478 | 2,366 | 1,978 | **388** | 0.8360 |
| **TOTAL (n=10)** | **4,579** | **22,565** | **21,378** | **1,187** | **0.9474** |

> **Note on n=1 vs n=10.** The **first** measurement of this quantity was
> **0.9927 on `locomo_0000` alone (n=1 conversation)**. The **n=10** figure is
> **0.9474** and **supersedes it**. The n=1 number is retained here only to show
> the sensitivity of the statistic to sample size — it moved 4.6 points. Per
> conversation the range is **0.8360–0.9927**; `locomo_0009` is the clear outlier
> (388 discriminative edges, 0.8360) and is what pulls the corpus figure down.

**Discriminative edges per conversation: 13, 15, 111, 37, 99, 166, 136, 187, 35,
388 — mean 118.7, median ~117, range 13–388 (a ~30× spread).**

> **Correction of a hypothesis.** An earlier proposal was that the discriminative
> count stays **near-constant (~13)** while the bank grows, which would have been a
> strong "fixed tiny signal" claim. **Measurement falsifies it:** the count varies
> ~30× and broadly *grows* with conversation size — the two largest banks
> (0002: 493 notes, 0004: 482) carry 111 and 99, while 0000 (365) carries 13. The
> defensible statement is the **ratio** (94.7% hub-only / 19:1 noise-to-signal),
> not a constant.

**The 19:1 ratio.** 21,378 hub-only : 1,187 discriminative ≈ **18.0:1**, so the
entity channel carries **~5.3% real signal** corpus-wide. (An interim n=1 estimate
of 2.6% / 38:1 is **withdrawn** — it was `locomo_0000` only.)

**Size alone does not degrade the channel.** `locomo_0002` (493 notes) and
`locomo_0004` (482) have *more* discriminative edges **and a lower hub fraction**
than the smaller `locomo_0000` (365) / `locomo_0001` (347). What matters is how
many **distinct one-off entities** a conversation introduces (distinct entities:
28/7/32/18/49/16/37/33/16/25), not how big it is.

### 4.3 The main lever is NOT the cap — measured both denominators

On `locomo_0000`:
- **UNION denominator:** 46,084 implied pairs → 1,792 stored edges = **25.7×**
  reduction; hub share **99.94%** (46,057/46,084) → **99.27%** (1,779/1,792).
- **PER-ENTITY denominator:** 49,346 → 1,792 = **27.5×**; hub share **99.94%** →
  **99.27%**.

**Removing 96.1% of the pair volume changes the hub share by 0.67 percentage
points.** The `max_entity_links` cap therefore provably does not restore
selectivity: it ranked peers by embedding similarity *within* the hub, which
cannot recover discriminative structure that was never in the entity set. **Only
re-weighting (e.g. hub-aware IDF) can fix this**, which is the basis of GB5.

### 4.4 Two-corpus hub comparison

| corpus | hub entity | df | df/N | distinct entities | median fan-out | fan-out basis |
|---|---|---|---|---|---|---|
| LongMemEval (smoke, 27 notes) | **`user`** | 18 | **0.667** | 5 | 17 | of 26 peers = 65% |
| LoCoMo locomo_0000 (365 notes) | **`caroline`** | 238 | **0.652** | 28 | 237 | of 364 peers = 65% |
| LoCoMo locomo_0000 | **`melanie`** | 206 | **0.564** | | | |

**The hub is a *class* of tokens, not a single token:** a generic role noun
(`user`) on LongMemEval and proper speaker names (`caroline`, `melanie`) on
LoCoMo. Two independent readers reproduced `caroline` 238/365 and `melanie`
206/365. **Both corpora land at ~65% median fan-out by different routes** — which
is why the fix must be hub-aware weighting rather than a hardcoded stop-word list
for `user`.

Fan-out detail, `locomo_0000` (365 notes): **mean 252.5, median 237, p90 362, max
363, zero-share 0.0**. Read plainly: the median note shares an entity with **237 of
the other 364 notes (65%)**, and p90 is **362/364 (99.5%)** — for a large fraction
of notes, "notes sharing an entity" is *the entire bank*. Zero-share 0.0 means the
channel never fails to fire; it fires almost always.

---

## 5. `superseded_by` — the versioning mechanism

### 5.1 Corpus counts and split

**166 unique undirected pairs** across the 10 conversations (332 stored records,
written on both endpoints). Split: **29 same-session / 137 cross-session (82.5%
cross-session)**.

| conversation | pairs | same-session | cross-session |
|---|---|---|---|
| locomo_0000 | 5 | 1 | 4 |
| locomo_0001 | 36 | 10 | 26 |
| locomo_0002 | 18 | 2 | 16 |
| locomo_0003 | 10 | 0 | 10 |
| locomo_0004 | 12 | 0 | 12 |
| locomo_0005 | 10 | 1 | 9 |
| locomo_0006 | 12 | 2 | 10 |
| locomo_0007 | 24 | 1 | 23 |
| locomo_0008 | 17 | 5 | 12 |
| locomo_0009 | 22 | 7 | 15 |
| **TOTAL** | **166** | **29** | **137** |

**The plumbing works; the semantics are wrong.** Versioning fires in every
conversation, is cross-session-dominated (82.5%), and the bi-temporal intervals are
being written — so this is not a wiring failure.

### 5.2 Semantic verdict — 22 read, 22 wrong

> **Honesty statement.** **22 of the 166 pairs were read in full by a human/agent.
> 22 of 22 are semantically wrong. The remaining 144 are UNREAD and this report
> does NOT extrapolate to 166/166.** All 144 carry `semantic_verdict: "unread"` in
> the appendix. Automatic labelling was deliberately refused: deciding whether two
> facts are genuine versions of one another requires semantic judgement, and a
> script-produced verdict would be a *reasoned* label masquerading as an
> artifact-read one.

**Failure modes observed, with verbatim examples.**

**(a) Restatement marked as replacement** — the *richer* note is superseded by a
*lossier* one, so correct evidence is deleted:

- locomo_0000 `s1 → s4`:
  - OLDER: *"Melanie reacted to Caroline's necklace, saying it was gorgeous and that such items can mean so much; Melanie's own grandmother gave her a necklace from Sweden."*
  - NEWER: *"Melanie reacted to Caroline's necklace, saying it is gorgeous and that items can mean a lot to us."*
  - The newer note **drops** Sweden/grandmother. Marking the older one obsolete
    destroys the richer evidence.
- locomo_0000 `s8 → s12`: *"Melanie appreciates Caroline's friendship and says it is great to have a supporter."* → *"Melanie said she appreciates her friendship with Caroline and that Caroline has always been there for her."*
  — a standing sentiment **re-confirmed**, not replaced.
- locomo_0001 `s6 → s5`: *"Jon thanked Gina for having his back and toasted to taking on new heights…"* → *"Jon thanked Gina, saying her pep-talk meant a lot…"* — the same gratitude restated.

**(b) Additive facts treated as competing values** (`owns`/`has` on the
single-valued allow-list, `asem/hyper_graph.py:290-298`):

- locomo_0000 `s7 → s13`: *"Melanie has a dog and a cat, which brighten up her day and always make her smile."* → *"Melanie's pets are doing well and she recently got another cat named Bailey."*
  — **both are true simultaneously.** The new fact is additive; the old is not obsolete.
- locomo_0007 `s20 → s10`: *"Jolene uses yoga as a way to escape studying and work stress."* → *"Jolene uses the Pomodoro Technique, working 25 minutes and taking 5-minute breaks, to avoid burnout."*
  — **two different coping techniques, both true.** A textbook false replacement.
- locomo_0009 `s2 → s9`: *"Calvin's new car is a red sports car."* → *"Calvin owns a red car with black rims."* — the same car, more detail.

**(c) Information flow inverted** — the newer note is the *less* informative one:

- locomo_0009 `s20 → s2`: OLDER *"Calvin shared a photo of his red sports car parked in a parking lot, calling it a sign of his hard work and dedication."* → NEWER *"Calvin's new car is a red sports car."*
  — a **time-inverted** supersession: the later mention carries less information
  and is marked as the replacement. Combined with (a), this is **two independent
  instances of the mechanism systematically preferring the thinner statement.**

**(d) Unrelated encouragements paired** — locomo_0001 `s4 → s4`: *"Gina tells Jon he will kill it in the competition…"* → *"Gina encourages Jon to hang in there, saying he will find the perfect dance studio soon."*

**Structural evidence the key is too coarse:** all 5 pairs read in `locomo_0000`
share `melanie` (2 also share `caroline`) — shared entity means shared subject, so
the `(subject, predicate)` slot key collapses every "Melanie …" pair onto one
another. 4 of those 5 come from `has`/`is`/`was`-class predicates, and
`_is_single_valued_predicate` (`asem/hyper_graph.py:314-315`) admits **any**
multi-word predicate starting `is`/`was` — unbounded.

### 5.3 Interval sanity — a THIRD defect class (script-classifiable)

Because edges are stored **symmetrically**, "which note holds the record" is NOT
the direction. **The direction is a property of the intervals:** the OLDER note's
interval must *close* (`valid_to` = newer's date) and the NEWER note's must be
*open* (`valid_to: null`). Classifying all 166 (`analysis/supersession_appendix.py`):

| interval class | count | meaning |
|---|---|---|
| `older_closed_newer_open` | **137** | the intended shape |
| **`inverted`** | **18** | newer note's interval is *closed* — direction bug |
| **`zero_width`** | **11** | `valid_from == valid_to` — degenerate window |
| `missing` | 0 | — |

**29 of 166 (17.5%) have a malformed validity interval.** The `zero_width` cause is
that same-session notes carry **one identical timestamp**, so any within-session
supersession produces a zero-length interval (e.g. locomo_0000 `s13 → s13`, both
`2023-08-23T15:31:00Z`). This is separate from the wrong-pair problem and should be
tracked as its own defect (GB2d): fixing "no `valid_to` ≤ `valid_from`" is
**necessary but not sufficient**, because 18 pairs are inverted rather than
zero-width.

Other script-classifiable columns over the 166: **56 pairs share a head verb, 110 do
not.** Predicates are marked **`derived`** in the appendix — the `(subject,
predicate, object)` triplet is **not persisted on the note** (see
`hyper_graph.hydrate`'s docstring), so no stored triplet exists to quote.

### 5.4 Why this is worse than an absent relation

`enhanced_retriever.relation_weights` gives `superseded_by` a **1.3× boost — the
highest weight in the table.** A wrong supersession is therefore not merely present
in the graph; it is **promoted above correct evidence** in ranking. Where the
mechanism marks a still-true fact obsolete, it converts a benign redundancy into an
active ranking inversion. That combination — no correct result in 22 attempts,
highest weight, destroyable evidence — is the basis for disabling it by default
(GB2e) and keeping it as an explicit ablation. **A fix must reduce this count, not
increase it.**

### 5.5 Latent risk in `superseded_by: 0`

The Part-1 smoke's final bank showed **0** `superseded_by` edges, while a DEBUG log
during that same ingest showed `THG versioning | f723bf62 supersedes 666a603d`.
Both are true: the final bank holds only the last question, and that question
produced no pair. So "0 edges" is **not** evidence the mechanism is dead — it
fires on LongMemEval too. Its LongMemEval rate is **not measured by this report**.

### 5.6 The wrong signal actually reaches the context — exposure 166/166 (100%)

`analysis/supersession_exposure.py` / `_corpus.py` build a query from each
mislabelled pair's own older-fact text, run the real retriever
(`EnhancedHybridRetriever.retrieve(query, M)`, `asem/enhanced_retriever.py:141`)
over the frozen bank, and ask whether either endpoint of the pair is retrieved.

```
banks: 10   pairs total: 166   evaluated ok: 166   errors: 0   exposed: 166
EXPOSURE RATE : 1.0000 (n=166)
per conversation: 5/5, 36/36, 18/18, 10/10, 12/12, 10/10, 12/12, 24/24, 17/17, 22/22
```

**100% on every conversation, with zero errors.** The n=5 single-conversation
result was not an anecdote; it is a corpus property. Every one of the 166
mislabelled pairs has at least one endpoint reaching the retrieved context for a
query built from its own fact text.

**100% exposure, with `superseded_by` at the highest relation weight (1.3×),
means the wrong signal does not merely exist — it reliably enters the context and
enters it promoted.** Together with 0-of-22-read pairs correct, that is the
evidence base for disabling the relation by default (GB2e) and keeping it as an
explicit ablation.

> **Measurement bug caught before reporting.** A first run of this script printed
> `exposure rate 0.0000 (n=5)`. That was **not a result**: `retrieve()` requires
> the `MemoryBank` as a second positional argument and was called with one, so all
> 5 calls raised `TypeError`, the loop recorded the error per row while leaving
> `hit_older`/`hit_newer` as `None`, and **`None` is falsy — a total method failure
> rendered as a clean zero.** It was caught only by printing per-row
> `n_retrieved`/`error` from the JSON rather than trusting the summary line. The
> `0.0000` is **withdrawn, not revised**. The scripts now **refuse to emit a rate
> when `n_errors > 0`** (exit non-zero, write `exposure_rate: null`,
> `refused: true`) — a warning beside a plausible number loses to the number.

### 5.7 Failure rendered as a plausible zero — a systemic pattern

Three independent instances occurred in this workstream and its neighbours. Each
produced a **well-formed, plausible-looking number** from a **total failure**:

| # | failure | rendered as | caught by |
|---|---|---|---|
| 1 | `retrieve()` called with 1 arg → `TypeError` on every call | `exposure rate 0.0000` | printing per-row fields, not the summary |
| 2 | judge model not entitled → `403 FORBIDDEN` on every call | `judge: 0.0, judge_coverage: 0.0` | entitlement preflight script |
| 3 | inherited bank history (`L` read back) mistaken for new output | a well-formed **nonzero** pair count | clearing `L` and re-running |

This is a **different and more dangerous error class** than an arithmetic mistake:
a wrong number is visible on recomputation, whereas these are
**indistinguishable from a legitimate measurement of the thing being measured** —
`0%` exposure was a *totally plausible finding*. The rule this implies: **any
aggregate computed over a loop that can fail must report its denominator AND its
error count, and must refuse to emit a number when errors > 0; and no failure may
ever flow into a summary as a falsy value (`None`/`[]`/`0`), because those are
indistinguishable from a measured zero.**

### 5.8 Scope limits on this section — what `thg_v2` can and cannot testify about

- **GB1 is forward-only.** `graph-dev` verified that `locomo_0000`'s `triplets`
  column is **NULL on every pre-existing row**, so `hydrate()` rebuilds no
  versioning index for it, and a clean-slate replay emits **0** supersession edges
  even with the flag on. The 166 pairs are therefore **output of the OLD build
  only** and are **not evidence about the current code at HEAD**. The 0-of-22-correct
  verdict is a statement about the revision that built `thg_v2`.
- **`hub_only_edge_fraction` is the WRONG acceptance metric for GB5.** GB5 is
  deliberately **ordering-only**: the fixture shows stored same-entity edges
  1792 → 2724 and hub share **99.27% → 99.01%**, i.e. essentially flat by design
  (the count moved only because the inverse index finds peers the old capped O(N)
  scan dropped, not because the cap changed). The real GB5 evidence is **ranking**
  (note `00cb74d3`'s rare-sharing peer `7ab53820` moved #5 @0.6589 → #1 @3.8245,
  displacing 4 hub-only peers). §4.2's 94.7% is therefore **the pre-fix defect
  measurement, NOT a post-fix acceptance target** — a flat post-fix hub share is
  the expected result, not a failed fix.
- **The 166 pairs are inherited history, not a leak from new code.** `graph-dev`
  confirmed loading that bank and re-running `add_note` with
  `enable_supersession=False` leaves exactly 10 links untouched.

### 5.9 What is NOT measured here

- **Full semantic classification of all 166 pairs** (only 22 read; 144 `unread`).
- **The accuracy ablation** (`superseded_by` at 1.3× vs disabled): this needs the
  judge-based evaluation pipeline (task-4). The exposure figure in 5.6 is a proxy,
  not the ablation. (Note: the judge config was itself broken — a `403` outage
  rendered as `judge: 0.0` — and is now fixed; run
  `scripts/check_judge_entitlement.py` before any judge number.)
- **LongMemEval per-question supersession columns**: `gb_full` has no per-question
  capture (see 3.4); they will come from the post-fix hooked run.

---

## 6. Artifacts produced

**Banks**
- `data/benchmarks/longmemeval_banks/asem_thg.sqlite` — the `gb_smoke`/`gb_full` scratch bank (last question only)
- `static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_000N/asem_thg.sqlite` × 10
- `static/memory_banks/locomo10/thg_v2/manifest.json`, `banks.json`

**Results**
- `data/benchmarks/results/longmemeval/gb_smoke/…json`, `…md`
- `data/benchmarks/results/longmemeval/gb_full/preds/…jsonl` (resumable)

**Measurement tools (all in `analysis/`, none modify the pipeline)**
`graph_metrics.py`, `final_tables.py`, `hub_only_edges.py`, `hub_only_corpus.py`,
`supersession_audit.py`, `supersession_corpus.py`, `supersession_appendix.py`,
`reconcile_pair_denominators.py`, `build_per_question_index.py`,
`snapshot_hook.py` + `snapshot_site/sitecustomize.py`,
`diagnose_double_ingest.py`.

**Data artifacts**
`_thg_v2_final_tables.json`, `_hub_only_corpus.json`, `_locomo0000_supersession.json`,
`_locomo0000_metrics.json`, `_locomo0000_hubonly.json`, `_part1_metrics.json`,
`_double_ingest_probe.json`, `supersession_appendix.json` / `.md`.

---

## 7. Failed, retracted, and unverified

| item | status |
|---|---|
| **Double-ingestion claim** (2 passes/question on LongMemEval) | **RETRACTED.** Direct experiment: bank untouched by `answer()`, ratio 1.00. 5 ingest lines = 3+2 haystack sessions. |
| **"Unique edges/note ≈ 8.57"** (locomo_0000) | **RETRACTED** — derived as 6252/2, not measured. Measured: **2,002 unique pairs = 5.48/note**. |
| **"near-constant ~13 discriminative edges"** | **FALSIFIED** by measurement: 13…388, ~30× spread. |
| **n=1 extrapolations** (99.3% hub-only, 2.6% signal, 38:1) | **SUPERSEDED** by n=10: 94.7%, ~5.3%, ~19:1. Retained with their n. |
| **92,114 implied pairs** (locomo_0000) | **NOT REPRODUCED** — my readers give 46,084 (UNION) / 49,346 (PER-ENTITY). Lead confirmed their figure was a double-count. |
| **External snapshot watcher** | **FAILED** — raced `reset()`; 7 of 10 first captures read an empty bank. Index discarded, not reported. Replaced by in-process hook. |
| **In-memory import state during builds** | **UNVERIFIED.** Hashes/mtimes reported; live code version not asserted. |
| **`thg_v2` as a clean pre-fix baseline** | **NOT VALID** — concurrent edits (see §1.1). Labelled **mixed-window**. Use `thg_v2_postfix` for before/after. |
| **Per-question index for `gb_full`** | **coverage 0/500.** Hook absent for that process; not reconstructed. |
| **166/166 semantic verdict** | **NOT CLAIMED.** 22 read, 144 `unread`. |
| **`superseded_by` accuracy ablation** | **NOT RUN** — needs task-4 judge pipeline. |

---

## 8. Appendix A — all 166 supersession pairs

Full table with both fact sentences, session ids/dates, script-classifiable
columns (`same|cross`, `interval_class`, `shared_head_verb`) and
`semantic_verdict: unread`:

- `analysis/supersession_appendix.md` (human-readable)
- `analysis/supersession_appendix.json` (machine-readable, 166 rows)

Counts: 137 cross-session / 29 same-session; 137 `older_closed_newer_open` /
18 `inverted` / 11 `zero_width`; 56 shared head verb / 110 not.
