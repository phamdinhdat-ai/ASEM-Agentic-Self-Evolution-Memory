# ASEM / ASEM-THG — temporal-grounding improvements (W1–W9)

**Status:** code changes complete and verified LLM-free. The end-to-end accuracy
re-run (which needs API access) is **not** done — see [§6](#6-what-is-not-done).

**Scope:** one shared temporal core, applied to **both** LoCoMo10 and LongMemEval.
Every change below is justified by a defect that exists in shared code, not by a
quirk of either dataset.

---

## 1. Root cause (verified, not inferred)

`asem/temporal.py:parse_session_datetime()` could not parse LongMemEval's session
stamp format and **silently fell through to `datetime.now()`**:

```
before:  parse_session_datetime('2023/02/20 (Mon) 12:15')  ->  2026-10-03T03:16:15Z   # today
after:   parse_session_datetime('2023/02/20 (Mon) 12:15')  ->  2023-02-20T12:15:00Z
```

Measured on `longmemeval_oracle.json`: **948 / 948 session dates (100%)** were
unparseable. Because `datetime.strptime` cannot consume the `(Mon)` group wedged
into the middle, and no fallback format accepted the slash-separated date.

Consequences, each confirmed by running the code:

1. `_human_date(note)` returned `"3 October 2026"` for **every** note, so the
   answer prompt read `Memory 1 — recorded on 3 October 2026` for a March-2022
   session — a confident, wrong anchor presented as evidence.
2. `_render_graph_context` sorts chronologically by `note.t`; every note tied on
   "now", so temporal ordering collapsed to input order.
3. `note.timestamp_iso` was a `now()` stamp, poisoning `_human_date`'s fallback.
4. The extractor received `SESSION DATE: 2022/03/02 (Wed) 22:25` — a string it was
   then asked to compute relative dates ("yesterday", "last Saturday") against.

### Why this produced the observed failures

In every failing case the needed dates are stated **verbatim in the evidence
turns**, and the task is pure subtraction:

| question_id | gold | predicted | dates present in the gold turns |
|---|---|---:|---|
| `2c63a862` | 14 days | **445** | turn says `2/15` and `3/1` |
| `08f4fc43` | 30 days | 49 | turn says `January 2nd`, `February 1st` |
| `2a1811e2` | 21 days | 37 | turn says `February 26th`, `January 2nd` |
| `0bb5a684` | 7 days | 3 | turn says `January 10th`, `January 17th` |

`445 days` is the model correctly subtracting a corrupted "today" from a
mis-resolved anchor. Ordering questions survived because a comparison of two
equally-corrupted dates is still correct.

---

## 2. Changes

### W1 — Shared temporal parsing (`asem/temporal.py`, `asem/answer_agent.py`)

* Added a `_parse_longmemeval_style` pre-pass handling `YYYY/MM/DD (Day) HH:MM`,
  the no-time form, single-digit components, and lowercase/spaced weekdays.
  Accepts `/`, `-`, `.` separators and an optional 12-hour suffix.
* **Removed the silent `now()` fallback.** A non-empty input that fails to parse
  now returns a `None` ISO string, which is the signal "this is a placeholder,
  not evidence". Empty input legitimately means "now" and is unchanged.
* `_human_date` honours that contract: an unknown date renders `date unknown`
  rather than publishing today's date. It also consults the central parser, so
  both modules share one contract instead of duplicating format lists.

### W2 — Absolute-date grounding (`asem/single_pass_ingest.py`, `data/prompts/P_temporal_qa.txt`)

* The extractor now receives the **resolved human date**
  (`SESSION DATE: 2 March 2022 (raw stamp: …)`), not the raw machine stamp, so the
  existing "resolve relative time" rule is executable.
* Extraction rule 2 was rewritten around the real failure mode: normalise inline
  shorthand (`3/1` → `1 March 2022`), infer the year from the session date, and —
  critically — **emit one date per fact**. A turn stating two dated events must
  become two notes with one absolute date each, never one sentence containing two
  competing dates. That is what makes subtraction possible later.
* `P_temporal_qa.txt` gained guideline 7: restate both dates, compute the
  difference, answer with number+unit only.

### W3 — Graph structure (`asem/hyper_graph.py`, `asem/enhanced_retriever.py`, `eval/systems.py`)

* **Entity hygiene:** `_entity_keys` now indexes named `entities` only. It
  previously unioned `K` (keywords), so shared *topic* words created
  `same-entity` edges and inflated the entity graph toward a clique. Keywords
  remain available via the BM25 channel.
* **Object-aware versioning:** two indices now exist — an exact
  `(subject, predicate, object)` key for restatements, and a
  `(subject, predicate)` slot key consulted **only** for single-valued predicates
  from a conservative allow-list. So `lives in Boston → lives in Seattle`
  supersedes, while `bought a car` / `bought a house` do not.
* **Personalized PageRank is now live.** `personalized_pagerank()` existed but was
  referenced only by its own unit test — dead code, so THG's defining graph had no
  effect at query time. It is now wired as a query-seeded channel (seeded on the
  Phase A+B winners), blended into Phase D via `ppr_weight` (default `0.15`,
  set `0.0` to reproduce the old ranking exactly). `eval/systems.py` constructs
  **one** `TemporalHyperGraph` shared by the ingestor and the retriever, so PPR
  walks the graph that ingestion actually built.

### W4 — Bi-temporal validity (`asem/note.py`, `asem/hyper_graph.py`, `asem/answer_agent.py`)

Following Zep/Graphiti ([arXiv:2501.13956](https://arxiv.org/abs/2501.13956) §2.2.3),
supersession now records **when a fact was true**, not just that it changed:

* `LinkRecord` gained optional `valid_from` / `valid_to`. Both are omitted from
  `to_dict` when unset, so **legacy notes round-trip byte-identically and frozen
  banks (e.g. the LoCoMo `ds_fixed` banks) load unchanged**.
* On supersession the older fact's interval is *closed* at the newer fact's
  timestamp and the newer fact opens an interval from its own date.
* Rendered as: `Memory 1 — recorded on 1 March 2022 (true until 3 March 2022,
  superseded by Memory 2)`, and the `Connections:` sentence is now **directional**
  (`is replaced by…` vs `replaces…`). Direction comes from the interval, because
  the edge itself is stored symmetrically.

### W5 — Measurement hygiene (`scripts/run_longmemeval_eval.py`, `scripts/audit_temporal_grounding.py`)
* **`turn_recall` was structurally always `0.0`.** The call site hardcoded
  `retrieved_texts = []`, forcing the `else: 0.0` branch. It now resolves the
  answered-from notes into text, and reports `None` ("not measured") when a system
  emits no telemetry.
* New LLM-free audit tool (`--corpus`, `--preds`, `--dump-slice`) that reports
  unparseable-date counts and splits results into **arithmetic** vs **ordering**
  questions, joining the `scores/` file for judge verdicts.
* **`--judge` is a separate flag from `--metrics judge`.** Passing `judge` in
  `--metrics` without `--judge` silently produces `judge = 0.0` with
  `judge_coverage = 0.0` — it looks like a total judge failure. Check
  `judge_coverage`, not `judge`, when a judge score reads zero.

---

## 3. `session_recall = 1.00` is not evidence of good retrieval

This one matters because it was the basis for the earlier conclusion that
"retrieval is not the bottleneck". Measured over all 500 oracle items:

```
items where EVERY haystack session is a gold session : 500/500  (100%)
items with <= 2 haystack sessions                   : 426/500  (85%)
```

Because every haystack session is also a gold session, **a retriever that returns
any single session scores `session_recall = 1.0` on 100% of items.** The metric
cannot fail, so `1.00` carries no information about retrieval quality. It is not a
contradiction of the retrieval story — it is the absence of a test.

Confirmed by direct probe: for `gpt4_2655b836` the three "retrieved" sessions are
all gold, all dated `2023/04/10`, and all near-duplicates of one another, so
"found the right session" is trivially true.

**Consequence for the thesis:** `session_recall` should not be reported as
evidence for LongMemEval. Use it only on LoCoMo (35 sessions per conversation,
where it can actually fail), or replace it with a metric that can discriminate —
e.g. answer-bearing-turn recall against a *verified* label. Note also that
`gold_session_ids` never matches a haystack id directly (0/948 sessions expose any
id field); it is reconstructed positionally, which happens to be correct here
(verified: `s1` ↔ `haystack_dates[0]`) but is fragile.

### Why `turn_recall` reads 0.0 even when the answer is right

`has_answer` turn flags in the oracle file are frequently set on the **wrong
turn**. For `gpt4_2655b836` the question is about the GPS issue after the first
service; 7 turns in session[1] mention GPS, but the single `has_answer` flag in
that session sits on an unrelated *car-detailing* turn. Scoring
substring-recall against that flag measures the annotation, not the retriever.

So `turn_recall ≈ 0` on LongMemEval oracle should be read as **"this label is
unreliable"**, not as a retrieval failure. Do not quote it in the thesis without
verifying the flags on the slice you cite.


---

## 4. Verification performed

### Date parsing — fixed on both datasets

| corpus | dates checked | unparseable |
|---|---:|---:|
| `locomo10.json` | 288 | **0** (was 0) |
| `longmemeval_oracle.json` | 1,448 | **0** (was 948/948) |
| `longmemeval_s_cleaned.json` | 24,367 | **0** |

Reproduce: `python scripts/audit_temporal_grounding.py --corpus locomo longmemeval`

### Prompts — the chain end-to-end

```
EXTRACTOR SESSION DATE : 2 March 2022 (raw stamp: 2022/03/02 (Wed) 22:25)   # was the raw stamp
ANSWER PROMPT          : Memory 1 — recorded on 2 March 2022, …             # was 3 October 2026
note.t                 : 2022-03-02                                          # was 2026-10-03
```

### Baseline reproduced from the existing run

The new audit independently reproduces the reported failure from
`check_thg_judge2/preds/*.jsonl`:

| kind | n | numeric acc | judge acc |
|---|---:|---:|---:|
| arithmetic | 5 | 0.200 (1/5) | 0.200 (1/5) |
| ordering | 5 | 1.000 (1/1) | 0.600 (3/5) |

### Tests

Run in the project environment (`.github/instructions/python-env.instructions.md`
mandates `memory-r1`; on this Windows host that is
`C:\Users\Dat Pham\anaconda3\envs\memory-r1\python.exe`, plus
`KMP_DUPLICATE_LIB_OK=TRUE` to avoid an OpenMP double-load abort):

```
230 passed, 1 skipped   (whole suite, 44s)
```

Earlier runs of this investigation used the base interpreter and reported
"11 failed"; that was a **wrong-interpreter artifact**, not a real state. Two host
problems are documented in `python-env.instructions.md`: the env name appears twice
in `conda env list`, and `memory-r1` double-links OpenMP (fix:
`KMP_DUPLICATE_LIB_OK=TRUE`).

### W6 — Injected backends were being silently discarded

`eval/systems.py::build_system` forwarded its `backend=` argument only to
ASEM-THG. `build_asem_system`, `build_asem_v2_system` and `build_fast_asem_system`
each called `build_backend(cfg["inference"])` unconditionally, so a caller-supplied
backend — such as the deterministic `StubBackend` the tests inject — was thrown away
and a real model built instead:

```
build_system('FastASEM', ..., backend=StubBackend())
  -> eval/systems.py:1092 build_fast_asem_system(config_path, db_dir)   # backend dropped
  -> eval/systems.py:936  backend = build_backend(asem_cfg.inference)
  -> huggingface_backend.py:135 hf_pipeline(...)       # loads Qwen2.5-1.5B
  -> huggingface_backend.py:145 SentenceTransformer(...)
```

Consequence: `tests/test_static_banks.py` took minutes (one test effectively hung —
it was loading and running a 1.5B model that the test then ignored), and pointing
those tests at an API config instead made them issue **live API calls**
(`403 Please subscribe to model: Qwen2.5-1.5B-Instruct`), because the langchain
backend constructs an `OpenAI` client. Neither config was correct; the bug was the
dropped argument.

All three builders now accept `backend=None` and use it when supplied. `build_system`
forwards it everywhere. Verified:

| builder | before | after |
|---|---|---|
| ASEM-THG | 0.05s | 0.05s |
| ASEM | ~30s (loaded 1.5B) | **0.05s** |
| ASEMv2 | ~30s | **0.04s** |
| FastASEM | ~30s | **0.06s** |

Default behaviour is unchanged for every existing caller (`backend=None` builds from
the config exactly as before). Effect on the suite: `test_static_banks.py` +
`test_phase_benchmark.py` went from hanging/9 failures to **23 passed in 11s**.

### W7 — `answer.temperature` was read by nothing

Every LongMemEval/LoCoMo eval run sampled at **`inference.temperature` (0.3)** while
the config advertised `answer.temperature: 0.0`. Nothing in the codebase read the
latter (`grep` for `ans_cfg.get("temperature")` and `ans_cfg.temperature` returned
zero hits before this change), and neither `AnswerAgent` nor
`SinglePassSessionIngestor` forwarded a temperature to the backend. The consequence
is that the benchmark was **not deterministic**: the same question with the same
config answered `15 days` in one run and `17 days` in another.

Fixes: `AnswerAgent.temperature` (None = inherit backend default), forwarded through
`_call` → `generate_with_cap` (which now accepts and falls back on a temperature
keyword), and wired from the config in all **five** `AnswerAgent` construction sites
(ASEM-THG, ASEM, ASEMv2, FastASEM ×2). Verified end to end: the API request now
carries `temperature=0.0` and `model=chib/deepseek-v4-flash`.

### W8 — Ingestion temperature was also unwired

The W7 fix covered the answer agent; the **ingestor that builds the graph** had the
same gap. `SinglePassSessionIngestor._call_model` passed no temperature, so fact and
entity extraction ran at `inference.temperature` (0.3) — meaning the **same sessions
produced a different graph, and therefore a different memory bank, on every ingest**.

`SinglePassSessionIngestor` now accepts a `temperature`, wired from
`ingestion.temperature` (falling back to `answer.temperature`). Verified: one graph
build now issues `model=chib/deepseek-v4-flash, temperature=0.0`.

This matters more than the answer-side fix, because a bank is *persisted*: two runs
that disagree at ingestion cannot be compared afterwards, and a frozen bank silently
encodes whichever sampling it happened to draw.

### W9 — Where each model runs: LLM via `.env` API, embedder local

Confirmed by instrumenting `openai`'s `chat.completions.create` and the embedder
across a real graph build (`SinglePassSessionIngestor` + `TemporalHyperGraph`):

| stage | model | where |
|---|---|---|
| **Graph builder LLM** (fact/entity extraction) | `chib/deepseek-v4-flash` | **`.env` API** |
| **Answer agent LLM** | `chib/deepseek-v4-flash` | **`.env` API** |
| **Embedder** | `sentence-transformers/all-MiniLM-L6-v2` | local HuggingFace |

Measured on one ingest session: **1 chat API call to `chib/deepseek-v4-flash`,
0 local generation calls**, `hyper_graph: TemporalHyperGraph`.

So "run model DeepSeek for the graph builder via `.env`" is already what happens —
`OPENAI_BASE_URL` / `LLM_MODEL` from `.env` drive every LLM call, including graph
construction.

**The embedder cannot move to the API on this key.** Probed 19 embedding model names
(`text-embedding-3-small/large`, `ada-002`, `bge-m3`, `nomic-embed-text`,
`multilingual-e5-large`, `all-MiniLM-L6-v2`, the full HF path, …) against
`https://api.vilao.ai/v1`: **all 19 return `403 FORBIDDEN` —
`Please subscribe to model in the API Key: <name>`**. `models.list()` advertises only
the chat model, and `api.deepseek.com` rejects the key outright (401). This is a
key-level entitlement limit, not a code problem.

It is also **not desirable to change it**: `asem/backends/openai_backend.py`
deliberately reuses the LangChain backend's `_build_embedder` so that banks stay
byte-compatible across backends (the bank stores embedder *vectors*). Swapping the
embedder invalidates every frozen bank, including the LoCoMo `ds_fixed` banks.

If embedding entitlement is added later, `_build_embedder` now supports the remote
path properly (`embedder_provider: openai`, with `base_url` / `api_key` /
`embedder_dimensions` read from the inference block rather than relying on
`OpenAIEmbeddings` picking up its own environment). Switching also requires a
re-ingest of every bank.

### Latent crash found and fixed during review

The first PPR draft referenced `seen_ids`, which was only bound inside the
`enable_link_traversal` branch. Any config with link traversal **off** and PPR
**on** would have raised `NameError` at query time. `seen_ids` is now initialised
before both expansion phases. Verified across all three paths:

| configuration | result |
|---|---|
| link traversal OFF + PPR ON | works, `phase_d_ppr_active = True` |
| PPR OFF (`ppr_weight: 0.0`) | works, `phase_d_ppr_active = False` |
| no `hyper_graph` (plain ASEM) | works, unchanged |

---

## 5. Confirming run — results

Run: 60 oracle questions, ASEM-THG only, `--tag thg_w1fix`, model
`chib/deepseek-v4-flash` (via `configs/models/deepseek_v41_flash.yaml`),
judge `configs/models/judge_v41_flash.yaml`.

### The four failures you reported

| question | pre-fix | post-fix | gold | verdict |
|---|---|---|---|---|
| `2c63a862` | 445 days | **15 days** | 14 (15 accepted) | fixed |
| `08f4fc43` | 49 days | **30 days** | 30 | fixed |
| `0bb5a684` | 3 days | **7 days** | 7 | fixed |
| `2a1811e2` | 37 days | 30 days | 21 | **still wrong** |

**Three of four fixed, including both wildly-off ones.** The two errors of
445 and 49 days — the signature of subtracting from a corrupted `now()` — are gone.
The remaining miss (`21` → `30`) is an ordinary retrieval/arithmetic error, not the
date-plumbing defect: the answer is a plausible number, not a garbage one.

### Slice breakdown (n=60)

| kind | n | numeric acc | judge acc |
|---|---:|---:|---:|
| arithmetic | 21 | 0.467 (7/15) | 0.476 (10/21) |
| ordering | 33 | 0.600 (3/5) | 0.727 (24/33) |
| temporal-other | 6 | 0.000 (0/2) | 0.500 (3/6) |

Overall: `em 0.1667`, `f1 0.3577`, `judge 0.6167`, abstention acc `0.9500`.

### Read this honestly

* The direction is right and the specific defect is demonstrably gone, but
  **arithmetic numeric accuracy is still below half** (0.467). The date fix removed
  one large error source; it did not make temporal arithmetic easy for this model.
* **The run above was sampled at temperature 0.3**, not the 0.0 the config claimed
  (see W7) — so individual numbers carry sampling noise. The four reported cases
  were re-checked afterwards at the corrected 0.0 and held (3 of 4 fixed). Any
  number quoted in the thesis should come from a post-W7 run.
* Compare against the reproduced pre-fix baseline (`arithmetic 0.200`), but note
  both numbers are computed over different question sets — the baseline was the
  n=10 `check_thg*` slice, this is the first 60 oracle questions. **Treat the
  improvement as directional, not as a like-for-like delta.** A like-for-like
  comparison needs the same questions run both ways with `--clean`.
* New failure modes are visible in the slice and are worth a follow-up: relative
  units the model converts loosely (`5.5 weeks` → "five weeks and three and a half
  days"; `Two weeks` → `21 days`), and several rows where the judge credits an
  answer the numeric check rejects.
* `abstention accuracy 0.95` with only 6 abstention questions — thin.

**Suggested next step:** run the same 60 indices with `retriever.ppr_weight: 0.0`
to isolate the W3 PPR contribution, and 60 matching indices on `s_cleaned` to see
whether the fix holds outside the oracle slice.

---

## 6. What is already done (do not redo)

The older roadmap `analysis/asem_thg_improvement_roadmap.md` lists two P0 items
that are **already implemented** in the current code:

* **P0-1 direct mode** — `eval/systems.py` already sets
  `direct_mode=bool(ans_cfg.get("direct_mode", True))`.
* **P0-2 relation weights** — `superseded_by` / `same-entity` are already in both
  `asem/enhanced_retriever.py` and `asem/retriever.py` weight tables.

Also: `session_recall = 1.00` was **not** evidence that retrieval is fine — see §3a.
It is trivially 1.0 on every LongMemEval oracle item.

---

## 7. Interpretation note (do not over-claim)

The `check_thg*` runs are **n = 10** (all `temporal-reasoning`). "4 of 5 arithmetic
wrong" is directionally solid but must not be quoted as a 100% failure rate.

`CATEGORY_NAMES` in `scripts/run_locomo10_experiments.py` is **correct**
(1 = multi_hop, 4 = single_hop); the mislabelling noted in `report.md` §4.1 is in
the report's own prose, not this mapping.

---

## 8. Remaining work

The confirming run (§4) is done for LongMemEval oracle. Still outstanding:

1. **A like-for-like before/after table.** The pre-fix baseline is the n=10
   `check_thg*` slice and the post-fix result is 60 different oracle questions, so
   the two are not directly comparable. Run the **same 60 indices** both ways
   (revert `parse_session_datetime` via `git stash`, or run the frozen pre-fix
   predictions) to produce a defensible delta for the thesis.
2. **LoCoMo10 re-run** to confirm no regression — the shared parser and renderer are
   what LoCoMo depends on:

   ```powershell
   $env:PYTHONPATH="."
   $env:KMP_DUPLICATE_LIB_OK="TRUE"
   $py="C:\Users\Dat Pham\anaconda3\envs\memory-r1\python.exe"
   & $py scripts/run_static_eval.py --tag w1_fixed `
       --config configs/models/deepseek_v41_flash.yaml `
       --judge-config configs/models/judge_v41_flash.yaml `
       --metrics em f1 rougeL judge --judge
   ```

3. **Full oracle run** (all 500, or at least the two largest categories:
   `temporal-reasoning` n=133, `multi-session` n=133) — 60 rows is a smoke-level
   sample. Note the script runs **all 10 default systems** unless `--systems` is
   given; pass it to keep runs short.

   ```powershell
   & $py scripts/run_longmemeval_eval.py `
       --input datasets/longmemeval/longmemeval_oracle.json `
       --tag w1_fixed_full --clean --systems ASEM-THG `
       --config configs/models/deepseek_v41_flash.yaml `
       --judge-config configs/models/judge_v41_flash.yaml `
       --metrics em f1 judge --judge --per-category --recall
   ```

   **`--judge` is required** for judge scoring; `--metrics ... judge` alone yields
   `judge = 0.0, judge_coverage = 0.0`.

4. **Investigate the remaining arithmetic errors** visible in §4 (loose relative-unit
   conversion: `5.5 weeks` → "five weeks and three and a half days"; `Two weeks` →
   `21 days`). These are generation-side and would be the next highest-leverage fix
   now that the date plumbing is sound.

### Optional ablations now available

| flag | default | effect of setting `0` / `false` |
|---|---|---|
| `retriever.ppr_weight` | `0.15` | disables query-seeded PPR |
| `retriever.enable_global_semantics` | `true` | disables community + global PageRank |

Adding `ppr_weight: 0.0` to `configs/models/deepseek_v41_flash.yaml` reproduces the
pre-W3 ranking exactly, which makes the PPR contribution measurable in isolation.
