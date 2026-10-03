# ASEM-THG — final summary

**Scope of this round (the user's own direction):** *"just only running and testing
my method, no run other baseline. run simple and verify please."* So this summary
covers **ASEM-THG only**. No baseline system was run. The graph-builder work from
the earlier round is summarised in §1; the evaluation is §2–§4.

**Both datasets are covered.** LoCoMo10 (6 arms, up to n=199) and LongMemEval oracle
(2 arms, **full n=500**) — see §2 and §2b. The ablation result is a **negative that
replicates on both**; §3 records what that leaves standing.

Two supporting reports carry the full detail:

| report | what it holds |
|---|---|
| [graph_build_report.md](graph_build_report.md) | the graph builder: banks, counting rules, supersession audit, honesty table |
| [thg_evaluation_report.md](thg_evaluation_report.md) | the evaluation: six LoCoMo arms + both LongMemEval arms at n=500, paired significance tests, per-category breakdown, the two-dataset headline (§9) |

---

## 1. What was built

**ASEM-THG** = `SinglePassSessionIngestor` + `TemporalHyperGraph` + `EnhancedHybridRetriever`,
with deterministic LLM-free relations (`same-entity`, `temporal`, `semantic`,
`superseded_by`). Graph-builder fixes GB1–GB5 landed and are frozen at `70f69c8`,
re-verified at `2f41444`.

**Banks built (both datasets, 10/10 conversations, 0 errors):**

| bank | notes | links | elapsed |
|---|---:|---:|---:|
| `thg_v2` (LoCoMo10) | 4,579 | 73,862 | 2,216.9 s |
| `thg_v2_postfix` (LoCoMo10) | 4,664 | 73,640 | 2,047.5 s |
| LongMemEval oracle | per-question (private haystacks, reset each question) | — | — |

**Three graph findings that constrain the design:**

1. **The hub is 94.7% of the entity channel** — 21,378 of 22,565 `same-entity`
   edges are hub-only; **1,187 are discriminative (5.3%)**, noise-to-signal **19:1**.
   The hub is a *class* of token: LongMemEval's hub is the role token `user` (67%
   fan-out), LoCoMo's is speaker names `caroline`/`melanie` (65%/56%). **Two corpora
   reach ~65% by different routes, so no hardcoded stop-set works — df-based IDF is
   the only mechanism that covers both.**
2. **Removing 96% of the edge volume moves the hub share by 0.67pp** — `locomo_0000`
   goes from 46,084 unweighted pairs to 1,792 stored edges (25.7×) while hub share
   moves 99.94% → 99.27%. **This proves `max_entity_links` is not the lever; only
   IDF weighting is.**
3. **GB5's reach is bounded and must be stated that way** — the IDF weighting
   demonstrably promotes rare-sharing peers where they exist (witness `00cb74d3`:
   rank #5 → #1, score 0.6589 → 3.8245, displacing 4 hub-only peers), **but only
   23/365 notes (6.3%) have any rare peer in their top-5, and the corpus-level
   top-5 split is unchanged at 1769/52.** 6.3% ≈ 5.3% of the channel carrying
   signal: two independent measurements agreeing. **Bounded reach, not a corpus
   improvement** — do not report GB5 as "improves discriminative ranking".

---

## 2. The evaluation — the headline is negative

Six arms = 2 banks × 2 `context_style` × 2 sample sizes. **`judge_coverage` is 1.00
in all six, with 0 judge failures — including 0/199 in both full-N arms.**

| bank | `context_style` | n | EM | F1 | BERTScore-F1 | judge |
|---|---|---:|---:|---:|---:|---:|
| `thg_v2` | `thg_v1` (the method) | 20 | 0.3000 | 0.5762 | 0.9090 | 0.7000 |
| `thg_v2` | `legacy` | 20 | 0.3000 | 0.5675 | 0.9006 | 0.6500 |
| `thg_v2_postfix` | `thg_v1` | 20 | 0.3500 | 0.5543 | 0.8925 | 0.6500 |
| `thg_v2_postfix` | `legacy` | 20 | 0.3500 | 0.5093 | 0.8937 | 0.7000 |
| **`thg_v2_postfix`** | **`thg_v1`** | **199** | **0.3065** | 0.4851 | 0.8702 | **0.6181** |
| **`thg_v2_postfix`** | **`legacy`** | **199** | **0.3065** | 0.4935 | 0.8685 | **0.6533** |

The two configs differ in **exactly 1 of 60 keys** (`answer.context_style`), so the
comparison is one-variable clean. Paired test at N=199 (both arms answer the same
199 questions, paired by `idx`), `analysis/ws4_paired_significance.py`:

```
both correct 115 | thg_v1 only 8 | legacy only 15 | neither 61
judge accuracy  thg_v1 0.6181 (123/199)   legacy 0.6533 (130/199)
discordant 23    exact two-sided binomial p = 0.2100   -> NOT significant
BERTScore  paired mean diff +0.0017, median 0.0000, 75/72/52 -> p = 0.8690
```

**Read plainly: EM ties exactly; judge leans to `legacy`; BERTScore is flat; none of
it is significant.** At N=20 on the mixed-window bank the method looked ahead on
judge (+0.05) — **the sign reverses on the clean bank, at both N.** The F1/ROUGE-L
tilt that briefly favoured the method at N=20 also **reverses at N=199**.

**So: the context builder is NOT established as an improvement on answer quality at
any N measured.** This is reported as the headline rather than buried, because a
"+0.05 win" from the first bank alone would not have survived a second bank.

### 2b. LongMemEval — same ablation, full 500 questions, and the null REPLICATES

Run after the LoCoMo arms, on the user's instruction to check LongMemEval as well.
**Both arms reached the full 500**, each with `judge_coverage` 1.00 and **0 judge
failures in 500**:

| arm | n | EM | F1 | ROUGE-L | BERTScore-F1 | judge |
|---|---:|---:|---:|---:|---:|---:|
| `thg_v1` (the method) | 500 | 0.2740 | 0.4166 | 0.4092 | 0.8833 | 0.5880 |
| `legacy` | 500 | 0.2680 | 0.4232 | 0.4131 | 0.8860 | 0.6140 |

Paired by question id, `analysis/longmemeval_analysis.py`:

```
both correct 236 | thg_v1 only 58 | legacy only 71 | neither 135
judge accuracy  thg_v1 0.5880 (294/500)   legacy 0.6140 (307/500)
discordant 129   exact two-sided binomial p = 0.2907   -> NOT significant
```

**The negative finding replicates across both datasets**: p ≥ 0.05 on LoCoMo (0.2100,
n=199) and LongMemEval (0.2907, n=500), with the sign leaning to `legacy` in both.

**And the long-term-memory weakness is now measured at large n, not asserted:**

| `question_type` | n | EM | judge |
|---|---:|---:|---:|
| multi-session | 133 | 0.1729 | **0.4586** |
| temporal-reasoning | 133 | 0.1353 | **0.4887** |
| knowledge-update | 78 | 0.3333 | 0.7051 |
| single-session-user | 70 | 0.6714 | **0.9571** |
| single-session-assistant | 56 | 0.4107 | 0.6071 |
| single-session-preference | 30 | 0.0000 | 0.4000 |

**`single-session-user` 0.9571 vs `multi-session` 0.4586 is a 0.50 absolute judge
gap** — larger than anything on LoCoMo, and the two slices the claim rests on
(`multi-session`, `temporal-reasoning`) are the weak ones. On LoCoMo that slice was
n=32; here it is **n=266 combined**.

**One metric lesson worth keeping:** `single-session-preference` (n=30) scores **EM
0.0000 in both arms** while the judge scores **0.4000 / 0.5333**. An EM-led headline
would report a total-failure cell where the judge measures 0.40–0.53 — which is why
the judge/BERTScore framing in §3 is the right one for this dataset.

---

## 3. What the original claim decomposes into

| claim | status | evidence |
|---|---|---|
| **less context than full history** | **PROVEN on both datasets** | 8 bounded notes vs ~15,687-token history; **16.491×** (LoCoMo) / **4.012×** (LongMemEval) median smaller, **0/120 inversions**, budget-matched at 32,192 tokens |
| **still answers LTM questions** | **PARTIALLY SHOWN** | temporal EM 0.4595 / judge 0.8649; single_hop judge 0.7286; LongMemEval `single-session-user` judge 0.9571 — but these are single-fact/ordering slices |
| **multi-hop** | **NOT SHOWN** | LoCoMo n=32: EM **0.0625**, judge **0.2500** vs legacy 0.3438. LongMemEval `multi-session` (n=133) judge **0.4586** / EM **0.1729** |

The cost claim is the strong one, and it is **independent of the ablation**: every
arm renders exactly **8 notes** (`max_context_notes=8`), so context is **bounded and
does not grow with history length**, while FullContext's cost swings 4× on
LongMemEval. That bounded-cost property is the most defensible result in this work.

**Cost-claim limitation, stated so it travels with the number:** the ratio comes
from `tests/efficiency_report.json`, whose note list is built **in code** and ranked
by a lexical-overlap stand-in (`dump_efficiency.py:206`). It therefore measures the
**context builder under a fixed note list**, not end-to-end bank retrieval.

---

## 4. One critical bug found and fixed, and four instances of one error class

**The bug (`b4aec5e`):** `asem/single_pass_ingest.py:174` read
`self.hyper_graph = hyper_graph or TemporalHyperGraph()`. **`TemporalHyperGraph`
defines `__len__`, so a freshly constructed graph is FALSY** — the `or` silently
**discarded** the caller's configured graph. Measured, with `supersession: false`:

```
retriever.hyper_graph.enable_supersession = False
ingestor.hyper_graph.enable_supersession  = True    <- constructor default restored
retriever.hyper_graph is ingestor.hyper_graph -> False
```

So the disable flag reached the **read side only**, and the ingestor kept writing
`superseded_by` edges: **242 records (~121 pairs) where ~0 was expected**, with those
semantically-wrong edges **promoted at 1.3×, the highest weight in the table**.
Fixed with an explicit `is not None` test; guard at `tests/test_system_wiring.py`,
**proven to fail on the bug** (revert → 3 failed; fixed → 6 passed). **`thg_v2_postfix`
predates the fix and must never be described as "supersession disabled".**

**Why 339 green tests missed it:** every existing test constructed
`TemporalHyperGraph` **directly** and checked the class — which was correct. The
defect was in **which object the ingestor ends up holding**, invisible to any
class-level test by construction.

**The error class, in six instances from several people:** a result that cannot
distinguish *the code is broken* from *the measurement was broken*.
1. `judge_api.yaml` named an unentitled model → `403` on every call → falsy `None` →
   `judge 0.0, coverage 0.0` — a clean-looking zero. It was the **default**
   `--judge-config`, referenced in 20 places. Fixed in `1164aa3`; new preflight
   `scripts/check_judge_entitlement.py`.
2. An exposure measurement called `retrieve(query, M)` with one argument → `TypeError`
   on every call → swallowed → `None` → **a false 0% exposure**. Corrected to
   **166/166 = 100%**.
3. An aggregate reported as `0.0` where the real object had a **well-formed nonzero**
   that measured the wrong thing.
4. **My own wiring guard** required `OPENAI_API_KEY` and printed `2 failed, 2 passed`
   — **identical output to the regression it existed to catch**. Fixable only because
   a teammate ran the suite to verify my quoted count rather than trusting it.
5. **The `--judge` flag omitted** from the LongMemEval run → `judge 0.0, coverage 0.0`
   with **no `scores/` file**. The two runners differ: one gates the judge on a flag,
   the other on `--metrics`. **A command pattern learned on one silently half-works
   on the other, producing a complete-looking file with a plausible zero.**
6. **My own analysis tool printed that missing measurement as `judge=0.0000` and
   exited 0** — its guard keyed on a field *inside* the absent file. **The instrument
   built to prevent this error class committed it.** Now raises, and refuses a judge
   figure at any coverage other than 1.0.

The operational rule this produced: every headline number must be **printed by a
tool**, never arithmetic on another number; any aggregate over a loop that can fail
must report its **denominator and an error count** and **refuse to emit a rate**
when errors > 0; and a test must not be able to fail for a reason unrelated to what
it asserts.

---

## 5. Verification status

Suite at final revision: **345 passed, 1 skipped, 0 failed** (reproduced WITH and
WITHOUT credentials). LongMemEval artifact integrity: both arms **500/500 preds,
500/500 judge rows, 0 judge failures, coverage 1.00**.

```
suite at final revision 6842ebc : 345 passed, 1 skipped, 0 failed
                                  (reproduced WITH and WITHOUT credentials)
wiring guard on the reintroduced bug : 3 failed, 3 passed
wiring guard with the fix            : 6 passed
asem/single_pass_ingest.py           : 2650CEEA7658116CDFD774E91DA05AAA89DD7373E0B1576B6C40C05FBA7A8841
asem/hyper_graph.py (frozen)          : 342ECB0A3A57BF600EB5C74AF6D589C1673403EBBE17DBB961BE2015D7FBC2E3
```

**Artifacts:** `data/benchmarks/results/longmemeval/gb_full/` and `gb_full_legacy/`
(aggregate `.json`/`.md`, `preds/` 500 rows, `scores/` 500 judge rows each);
LoCoMo arms under `data/benchmarks/results/locomo/`.

**Not done, and not claimed:** no baseline system was run (per direction) so the
quality-vs-FullContext axis is **unmeasured**; the ablation is unresolved on both
datasets; `multi_hop` is not shown; the `superseded_by` defect is documented but the
production bank was not rebuilt; 144/166 supersession pairs were left semantically
`unread`. A **paired BERTScore test was not computable on LongMemEval** (BERTScore is
not carried in `scores/*.jsonl` and the legacy `preds/*.jsonl` lacks the column), so
its BERTScore figures are **unpaired aggregates** and must not be read as a paired
effect. Both LongMemEval arms are **two-window predictions** (164 carried forward,
336 generated here) with **single-pass uniform scoring** over all 500 — the code was
verified unchanged across the boundary, so this is weaker than `thg_v2`'s
mixed-window issue and applies equally to both arms.

**Two further defects found during the LongMemEval runs, both in the Lead's own
tooling** — the error class is now at **six instances**:
5. **The `--judge` flag was omitted** from the first `gb_full` run, giving
   `judge 0.0 / coverage 0.0` with **no `scores/` file at all**.
   `run_longmemeval_eval.py` gates the judge on the **flag** (`:676-677`, `:722`),
   *not* on `"judge" in --metrics`; `run_static_eval.py` has no such flag. **A
   command pattern learned on one runner silently half-works on the other.** Repair
   was a record re-score, not a re-answer.
6. **`analysis/longmemeval_analysis.py` printed that missing measurement as
   `judge=0.0000` and exited 0** — the guard keyed on `judge_correct is None` *in the
   scores file*, and with no scores file there were no rows, so it never fired. **The
   instrument built to prevent this error class committed it.** Fixed: raises on an
   absent scores file and refuses when `judge_coverage != 1.0` (exit 3).
