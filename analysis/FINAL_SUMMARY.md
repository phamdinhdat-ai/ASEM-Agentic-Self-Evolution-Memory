# ASEM-THG — final summary

**Scope of this round (the user's own direction):** *"just only running and testing
my method, no run other baseline. run simple and verify please."* So this summary
covers **ASEM-THG only**. No baseline system was run. The graph-builder work from
the earlier round is summarised in §1; the evaluation is §2–§4.

Two supporting reports carry the full detail:

| report | what it holds |
|---|---|
| [graph_build_report.md](graph_build_report.md) | the graph builder: banks, counting rules, supersession audit, 4-instance honesty table |
| [thg_evaluation_report.md](thg_evaluation_report.md) | the evaluation: six measured arms, paired significance test, per-category breakdown |

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
| LongMemEval oracle | 27 (smoke, last question) | 237 | — |

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

---

## 3. What the original claim decomposes into

| claim | status | evidence |
|---|---|---|
| **less context than full history** | **PROVEN** | 8 bounded notes vs ~15,687-token history; **16.491×** (LoCoMo) / **4.035×** (LongMemEval) median smaller, **0/120 inversions**, budget-matched at 32,192 tokens |
| **still answers LTM questions** | **PARTIALLY SHOWN** | temporal EM 0.4595 / judge 0.8649; single_hop judge 0.7286 — but these are single-fact/ordering slices |
| **multi-hop** | **NOT SHOWN** | n=32: EM **0.0625**, judge **0.2500** vs legacy judge 0.3438 |

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

**The error class, in four instances from four different people:** a result that
cannot distinguish *the code is broken* from *the measurement was broken*.
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

The operational rule this produced: every headline number must be **printed by a
tool**, never arithmetic on another number; any aggregate over a loop that can fail
must report its **denominator and an error count** and **refuse to emit a rate**
when errors > 0; and a test must not be able to fail for a reason unrelated to what
it asserts.

---

## 5. Verification status

```
suite at final revision 6842ebc : 345 passed, 1 skipped, 0 failed
                                  (reproduced WITH and WITHOUT credentials)
wiring guard on the reintroduced bug : 3 failed, 3 passed
wiring guard with the fix            : 6 passed
asem/single_pass_ingest.py           : 2650CEEA7658116CDFD774E91DA05AAA89DD7373E0B1576B6C40C05FBA7A8841
asem/hyper_graph.py (frozen)          : 342ECB0A3A57BF600EB5C74AF6D589C1673403EBBE17DBB961BE2015D7FBC2E3
```

**Not done, and not claimed:** no baseline system was run (per direction) so the
quality-vs-FullContext axis is **unmeasured**; the ablation is unresolved; `multi_hop`
is not shown; the `superseded_by` defect is documented but the production bank was
not rebuilt; 144/166 supersession pairs were left semantically `unread`; `gb_full`
(LongMemEval) is paused at **160/500** and must **not** be reported as a
500-question result.
