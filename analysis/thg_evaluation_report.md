# ASEM-THG evaluation report (WS4)

**Owner:** WS4 (`evaluator`), shared task **task-4**
**Scope (final, per user directive):** **ASEM-THG only.** The user narrowed the
request to *"just only running and testing my method, no run other baseline. run
simple and verify please."* The multi-system matrix in `task-4`'s original
description (FullContext / NoMemory / SimRetrieval columns) was **cancelled** and
**no baseline system was run**. See §3 for what that means for the claim.

**THE HEADLINE, STATED UP FRONT (it is negative, and that is the finding):** the
`thg_v1`-vs-`legacy` context-builder ablation **does not establish an improvement
on answer quality at any N measured**. At the full **N=199** on the clean bank,
**EM ties (0.3065 vs 0.3065)**, `judge` **leans to legacy** (0.618 vs 0.653,
paired **p = 0.21 — not significant**), and **BERTScore is flat** (paired median
diff **0.0000**). What **is** established is the **cost** axis. Details §1.2, §7.

**Frozen revision under test:** commit `70f69c8f…`, `asem/hyper_graph.py` =
`342ECB0A3A57BF600EB5C74AF6D589C1673403EBBE17DBB961BE2015D7FBC2E3` — recorded
immediately before the arms, and re-verified unchanged after the full-N runs. The
suite at final revision `6842ebc` was **345 passed, 1 skipped, 0 failed**
(reproduced both with and without credentials; the 339 figure stood at the
`70f69c8f` freeze and grew as the wiring guard was added — see L2b).

**Six arms were measured** = 2 banks × 2 `context_style` × 2 sample sizes
(N=20 and the full N=199). **`judge_coverage` is 1.00 in all six, with 0 judge
failures — including 0/199 in both full-N arms.**

Every headline number in this report was **printed by a script in `analysis/`**
reading raw artifacts. No headline number is arithmetic on another headline
number, and no number is copied from another teammate's prose. Generators:

| generator | produces |
|---|---|
| `analysis/ws4_ablation_table.py` | §1, §1.1 — the quality table and the N=20 ablation delta |
| `analysis/ws4_paired_significance.py` | §1.2 — the N=199 paired test (Lead's, `6842ebc`; **re-run by me, reproduced exactly**) |
| `analysis/ws4_tables.py` | §2 — cost distributions, ratios, inversions |
| `analysis/ws4_variance.py` | §6 L5 — variance band (written; **not run**, per Lead) |
| `analysis/_ws4_ablation_table.txt` | verbatim console output of the ablation table |

---

## 1. The measurement (ASEM-THG only, one variable)

**Design.** One real frozen bank, `static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000`
(365 notes, 19 sessions) — a **read-only** reuse; nothing was written into the bank
tree. Two arms differing in **exactly one of 60 config keys**
(`answer.context_style`; the difference is machine-verified in
`analysis/_ablation_config_diff.json`). Identical everything else: bank, system,
conversation, limit, metrics, judge config, judge workers.

```powershell
$py="C:\Users\Dat Pham\anaconda3\envs\memory-r1\python.exe"
$env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONPATH="."

# Arm A — the method (context_style = thg_v1)
& $py scripts/run_static_eval.py --tag thg_v2 `
    --input datasets/locomo/locomo10.json `
    --config configs/models/deepseek_v41_flash_thg_v1.yaml `
    --systems ASEM-THG --conversations locomo_0000 --limit 20 `
    --metrics em f1 rougeL bertscore_f1 judge `
    --judge-config configs/models/judge_v41_flash.yaml --judge-workers 4 `
    --results-dir data/benchmarks/results/ws4_ablation/thgv1

# Arm B — the ablation (context_style = legacy). Only --config and --results-dir differ.
& $py scripts/run_static_eval.py --tag thg_v2 `
    --input datasets/locomo/locomo10.json `
    --config configs/models/deepseek_v41_flash.yaml `
    --systems ASEM-THG --conversations locomo_0000 --limit 20 `
    --metrics em f1 rougeL bertscore_f1 judge `
    --judge-config configs/models/judge_v41_flash.yaml --judge-workers 4 `
    --results-dir data/benchmarks/results/ws4_ablation/legacy
```

### Results — N = 20, `locomo_0000`, judge coverage 1.0 in both arms

| metric | thg_v1 | legacy | delta | sign | n |
|---|---:|---:|---:|---|---:|
| EM | 0.3000 | 0.3000 | **+0.0000** | tie | 20 |
| F1 | 0.5762 | 0.5675 | +0.0087 | thg_v1 | 20 |
| ROUGE-L | 0.5429 | 0.5342 | +0.0087 | thg_v1 | 20 |
| BERTScore-F1 | 0.9090 | 0.9006 | +0.0084 | thg_v1 | 20 |
| **judge** | **0.7000** | **0.6500** | **+0.0500** | thg_v1 | 20 |
| `judge_coverage` | **1.0** | **1.0** | — | valid | 20 |
| judge failures (`judge_correct is None`) | **0 / 20** | **0 / 20** | — | no outage | 20 |
| judge labelled "incorrect" | 6 / 20 | 7 / 20 | −1 | thg_v1 | 20 |

**Artifacts (raw):**
* `data/benchmarks/results/ws4_ablation/thgv1/locomo10/thg_v2__chib_deepseek_v4_1_flash.json`
* `data/benchmarks/results/ws4_ablation/thgv1/locomo10/scores/thg_v2__chib_deepseek_v4_1_flash__ASEM-THG.jsonl`
* the same two paths under `.../ws4_ablation/legacy/locomo10/`

### Reading this honestly

* `judge` moves **+0.05 = exactly ONE question** (14/20 → 13/20 correct).
* `EM` is a **tie**.
* **It is not statistically meaningful at N=20**, and §5 shows this exact command
  reproduces *different* numbers run to run.
* **§1.1 and §1.2 below show this advantage does not survive** — it reverses on a
  second bank, reverses again on a full-N run, and the paired test is
  non-significant. **§1.2 is the primary ablation result; read it before quoting
  anything above.**

### 1.1 Replication on the clean bank (N=20)

`thg_v2` is a **mixed-window** bank (§6 L2). The Lead's preferred bank is
`thg_v2_postfix`, built from the frozen revision. Once it completed (10/10
conversations) I ran **the same two commands** with `--tag thg_v2_postfix` and
isolated `--results-dir`s. `hyper_graph.py` = `342ECB0A…`, **verified unchanged
immediately before AND after these runs**.

### 1.2 The primary result — full N=199, paired test (READ THIS ONE)

The two arms answer the **same 199 questions**, so they can be **paired by `idx`**
— which turns "not resolved at N=20" into a real test. `--limit` was dropped;
everything else is identical to §1. `judge_coverage` **1.0 in all arms**, and
**0 judge failures in both N=199 arms** — the coverage property holds at full
scale, so these judge numbers are the most trustworthy in the project.

```powershell
# thg_v1
& $py scripts/run_static_eval.py --tag thg_v2_postfix `
    --input datasets/locomo/locomo10.json `
    --config configs/models/deepseek_v41_flash_thg_v1.yaml --systems ASEM-THG `
    --conversations locomo_0000 --metrics em f1 rougeL bertscore_f1 judge `
    --judge-config configs/models/judge_v41_flash.yaml --judge-workers 6 `
    --results-dir data/benchmarks/results/ws4_ablation/full199_postfix_thgv1
# legacy: identical except --config configs/models/deepseek_v41_flash.yaml
#   and --results-dir .../full199_postfix_legacy
```

**All six arms measured in this work** (`judge_coverage` 1.00 in every one):

| bank | arm | n | EM | F1 | ROUGE-L | BERTScore-F1 | judge | judge failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `thg_v2` (mixed) | thg_v1 | 20 | 0.3000 | 0.5762 | 0.5429 | 0.9090 | 0.7000 | 0/20 |
| `thg_v2` (mixed) | legacy | 20 | 0.3000 | 0.5675 | 0.5342 | 0.9006 | 0.6500 | 0/20 |
| `thg_v2_postfix` | thg_v1 | 20 | 0.3500 | 0.5543 | 0.5209 | 0.8925 | 0.6500 | 0/20 |
| `thg_v2_postfix` | legacy | 20 | 0.3500 | 0.5093 | 0.4759 | 0.8937 | 0.7000 | 0/20 |
| **`thg_v2_postfix`** | **thg_v1** | **199** | **0.3065** | 0.4851 | 0.4717 | 0.8702 | **0.6181** | **0/199** |
| **`thg_v2_postfix`** | **legacy** | **199** | **0.3065** | 0.4935 | 0.4836 | 0.8685 | **0.6533** | **0/199** |

**Paired test** (`analysis/ws4_paired_significance.py`, committed `6842ebc`;
**I re-ran it myself and reproduced every figure below exactly** — exits 0):

| | count |
|---|---:|
| paired questions | 199 |
| both correct | 115 |
| **thg_v1 correct only** | **8** |
| **legacy correct only** | **15** |
| neither correct | 61 |
| discordant pairs | 23 |
| **exact two-sided binomial p** | **0.2100** → **NOT significant** |
| judge accuracy | thg_v1 **0.6181** (123/199) · legacy **0.6533** (130/199) |
| BERTScore paired mean diff | **+0.0017** |
| BERTScore paired **median** diff | **+0.0000** · wins/loses/ties 75/72/52 · **p = 0.8690** |

**THE HONEST HEADLINE — the context builder is NOT established as an improvement:**

* **EM is a dead TIE at N=199 (0.3065 = 0.3065)** — not merely "tied at N=20".
* **`judge` leans to LEGACY on the clean bank, at both sample sizes**: 0.6533 vs
  0.6181 at N=199, and 0.70 vs 0.65 at N=20 on the same bank. The reversal first
  seen at N=20 **persists and reproduces at N=199**. This is no longer "one bank
  disagrees" — it is *the clean bank disagreeing at both N*.
* **It is still not significant (p = 0.21).** 8 vs 15 discordant is a real lean,
  not a proven loss. Correct statement: **the method does not beat legacy, and on
  the clean bank leans slightly behind on judge while tying on EM — neither
  established.**
* **BERTScore — the metric the user named — is exactly nothing**: median paired
  difference **0.0000**, 75/72/52, p = 0.87. Reporting this plainly matters,
  because it is the metric that was specifically requested.
* **The F1/ROUGE-L tilt toward thg_v1 does NOT survive the larger sample.** At
  N=20 on `thg_v2_postfix` thg_v1 led on F1 (0.5543 vs 0.5093); at N=199 **legacy
  leads** (0.4935 vs 0.4851) and on ROUGE-L (0.4836 vs 0.4717). That removes the
  last holdout for a "slightly positive" reading.

**So: the context builder's contribution is not established on answer quality at
any N measured.** The cost axis (§2) stands alone as the positive contribution.

### 1.3 Per-category at N=199 — the multi-hop cell, at usable n

| category | n | arm | EM | F1 | BERTScore-F1 | judge |
|---|---:|---|---:|---:|---:|---:|
| **multi_hop** | **32** | thg_v1 | **0.0625** | 0.2585 | 0.8463 | **0.2500** |
| **multi_hop** | **32** | legacy | **0.0625** | 0.2445 | 0.8470 | **0.3438** |
| temporal | 37 | thg_v1 | 0.4595 | 0.6830 | 0.9316 | 0.8649 |
| temporal | 37 | legacy | 0.3784 | 0.6598 | 0.9249 | 0.8378 |
| single_hop | 70 | thg_v1 | 0.2286 | 0.4813 | 0.9016 | 0.7286 |
| single_hop | 70 | legacy | 0.2429 | 0.4877 | 0.8989 | 0.7000 |
| adversarial | 47 | thg_v1 | 0.5532 | 0.5731 | 0.7961 | 0.5532 |
| adversarial | 47 | legacy | 0.5957 | 0.6278 | 0.7974 | 0.6596 |
| open_domain | 13 | thg_v1 | 0.0000 | 0.1824 | 0.8534 | 0.4615 |
| open_domain | 13 | legacy | 0.0000 | 0.1783 | 0.8544 | 0.6154 |

**`multi_hop` is now measured at n=32 (not n=8) and it is still the worst cell:
EM 0.0625 = 2 of 32, and judge 0.2500 vs legacy's 0.3438.** The method does not
win the multi-hop slice at any n measured. The one slice where thg_v1 leads is
`temporal` (EM 0.4595 vs 0.3784) — a **single-fact/ordering** slice — so **a win
there does not support a multi-hop claim.**

> **Note on `--limit` semantics.** §1/§1.1 use the first 20 pairs; §1.2/§1.3 use
> all 199. These are **different question sets**, so an N=20 vs N=199 numeric
> difference is **not** a like-for-like delta and is not presented as one.

**Two confounds must be attached to any cross-bank reading of the six-arm table**
(both found by `bank-runner`; the second was fixed by the Lead *after* the bank
was built):
* **C1 — note population differs**, so the banks are not comparable note-for-note
  (`locomo_0000`: 365 → 406 notes). A cross-bank accuracy delta therefore mixes
  "more evidence available" with "more retrieval noise". **The §1.2 paired test is
  within a single bank, so it is not exposed to C1.**
* **C2 — `superseded_by` is LIVE in `thg_v2_postfix` despite the config intending
  it off.** The root cause was a **truthiness bug**, not a missing wire:
  `asem/single_pass_ingest.py:174` read `hyper_graph or TemporalHyperGraph()`, and
  **`TemporalHyperGraph` defines `__len__`, so a freshly constructed graph is
  FALSY** — the `or` silently discarded the configured graph and the ingestor kept
  its own constructor-default (`enable_supersession=True`). **Fixed by the Lead in
  `b4aec5e`** (explicit `is not None` test; guard at `tests/test_system_wiring.py`,
  proven to fail on the bug: reverting the line gives `2 failed, 2 passed`).
  **`thg_v2_postfix` was built BEFORE this fix, so the 242 live records remain in
  the artifact measured here** — a rebuild would be needed to remove them, and the
  Lead deliberately did not order one, since the confound would only *help* the
  method and therefore cannot rescue a tie. **`thg_v2_postfix` is NOT a
  "supersession-disabled" bank and is not described as one anywhere here.**
  The original defect report was `bank-runner`'s (§4A.1 of their report).
  `bank-runner` measured **242 `superseded_by` records (~121 pairs)** in
  `thg_v2_postfix`. Since `relation_weights["superseded_by"] = 1.3` is the
  **highest** weight in the table, and 22/22 audited pairs were semantically
  **wrong**, these wrong edges are **promoted** in ranking. Any ASEM-THG
  knowledge-update figure from this bank therefore includes that effect — a
  **candidate explanation**, not evidence of a retrieval-design failure.
* **The cost axis (§2) is unaffected by both confounds** — `max_context_notes`
  bounds the rendered note count directly, independently of the bank.

### Per-category at N=20 (`thg_v2`) — superseded by §1.3 at usable n

| category | n | EM | F1 | BERTScore-F1 | judge |
|---|---:|---:|---:|---:|---:|
| multi_hop | 8 | **0.0000** | 0.4390 | 0.8836 | 0.5000 |
| open_domain | 2 | **0.0000** | 0.2481 | 0.8416 | 1.0000 |
| temporal | 10 | 0.6000 | 0.7516 | 0.9429 | 0.8000 |

Legacy: `multi_hop` n=8 EM 0.1250 F1 0.4648 judge 0.5000 · `open_domain` n=2
EM 0.0000 F1 0.1577 judge 0.5000 · `temporal` n=10 EM 0.5000 F1 0.7316
judge 0.8000. The `open_domain` cell is **n=2** and carries no weight.
**§1.3 supersedes this with the full N=199 breakdown — use that one.**

---

## 2. Cost — "materially less context"

Two independent measurements, with their different scopes kept separate.

### 2.1 The bounded-context property (this run)

Every question in §1 renders **exactly `max_context_notes = 8`** notes. The
conversation's full history is **~15,687 tokens** (Lead's measurement,
`scripts/verify_asem_thg_simple.py`, commit `1d25533`). So the answer context is
**bounded and independent of conversation length**, while full-history context
grows with it.

### 2.2 The ratio and its distribution (reproduced independently)

From `tests/efficiency_report.json`, **recomputed by `analysis/ws4_tables.py`**,
not copied from prose. Both arms budget-matched via the shared
`asem.token_budget.resolve_budget`: **`context_window 32768 − max_tokens 512 −
safety 64 = 32,192` token budget** (re-derived from the configs by the script).

| dataset | system | min | median | p90 | max | inversions |
|---|---|---:|---:|---:|---:|---:|
| LoCoMo (N=60) | ASEM-THG thg_v1 | 887 | **1077** | 1231 | 1360 | **0 / 60** |
| LoCoMo (N=60) | FullContext | 17764 | **17769** | 17774 | 17783 | — |
| LongMemEval (N=60) | ASEM-THG thg_v1 | 1712 | **1928** | 2024 | 2080 | **0 / 60** |
| LongMemEval (N=60) | FullContext | 5122 | **7782** | 9510 | 22057 | — |

**Ratio (computed by the script):** LoCoMo median **16.491×** smaller;
LongMemEval median **4.035×** smaller. **Worst case is still smaller** — LoCoMo
ratio max 0.0765, LongMemEval ratio max 0.3772, and **0/120 inversions**.

This reproduces the previously-recorded figures (16.5× / 4.0×, 0 inversions in 120
questions) to the stated precision. No divergence found.

> **LIMITATION — carries with the number.** `tests/efficiency_report.json` is
> generated by `tests/dump_efficiency.py`, which **builds its note list in code**
> and ranks it with a deterministic lexical-overlap stand-in for the retriever
> (`rank()`, line 206). It uses real question text and real FullContext history,
> but the ASEM-THG side is **not a real bank retrieval**. It is therefore a valid
> measurement of **the context builder's cost under a fixed note list** — correct
> for the ablation and the budget contract, and **not the end-to-end cost of the
> real system**. The end-to-end bounded-context evidence is §2.1.

---

## 3. What was NOT measured, and why it matters to the claim

The original claim was two-axis: *"ASEM-THG answers long-term-memory and multi-hop
questions at least as well as FullContext while spending materially less
context."*

| axis | status in this report |
|---|---|
| cost vs FullContext | **supported** — 16.5× / 4.0× median smaller, 0/120 inversions, plus a bounded 8-note context vs ~15,687-token history (§2) |
| quality vs FullContext | **NOT MEASURED** — no baseline was run, by user directive |

**Therefore this report does not prove the original two-axis claim, and does not
claim to.** What it establishes is the *method-side* half: ASEM-THG produces
answers at judge 0.618–0.700 / BERTScore-F1 0.8685–0.9090 / F1 0.4851–0.5762 /
EM 0.3000–0.3500 across the **six arms** of §1–§1.3 while spending a bounded, far
smaller context. Whether that quality **matches FullContext** is exactly the
question this scope cannot answer.

One FullContext LoCoMo datapoint does exist from a run before the scope change —
`n=199, em 0.171, f1 0.455, rougeL 0.440, bertscore_f1 0.881, judge 0.653,
judge_coverage 1.0`. It is **recorded here only as a secondary artifact, not as
the controlled comparison**: it is the **full 199-pair** set whereas §1 is the
**first 20** pairs of one conversation, so the two are **not like-for-like** and
must not be tabulated as a head-to-head.

---

## 4. Metrics hygiene — two defects found, both fixed/flagged

### 4.1 A total judge outage that rendered as a plausible `0.0` (found, fixed)

My first bootstrap run (`--limit 3`) printed **`judge = 0.0, judge_coverage =
0.0`**. That is not a score. All 3/3 rows carried
`judge_error = "403 FORBIDDEN … Please subscribe to model in the API Key: deepseek-v4-flash"`.
`configs/models/judge_api.yaml` named a model id this key is not entitled to —
and it was the **DEFAULT `--judge-config`** in `scripts/run_longmemeval_eval.py:638`,
so the default invocation produced a silent all-zero judge row.

Proven root cause (direct probe of the same key):

```
https://api.vilao.ai/v1/models advertises exactly: ['deepseek-v4.1-flash']
  deepseek-v4-flash        -> 403 FORBIDDEN (not entitled)
  chib/deepseek-v4-flash   -> 403 FORBIDDEN (not entitled)
  chib/deepseek-v4.1-flash -> 200 'OK'
```

Independently reproduced by the Lead. Fixed in `1164aa3`; a preflight
(`scripts/check_judge_entitlement.py`) now fails loudly. **Every judge number in
this report was produced with `configs/models/judge_v41_flash.yaml` and reports
its coverage.**

### 4.2 My own tool produced a false alarm (found in my own code)

My first `ws4_ablation_table.py` counted `judge_error != null` as a judge error
and **flagged both arms INVALID with "6 and 7 errors"**. That was wrong:
`judge_error` is **not an exception slot** — `eval/llm_as_a_judge.py` uses it to
carry the judge's **verdict category** for an *incorrect* answer (`missing_info`,
`contradiction`, `vague`). A row is a genuine failure only when
**`judge_correct is None`** (which is also what distinguishes a real transport
outage). Both arms have **0**. Recorded because a reader grepping `judge_error`
will hit the same false positive; the check and a docstring warning are now in
`analysis/ws4_ablation_table.py`.

### 4.3 `run_static_eval.py` overwrites one arm's results with the other's (flagged, not fixed — outside my scope)

`model_tag_from_config()` returns the **same** `display_tag` for both configs:

```
configs/models/deepseek_v41_flash.yaml        -> chib_deepseek_v4_1_flash
configs/models/deepseek_v41_flash_thg_v1.yaml -> chib_deepseek_v4_1_flash
```

The output path is `<results_dir>/<dataset>/<tag>__<display_tag>.json` (`:350`),
and `--tag` **must** stay `thg_v2` to find the bank. **Both arms therefore write
the identical filename, and the second run silently overwrites the first.** My
first legacy run did exactly that; I caught it only by comparing files, because
the printed table looked perfectly plausible. **Workaround used: a distinct
`--results-dir` per arm.** Recommended fix: include `context_style` in
`display_tag`, or document the `--results-dir` requirement.

---

## 5. Reproducibility — the same command gave different numbers

> **Status: superseded as the reason for caution, retained as an observation.**
> The claim no longer rests on this variance — it rests on the **direct paired
> test at N=199** (§1.2, p = 0.21). Kept because it is a real, unresolved property
> of the harness, and because it is what first showed that a one-question delta at
> N=20 could not carry a result.

Two runs of the **identical** thg_v1 command:

| run | EM | F1 | BERTScore-F1 | judge |
|---|---:|---:|---:|---:|
| first (artifact overwritten by §4.3) | 0.35 | 0.5961 | 0.9047 | 0.70 |
| second (artifact in §1) | 0.30 | 0.5762 | 0.9090 | 0.70 |

**`answer.temperature: 0.0` is configured and the W7/W8 wiring is in place, so
this variation should not occur.** It is reported as an unresolved finding.

**Consequence for reading §1:** a one-question difference at N=20 is **noise**.
The +0.05 judge delta in the ablation **is** one question. The first-run artifact
was destroyed by §4.3, so the two answer sets cannot be diffed — a limitation of
this evidence. **Any future single-run A/B at this N should be treated as
directional**, and a real claim needs either multiple seeds or a much larger N.

---

## 6. Limitations (carry verbatim)

* **L1** N=20 questions from **one** conversation (`locomo_0000`), the **first 20
  in file order** — not a random sample, not the full 199-pair set. No
  generalisation beyond it. **N was budget-limited by HOST MEMORY, not by
  design:** this is a **7.9 GB** machine with two long bank builds resident, and
  the Lead diagnosed that five of my earlier runs were killed by **OOM, not a job
  cap** (free RAM was 0.7–1.4 GB with four Python processes; a run loading torch +
  sentence-transformers + a FAISS index needs 1.5–2.5 GB). The Lead paused the
  resumable `gb_full` build to make room, and I serialised to **one heavy process
  at a time**. §1.2 later extends the pair to the **full 199 pairs**, so this
  limitation applies to §1/§1.1 only.
* **L2** Bank tag `thg_v2` is a **mixed-window** bank: `hyper_graph.py` changed on
  disk during conversations 0008–0009 while it was built. Its notes for
  `locomo_0000` (which these runs use) were ingested before that change, but the
  bank as a whole must not be described as a clean post-fix baseline. §1.1
  therefore **replicates both arms on `thg_v2_postfix`**, which is complete
  (10/10) and built from the frozen revision at hash `342ECB0A…` (verified before
  and after). **The replication reverses the sign of the `judge` delta** — see
  §1.1, and C1/C2 for the two confounds that still apply to cross-bank reading.
* **L2b** `thg_v2_postfix` contains **live `superseded_by` edges** (~121 pairs)
  even though the config intends supersession off — the truthiness bug at
  `asem/single_pass_ingest.py:174`, **fixed in `b4aec5e` after the bank was
  built**. Its numbers are therefore **not** "supersession-disabled" numbers, and
  a `thg_v2` vs `thg_v2_postfix` accuracy delta is confounded (note population
  **and** a live relation with a 22/22-wrong audited pair rate promoted at weight
  1.3). The **cost** axis is unaffected, and the §1.2 paired test is within one
  bank so it is not exposed to the cross-bank half of this confound.
* **L3** **No baseline system was run** (user directive). The "less context"
  claim rests on §2.1 + §2.2, **not** on a FullContext quality comparison; the
  quality-vs-FullContext axis is **unmeasured** (§3).
* **L4** §4.3 means the two arms' results collided under the default results
  directory; the numbers here come from isolated `--results-dir` runs.
* **L5** Run-to-run variance was observed at N=20 (identical command:
  `em 0.35 → 0.30`, `f1 0.5961 → 0.5762`, with `answer.temperature: 0.0`
  configured) and is **unresolved**. It is **no longer load-bearing**: the
  caution now rests on a **direct paired test at N=199** (§1.2, p = 0.21), not on
  the N=20 variance. `analysis/ws4_variance.py` exists to quantify the band but
  was **not run** — the Lead directed that the paired test supersedes it.
* **L6** §2.2's ratio comes from an in-code note list with a lexical ranker, so
  it is a context-builder measurement, not end-to-end retrieval cost.
* **L7** `session_recall` on LongMemEval is **not reported** — it is trivially
  1.00 on 100% of oracle items. `turn_recall` is **not reported** — the
  `has_answer` flags are frequently on the wrong turn. Neither is usable as
  evidence here.
* **L8** Pre-existing results under `data/benchmarks/results/**` from other code
  revisions are cited, if at all, **only** as clearly-labelled historical context
  (as in §3), never as evidence for the current revision.

---

## 7. Bottom line

### 7.1 What the user asked for — proven vs not proven

The user's request has three components. Splitting them is the point of this
section, so the claim boundary is unambiguous:

| # | what was asked | status | evidence |
|---|---|---|---|
| **i** | **less context than full history** | **PROVEN** | bounded **8 notes/question** vs **~15,687-token** full history; **16.491×** (LoCoMo) and **4.035×** (LongMemEval) median smaller, **0/120 inversions**, budget-matched at 32,192 tokens (§2) |
| **ii** | **still answers long-term-memory questions** | **PARTIALLY SHOWN** | `temporal` EM **0.4595**, judge **0.8649** (n=37); `single_hop` judge **0.7286** (n=70) — but these are **single-fact/ordering** slices, not the multi-hop case |
| **iii** | **multi-hop** | **NOT SHOWN** | `multi_hop` EM **0.0625** (2/32), judge **0.2500** vs legacy **0.3438** at N=199 — the method does **not** win the multi-hop slice at any n measured |

### 7.2 The ablation, stated exactly

* **EM ties: 0.3065 vs 0.3065 at N=199.**
* **judge leans to legacy on the clean bank at both N** — **0.618 vs 0.653 at
  N=199** (p = 0.21, **not significant**), and 0.65 vs 0.70 at N=20 on the same
  bank.
* **BERTScore is flat** — paired **median diff 0.0000**, 75/72/52, p = 0.87.
* **The context builder is NOT established as an improvement on answer quality at
  any N measured.** The F1/ROUGE-L tilt that briefly favoured thg_v1 at N=20
  **reverses at N=199** (legacy 0.4935/0.4836 vs 0.4851/0.4717), removing the
  last holdout for a "slightly positive" reading.
* **The cost axis therefore stands alone as the positive contribution.**

### 7.3 Method-side, verified

ASEM-THG answers at **judge 0.618–0.700** (coverage **1.0**, **0 failures in all
six arms — including 0/199 in both full-N arms**), BERTScore-F1 **0.8685–0.9090**,
F1 0.4851–0.5762, EM 0.3000–0.3500, depending on bank, context style and N.

### 7.4 Caveats that travel with every number above

* **Honest gap:** quality **versus FullContext is not measured here**, so the
  original two-axis claim is **not** proven by this report.
* **Confound:** `thg_v2_postfix` has **live `superseded_by`** edges despite the
  config intending them off (the truthiness bug fixed in `b4aec5e`, **after** the
  bank was built), and the audited pair rate for that relation is **22/22 wrong**
  at the **highest** ranking weight. Any ASEM-THG accuracy figure from that bank
  includes that effect — a **candidate explanation**, not proof of a
  retrieval-design failure.
* **Variance:** an identical command varied run-to-run at N=20 (§6 L5); this is
  **unresolved** but no longer load-bearing, since the caution now rests on the
  direct paired test at N=199.
* **Scale/labelling:** §1/§1.1 are N=20 (first 20 pairs); §1.2/§1.3 are the full
  199. Different question sets — not comparable as deltas.
* **Single conversation:** everything is `locomo_0000`. No claim generalises to
  the other nine conversations or to LoCoMo10 as a whole.
