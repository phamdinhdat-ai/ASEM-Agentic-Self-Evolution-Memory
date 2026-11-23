# CARE paper: review, novelty upgrade plan, contributions (16 Nov 2026)

Scope: review of `asem_v2.tex` (15 pp., Qwen3-8B, LoCoMo) against the Kaggle results gathered today (`analysis/CARE_kaggle_status_20261116.md`) and two new offline analyses on saved traces (no new model calls). The paper itself is **not modified**.

## 1. Review: where novelty is weak today

| # | Weakness | Evidence in the paper | Reviewer attack |
|---|---|---|---|
| W1 | Core mechanism is established | Sec. 2.2/2.3 itself says conditional retrieval and corrective reading are known (FLARE, Self-RAG, CRAG, Adaptive-RAG, MemR3, CueMem) | "What is new beyond a two-call CRAG on dialogue turns?" |
| W2 | No head-to-head against those gate families | "does not demonstrate superiority ... under a shared implementation" | Quality-cost frontier claim is unsupported |
| W3 | Repair hurts refusals and the paper does not fix it | -46 correct refusals (BM25), -68 (dense) | Authors diagnose a flaw in their own method and stop |
| W4 | Hand-coded gate checks look LoCoMo-specific | five holidays, "stopped" cue (Sec. 3.3) | Overfit to 10 public conversations |
| W5 | Development and test on the same 10 conversations | "not an independent generalization test" | Intervals are optimistic |
| W6 | Mechanism not identified | expansion, correction text, binding checks never toggled separately | No causal claim possible |
| W7 | "Source-anchored" support is unverified | 100-answer annotation queue has no labels | Citation IDs are not evidence of entailment |
| W8 | Disconnected from the thesis theme | no memory writes, no learning, "self-evolving" absent | Title/framing mismatch with ASEM |

## 2. New evidence gathered today (use in the paper)

**E1. LongMemEval cleaned-oracle, 308 of 500 questions (shards 0-2; shards 3-4 queued).** Answerable F1 0.3728 -> 0.4059, paired +0.0331, 95% CI [+0.0151, +0.0540], 13 better / 0 worse. Correct refusals 12 -> 8 of 14. Source-session recall 0.969 -> 0.927 (-0.042, CI excludes 0). The LoCoMo pattern repeats: **repair gains answers, costs refusals**. This is the first evidence outside the development data, and it supports W3 as a real, general phenomenon rather than a LoCoMo artifact.

**E2. Selection ceiling (LoCoMo, 1,986 q, saved traces).**
- Every one of the 1,004 repairs starts from a first-pass refusal; 419 flip to an answer: **279 gains** (score mass 178.6), **46 losses** (mass 46.0), 94 ties.
- All 46 losses are unsupported-premise (category 5); 40 came from "refuse + search query" triggers, 6 from invalid first passes. 29 of the 46 cite a turn that was *not* in the first bundle, so "cites new evidence" does not separate them.
- Oracle choice between first and final answer (uses gold, ceiling only): **0.5487**, i.e. +0.0231 over CARE's 0.5256.
- Parameter-free structural rules do not beat CARE: "accept only if final cites a new turn" gives 0.5190 (refusals 345, answerable 0.4452); "non-refusal only" and "risk-only" rules change nothing.
- **Break-even for any verifier:** rejecting a flip is profitable only if gains rejected <= 1.56 x losses caught (a gain is worth 0.64 on average, a loss costs 1.0). A verifier that catches half the losses (23) may wrongly reject at most ~36 of the 279 gains.

**E3. Where the remaining error sits (1,536 answerable questions with annotated sources).**

| Annotated sources in final bundle | Good (F1 >= .5) | Refused | Wrong answer |
|---|---:|---:|---:|
| All visible | 662 | 76 | 247 |
| Some visible | 52 | 42 | 117 |
| None visible | 54 | 127 | 159 |

Failures with sources **missing** from the bundle: 445; with all sources visible (reader/policy failures): 323. By category (missing vs visible): single-hop 178 vs 28, conversational 143 vs 184, temporal 66 vs 90, commonsense 58 vs 21. So multi-evidence retrieval coverage (single-hop list questions, F1 0.26) is the largest pool, while temporal and conversational questions are mostly reader-limited. Caveat: "visible" does not mean "sufficient", and F1 < 0.5 is a coarse failure cut.

**E4. Possible simplification (verify in code before claiming).** In the saved traces every repair has either an invalid first pass (137) or a non-empty search-query list (867); none is triggered by the risk flag alone. If confirmed, the risk term $\rho$ in Eq. (gate) never acted as a sole trigger and could be dropped or ablated.

**E5. Operations.** 40K full-history pilot failed on a vLLM KV-cache limit (fixed, see below). Chain `LongMemEval -> 40K pilot + shards -> Qwen3-4B` is running in the background.

## 3. Upgrade plan (ranked)

Key enabler, **one new run powers most of the plan: the two-pass table.** Run second-pass repair for the 982 questions CARE did not escalate (about +982 calls, ~1-2 GPU hours). Then every question has a saved first answer and a saved repaired answer, and *any* gate or acceptance policy becomes an offline selection over that table, with leave-one-conversation-out cross-validation, no further inference. It also gives the always-on repair arm the paper already lists as a next experiment.

### U1. Abstention-preserving repair acceptance (main method upgrade; addresses W3, W1)
- **Idea:** keep repair's gain but stop it overriding justified refusals. One short verifier call, only on the 419 refusal-to-answer flips (not all 1,004): "Do the cited turns state this answer to this question? / Is the question's premise supported?" Output accept or keep-refusal.
- **Cost:** about +419 short calls (~+3-4% tokens).
- **Hypothesis:** recovers a large share of the 46 losses while rejecting few of the 279 gains.
- **Success criterion (pre-register):** net aggregate > 0.5256 on LoCoMo, and the break-even rule in E2 satisfied; confirm on LongMemEval abstention questions (30 `_abs`). Ceiling is +0.0231, so frame the claim as "removes the abstention regression" not "large gain".
- **Falsifier:** verifier catches < 20 of 46 losses or rejects > 30 gains -> report as a negative result; the diagnosis (E1-E2) remains a contribution.
- **Side benefit:** the verifier's labels give the claim-level support measurement for W7.

### U2. Gate-family comparison on one reader (addresses W2, W1)
Arms, all selections over the two-pass table plus a few gate-judge calls: never repair; always repair; CARE gate; confidence gate (FLARE-style, vLLM token log-probs); sufficiency-judge gate (CRAG-style); reflection gate (Self-RAG-style prompt); the learned gate of U3. Report score versus calls and tokens as a Pareto frontier with conversation-cluster intervals. This is the missing "quality-cost frontier" claim; state clearly these are re-implementations under a shared reader, not the original systems.

### U3. Learned repair arbitration with utility feedback (links to ASEM; addresses W8)
- Features available at inference: first-answer refusal, |Q|, number and novelty of citations, risk cues, question-type tokens, bundle overlap, answer length, log-prob.
- Model: logistic regression or small gradient-boosted tree predicting final - first; train leave-one-conversation-out on LoCoMo, **freeze, then test on LongMemEval**.
- Optional ASEM tie-in: replace the offline fit by the Stage-5 EMA update, `q <- q + alpha (r - q)`, per feature bucket, keeping the backbone frozen. Be explicit that the reward needs feedback (benchmark labels or user signal) that is not free at deployment.
- Risk: ceiling is +0.0231 on LoCoMo; if the learned gate only matches U1, drop it.

### U4. Multi-evidence coverage (largest error pool, E3)
Test, offline first (source recall needs no GPU): BM25 + dense reciprocal-rank-fusion first bundle; **session-diverse window in the repair expansion** (your G3 result "the multi-hop fix is a session-diverse window, not a query planner" transfers here); per-facet use of the reader's two queries. Pre-screen on single-hop recall (currently weakest), spend GPU only on the best variant. Report single-hop and conversational F1 separately.

### U5. Mechanism ablation (addresses W6, W4)
Three arms on the 1,004 repaired questions: expansion only; correction text only; source-binding checks off. If the holiday and "stopped" checks contribute about zero, **remove them** and say so (fixes W4 and simplifies the method). Roughly 3 x 1,004 calls.

### U6. External validity and controls (addresses W5)
- Finish LongMemEval cleaned-oracle (500) - running.
- **Add LongMemEval-S (full haystack).** The oracle setting hands over relevant sessions (first-pass recall 0.97), so it barely tests retrieval; CARE's claim lives in large haystacks. Needs dataset download and a BM25 index over ~500 sessions.
- Qwen3-4B second backbone, 40K full-context control - queued.
- **Freeze code and commit a tag before any new run** (git tag + protocol file), so LongMemEval-S is a clean held-out test.

### U7. Claim-level support audit (addresses W7)
Label the frozen 100-answer queue (or a verifier-assisted subset checked by hand on a sample). Report support rate for CARE and baselines, separately for answerable and unsupported questions. Do not claim "grounded" before this exists.

## 4. Proposed contributions (target paper v3)

If U1 works:
1. **Diagnosis:** a decomposition of answer-time evidence repair into gain and abstention loss on two benchmarks, with a selection ceiling and a break-even rule (E1-E2).
2. **Method:** abstention-preserving conditional repair (gate + support-verified acceptance) over immutable, source-addressed turns.
3. **Controlled comparison:** gate-family Pareto frontier on one frozen reader with a reusable two-pass table (U2), plus matched-reader retrieval ablation (existing).
4. **Validation and measurement:** LoCoMo + LongMemEval(-S), two backbones, full-context control, claim-level support audit.

If U1 fails, keep contributions 1, 3, 4 and retitle toward "when does a second read help or hurt". Suggested title only if U1 succeeds: *CARE: Abstention-Preserving Conditional Evidence Repair for Long-Horizon Conversational Memory.*

Downgrade: the relation+temporal extension (no gain, 15 changed answers) moves to an appendix note.

## 5. Paper edits needed (section by section)

| Section | Change |
|---|---|
| Abstract | add LongMemEval result and the abstention finding; state the verifier outcome; drop "more specific test" hedging once U2 exists |
| Intro 1.3 | rewrite contributions per Section 4; keep scope sentence |
| Related work 2.2/2.3 | position against CRAG/Self-RAG/FLARE with the shared-reader comparison; add recent memory systems only with verified bib entries |
| 3.3-3.4 | simplify gate if E4 confirmed; add the acceptance step to Alg. 1 and Eq. (gate); state call count n_q = 1 + tau + verifier |
| 4 | add datasets (LongMemEval, -S), second backbone, pre-registration and freeze statement; revise the "development data" caveat per dataset |
| 5 Results | new subsections: two-pass table and ceiling; abstention-preserving acceptance; gate frontier; external transfer; multi-evidence coverage; ablation |
| Tables/figures | add frontier plot (score vs calls), failure decomposition by category (E3), repair gain/loss waterfall, LongMemEval table |
| 6 Discussion | move completed items out of "next experiments"; update "what remains unproven" |
| Appendix | reproducibility (kernel refs, SHA-256, schedulers), prompts including the verifier |
| Numbers | refresh abstract, conclusion and the 15-page count after edits |

## 6. Sequencing and cost

1. **Now (running):** LongMemEval shards 3-4 -> 40K pilot (KV fix, utilization 0.94) + shards -> Qwen3-4B.
2. **Offline, no GPU:** U4 retrieval pre-screen, feature extraction and cross-validation scaffolding, LongMemEval-S download and indexing.
3. **Small GPU jobs (hours):** two-pass table (+982 calls), U1 verifier (+419 calls), U5 ablations (~3,000 calls).
4. **Then:** U2 frontier, U3 learned gate, LongMemEval-S, U7 labels.
5. Write v3 only after 3-4, because contributions depend on whether U1 clears its pre-registered criterion.

## 7. Caveats

- The +0.0231 ceiling is on LoCoMo only; any acceptance rule is bounded by it there. The case for U1 rests on removing a documented regression and on LongMemEval, not on a large aggregate jump.
- E2-E3 use gold labels for analysis only; no rule above uses gold at inference.
- LongMemEval numbers are partial (308 of 500) and are diagnostic metrics, not official LLM-judged accuracy.
- The 0.94 GPU-utilization fix for the 40K run is untested on hardware until the pilot finishes.
