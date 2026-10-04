# ASEM-THG: TEMPORAL HYPER-GRAPH MEMORY FOR

# LARGE LANGUAGE MODEL AGENTS

## SINGLE-PASS TEMPORAL GROUNDING, DETERMINISTIC RELATIONAL STRUCTURE,

## AND GRAPH-AWARE VALUE RETRIEVAL

```
Dat Pham Dinh
```

> **Status of this draft.** This document *extends* the original ASEM paper
> (`ASEM_PAPER.md`) with the **ASEM-THG** method — the system actually implemented
> and measured in this repository. It re-states the inherited five-stage
> framework, supersedes the batch-construction and free-form-linking sections with
> the single-pass temporal-grounding design, and reports the **measured ablation
> evidence as it stands**, including the negative result. Every headline number
> below is traceable to an artifact via `analysis/`; nothing is extrapolated from
> prose.

---

## ABSTRACT

```
Large language model (LLM) agents remain fundamentally stateless across sessions:
knowledge acquired in one interaction is lost before the next, imposing a
structural ceiling on long-horizon adaptation. The original ASEM framework
addressed three root causes of this ceiling — semantic impoverishment, heuristic
write policies, and utility blindness — through a five-stage pipeline of
multi-attribute note construction, RL-driven write operations, dynamic linking
with bidirectional evolution, two-phase value-aware retrieval, and non-parametric
utility updates.

We present ASEM-THG (ASEM with a Temporal Hyper-Graph), a redesign that makes the
memory representation explicitly temporal and relational while preserving ASEM's
frozen-backbone, self-evolving contract. ASEM-THG introduces three shifts.
(i) Single-pass temporal grounding: a session is converted into atomic facts by
ONE LLM call that resolves every pronoun and every date to an absolute calendar
date, eliminating the O(turns) construction cost and the single largest class of
unanswerable temporal questions. (ii) A deterministic hyper-graph: four LLM-free
relation types — same-entity, temporal, semantic, and superseded-by — are built in
sub-millisecond time, with hub suppression by smoothed inverse document frequency
so the entity channel does not collapse into a clique. (iii) Graph-aware
value retrieval: the two-phase pipeline is extended with relation-type-weighted
N-hop traversal and query-seeded Personalized PageRank over the shared graph.

We report a one-variable ablation of the context builder at n=199 (LoCoMo10) and
n=500 (LongMemEval), both at judge coverage 1.00: the ablation is a NEGATIVE that
replicates across both datasets (paired binomial p = 0.21 and 0.29 respectively;
EM ties exactly). We state this result as the primary finding and separate it from
the graph-construction diagnostics that ARE established — a 19:1 hub-to-signal
ratio in the entity channel, and the measurement that fan-out capping does not
reduce hub dominance while IDF weighting does. The contribution of this paper is
therefore both a method and a rigorously bounded account of what its relational
structure does and does not buy on long-horizon QA.
```

---

## 1 INTRODUCTION

### 1.1 THE STATELESSNESS PROBLEM (unchanged from ASEM)

The deployment of LLM agents in multi-session environments exposes a fundamental
architectural mismatch: agents operate under a hard context-window bound and begin
every interaction from a blank slate. For tasks that depend on information spanning
interactions — tracking evolving preferences, resolving references to entities
mentioned weeks earlier, building a coherent model of a long-running project — this
is a structural barrier, not an inconvenience.

Human cognition resolves the analogous problem through the interplay of episodic
and semantic memory: episodes are distilled into generalised structures, and a
constructive, *selective* retrieval process reconstructs only what the current
demand requires. Replicating this in LLM agents requires solving four interlocking
sub-problems:

- **P1 — Semantic enrichment.** Raw logs are redundant and impoverished; retrieval
  over them is dominated by lexical overlap.
- **P2 — Principled write selection.** Append-only or recency-based policies
  fragment complementary facts and accumulate contradictions.
- **P3 — Relational structure.** Multi-hop reasoning needs explicit cross-references,
  not a flat store.
- **P4 — Utility-aware retrieval.** Semantic proximity is weaker than functional
  value.

### 1.2 WHY TEMPORALITY AND DETERMINISM ARE THE MISSING AXES

The original ASEM framework satisfies P1–P4 on paper, but the deployed
implementation exposed two gaps that the abstract-level treatment did not name:

1. **Temporal grounding is a first-class retrieval requirement, not a metadata
   detail.** Long-horizon conversational QA contains questions that *subtract*
   dates ("how many days between X and Y"). A note that retains a relative or
   shorthand date ("on 3/1", "last week") is unusable no matter how well it is
   embedded, linked, or value-ranked. The date must be resolved **at write time**,
   against the session date, because the resolution context is unavailable later.
2. **Relational structure must be deterministic to be evaluable.** Free-form
   LLM-generated links make a bank non-reproducible: the same session yields a
   different graph on every ingest, so a frozen-bank comparison is impossible.
   A graph that cannot be rebuilt byte-for-byte cannot be ablated.

ASEM-THG addresses both by moving temporal resolution into the extraction prompt
and by building the graph's four relation types **without any additional LLM calls**.

### 1.3 CONTRIBUTIONS

**(1) Single-pass, temporally grounded atomic ingestion (extends P1).**
`SinglePassSessionIngestor` converts an entire dialogue session into atomic,
standalone facts in **one** LLM call, under five enforced extraction rules:
pronoun resolution, **absolute-date resolution for every dated event**, one fact
per note, an atomic `(subject, predicate, object)` triplet per fact, and explicit
capture of reactive/attribution turns with a `speaker` field. Construction cost
becomes O(1) LLM calls per session instead of O(turns), and temporal questions
become answerable from note content alone.

**(2) A deterministic temporal hyper-graph (extends P3).**
`TemporalHyperGraph` builds four LLM-free relation types —
`same-entity`, `temporal`, `semantic`, `superseded_by` — in sub-millisecond time
over persisted state, and maintains **two synchronised views**: an in-memory
networkx graph with explicit entity nodes (walked by PPR) and the persisted
note-to-note projection on `Note.L` (traversed by the retriever). Determinism
(temperature pinned + JSON-schema-guided decoding) makes frozen-bank ablation
possible.

**(3) Hub-aware entity selectivity.**
We diagnose that the entity channel is dominated by a *class* of high-degree
tokens (speaker names in LoCoMo; the role token `user` in LongMemEval), and show
by measurement that **fan-out capping is not the lever** — removing 96.1% of edge
volume moves the hub share by 0.67 pp — whereas **smoothed IDF weighting** is, and
its reach is bounded (only 6.3% of notes have a rare peer in their top-5). This is
a negative-space contribution: it rules out the obvious fix.

**(4) Graph-aware value retrieval (extends P4).**
The two-phase pipeline is extended with relation-type-weighted N-hop traversal
(`superseded_by` 1.3, `contradicts` 1.2, … `semantic` 0.6) and query-seeded
Personalized PageRank over the shared graph, with a relevance floor that prevents
dense-graph appends from evicting gold notes.

**(5) A rigorously bounded empirical account.**
A one-variable ablation of the context builder at n=199 and n=500, judge
coverage 1.00 in every arm, with paired significance tests and an explicitly
reported **negative** result that replicates across two datasets.

---

## 2 THE ASEM-THG METHOD

ASEM-THG is the five-stage ASEM pipeline with Stages 1 and 3 re-specified for
temporal grounding and deterministic structure, and Stage 4 extended with graph
traversal. The full architecture (with component-, ingestion-, retrieval-, and
end-to-end sequence diagrams) is documented separately in
[`docs/ASEM-THG_architecture.md`](ASEM-THG_architecture.md); this section states
the method as a research contribution.

### 2.1 Notation

The bank at time $t$ is $\mathcal{M}_t=\{m_i\}_{i=1}^{N_t}$. In ASEM-THG the note
tuple of Eq. (1) is extended with five temporally-grounded metadata fields:

$$
m_i = (c_i, t_i, K_i, G_i, X_i, e_i, L_i, z_i, q_i,\; s_i, d_i, \tau_i, E_i, \psi_i),
\tag{1$'$}
$$

where $s_i$ is the session identifier, $d_i$ the human-readable session date,
$\tau_i$ an ISO-8601 absolute timestamp, $E_i$ the named entities, and $\psi_i$ the
speaker. All five are optional; when absent the note degrades gracefully to
Eq. (1). They are populated at ingestion and consumed by the temporal and entity
retrieval channels.

### 2.2 Stage 1 — Single-pass, temporally grounded note construction

**Motivation (P1, temporal).** A memory stored as raw tokens is redundant and
impoverished, *and* any date it records in relative form is unusable at query
time. Both defects are fixed in the same step.

**Procedure.** For a session with date $d_s$ and turns $\{u_1,\dots,u_T\}$, a single
call to the frozen backbone with prompt $P_1^{\text{sp}}$ yields an array of atomic
facts:

$$
\{(c_j, K_j, G_j, E_j, \psi_j, \text{sub}_j, \text{pre}_j, \text{obj}_j)\}_{j=1}^{F}
= \mathrm{LLM}\big(\{u_t\}\;\|\;d_s\;\|\;P_1^{\text{sp}}\big).
\tag{4$'$}
$$

Each fact becomes a note $m_j$ with $e_j = f_{\text{enc}}([\text{concat}(c_j,K_j,G_j,X_j)])$,
$z_j = f_{\text{enc}}(c_j)$, $q_j = q_0$, stamped with the **session** datetime
(not wall clock) and with $s_j, d_j, \tau_j, E_j, \psi_j$.

**Five enforced extraction rules.** The prompt $P_1^{\text{sp}}$ is the method's
core artifact and encodes:

$$
\begin{aligned}
&\text{(R1) resolve pronouns} && "\text{I went there}" \to "\text{Caroline went to Hawaii}" \\
&\text{(R2) resolve dates} && "\text{on 3/1}" \to "\text{on 1 March 2022}",\;
   "\text{yesterday}" \to d_s-1 \\
&\text{(R3) one fact per note} && \text{no thematic merging; past and plan are separate} \\
&\text{(R4) atomic triplet} && (s,p,o) \text{ for deterministic versioning (Stage 3)} \\
&\text{(R5) attribution} && \text{reactive turns kept with a } \texttt{speaker} \text{ field}
\end{aligned}
$$

Rule R2 admits exactly one date per fact: a turn stating two dated events
("started on 2/15 … saw a house on 3/1") is split into two notes, so each event
pairs with exactly one date and **difference-of-dates questions remain computable**.

**Determinism contract.** The call pins `ingestion.temperature` and may request
`response_format = json_schema`, so a bank built twice is byte-comparable. Malformed
outputs are recovered by a format-correction retry (`LLMRetryHandler`) before the
per-turn fallback is used — the fallback discards triplets and therefore the
entity/versioning edges.

### 2.3 Stage 2 — RL-driven write operations (unchanged)

The Memory Manager selects $o \in \{\text{ADD},\text{UPDATE},\text{DELETE},\text{NOOP}\}$
and, where applicable, a target $m'$, trained with GRPO against the exact-match
reward of a frozen Answer Agent:

$$
\mathcal{J}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\rho(\theta_i)A_i
- \beta\, D_{\mathrm{KL}}[\pi_\theta \,\|\, \pi_{\text{ref}}]\right].
\tag{8}
$$

ASEM-THG adds a **deterministic Tier-0 write gate** in front of this stage: only
the ambiguous similarity band ($\tau_{\text{redund}} \ge \text{sim} > \tau_{\text{high}}$)
pays the Memory-Manager LLM cost; clearly-novel turns go straight to ADD and
near-duplicates to NOOP. This bounds the S2 prompt regardless of bank size.

### 2.4 Stage 3 — Deterministic temporal hyper-graph

**Motivation (P3, determinism).** Free-form linking is non-reproducible. ASEM-THG
replaces it with four relation types computed **without any LLM call**.

**Relation types.**

$$
L_i \;\supseteq\; \underbrace{E_i}_{\text{same-entity}}
\;\cup\; \underbrace{T_i}_{\text{temporal}}
\;\cup\; \underbrace{S_i}_{\text{semantic}}
\;\cup\; \underbrace{V_i}_{\text{superseded-by}}.
\tag{10$'$}
$$

| Relation | Rule | Guard |
|---|---|---|
| `same-entity` | peers sharing any entity key, ranked by weighted co-occurrence + cosine | fan-out cap $L^{\text{ent}}_{\max}$; IDF floor |
| `temporal` | chronological neighbours within one session | session-local |
| `semantic` | $\cos(e_i,e_j) \ge \tau$ ($\tau=0.70$) | top-$L^{\text{sem}}_{\max}$ |
| `superseded_by` | restatement, **or** single-valued slot whose object changed | object must differ; predicate must be single-valued |

All edges are emitted **symmetrically**, because the retriever only follows
outgoing links from a seed; a one-directional edge is invisible when the seed is
the older endpoint.

**Two synchronised views.** Every relation is written to *both* an in-memory
networkx graph with explicit entity nodes (PPR walks this) and the persisted
note-to-note projection on `Note.L` (the retriever traverses this). A relation that
lived only in the in-memory graph would be silently lost across the process
boundary between ingestion and evaluation.

**Versioning.** A fact is superseded in two distinct ways:

$$
\underbrace{(s,p,o)\ \text{seen again}}_{\text{restatement: exact key}}
\qquad\text{vs.}\qquad
\underbrace{(s,p)\ \text{with } o \to o',\ p \text{ single-valued}}_{\text{replacement: slot key}}.
$$

Keying on $(s,p)$ alone conflates unrelated facts sharing a verb ("bought a car"
vs. "bought a house"); keying on $(s,p,o)$ alone misses replacement entirely. ASEM-THG
uses an exact key for restatement and a predicate-slot key for replacement,
restricted to genuinely single-valued predicates, with a guard that the object
must actually differ.

**Supersession defaults OFF.** Measured on the frozen 365-note LoCoMo bank, the
supersession relation was **0 of 5 pairs correct** — each marked a still-true fact
obsolete, one replacing a richer note with a lossier paraphrase — while the
retriever promotes `superseded_by` at the *highest* weight (1.3). It is therefore
retained as an explicit ablation toggle, not a default.

**Hub suppression.** A dominant entity (the speaker name; or the role token
`user`) is shared by most notes, so an unweighted entity channel links every note
to every other — a clique. The design down-weights a shared entity by its smoothed
IDF rather than severing it, preserving the rare-entity tail. §4.3 measures this.

### 2.5 Stage 4 — Graph-aware value retrieval with distillation

**Phases A–B (unchanged).** Phase A recalls a candidate pool
$\mathcal{C}(q)=\mathrm{TopK}_{k_1}\{i \mid \mathrm{sim}(e_q,e_i)>\delta\}$; Phase B
re-ranks by the z-score-normalised composite

$$
\mathrm{score}(q,m_i) = (1-\lambda)\,\widehat{\mathrm{sim}(e_q,e_i)}
+ \lambda\,\widehat{q(z_i,m_i)}.
\tag{14}
$$

Z-score normalisation within $\mathcal{C}(q)$ is essential: without it the composite
is dominated by whichever component has larger absolute magnitude.

**Phase C — relation-weighted N-hop traversal (new).** Links are followed
recursively to depth $H$ (default 2) with decay $\gamma^h$, and each traversed edge
is weighted by its relation type:

$$
w(\text{superseded\_by})=1.3,\;
w(\text{contradicts})=1.2,\;
w(\text{extends})=w(\text{causal})=1.1,\;
w(\text{same-entity})=0.9,\;\dots,\;
w(\text{semantic})=0.6.
\tag{14$'$}
$$

A minimum-similarity floor ($\mathrm{sim}\ge 0.30$ against the query) prevents
relevance-blind appends from evicting gold notes when the graph is dense.

**Phase D — query-seeded PPR (new).** Global PageRank is query-independent: it
rewards notes central to the whole bank, not evidence reachable from the query.
ASEM-THG adds a *query-conditional* signal — Personalized PageRank seeded on the
Phase A+B winners over the shared graph:

$$
\text{ppr}(m) = \mathrm{PPR}_{\alpha}\big(G,\; \text{seeds}=\text{TopK}_{k}(A{+}B)\big),
\tag{14$''$}
$$

which is the mechanism that surfaces the *other half* of a multi-hop fact (the
premise note sharing no query terms with the question). The retriever and the
ingestor MUST share one graph instance, or PPR walks an empty structure — a
fail-silent bug we document.

**Distillation.** $|\mathcal{M}_{\text{ret}}|$ notes are passed to the Answer Agent,
which selects the minimal sufficient subset and answers:

$$
(\mathcal{M}_{\text{distil}}, \hat{y}) \leftarrow \pi_\phi(q, \mathcal{M}_{\text{ret}}).
\tag{16}
$$

**One answer budget for every system.** Answering is a single model call, and
ASEM-THG trims history turns, retrieved notes, and note payload to
`answer.context_window − answer.max_tokens − safety` via a shared
`asem/token_budget.py`. This is what makes a Head-to-Head comparison against
FullContext a comparison of *retrieval*, not of context length.

### 2.6 Stage 5 — Non-parametric runtime utility update (unchanged)

EMA over outcome reward, with experience consolidation:

$$
q_i \leftarrow q_i + \alpha\,(r - q_i), \qquad \alpha = 0.10.
\tag{17}
$$

### 2.7 Recovery loop (new)

If the first answer is an abstention, the pipeline runs **one** widened retrieval
(larger $k_2$, lower $\delta$, more link neighbours), re-sorts the merged pool by
query similarity, and re-asks with an explicit "your first refusal was too hasty"
notice. Because abstention is roughly a third of "I don't know" answers *even when
the gold note is in context*, this one cheap retry is high-leverage; it is accepted
only if the second answer is still not a refusal, so a genuinely absent fact still
ends as "I don't know".

---

## 3 RELATION TO PRIOR WORK

ASEM-THG inherits ASEM's positioning (Table 1 of the original draft: Rich / Write /
Link / Util / Evolve). Its added axis is **temporal grounding + determinism**.

| System | Rich | Write | Link | Util | Evolve | **Temporal grounding** | **Deterministic graph** |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| MemGPT | × | ◦ | × | × | × | × | — |
| MemoryBank | × | ◦ | × | × | × | × | — |
| Mem0 | ◦ | ◦ | ◦ | × | × | ◦ | × |
| MemoryOS | ◦ | ◦ | × | × | × | ◦ | × |
| Atomic-Linking | ✓ | × | ✓ | × | ◦ | ◦ | × |
| RL-Manager | × | ✓ | × | × | × | × | × |
| Value-Retrieval | ◦ | × | × | ✓ | ✓ | ◦ | × |
| **ASEM-THG (ours)** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓ absolute dates** | **✓ LLM-free edges** |

The distinguishing claims are: (a) dates are resolved at write time so
difference-of-dates questions are answerable from content alone, and (b) the
relation graph is built with **zero** LLM linking calls and is byte-reproducible,
which is what makes the ablation in §4 possible at all.

---

## 4 ABLATION STUDY

> **Honesty preamble (R1 discipline).** Every number in this section is printed by
> a script in `analysis/` reading raw artifacts — the printed-ablation table
> (`analysis/ws4_ablation_table.py`), the paired tests
> (`analysis/ws4_paired_significance.py`, `analysis/longmemeval_analysis.py`), and
> the graph diagnostics (`analysis/graph_metrics.py`, `hub_only_*.py`,
> `supersession_audit.py`). No headline number is arithmetic on another headline
> number, and no number is copied from prose. Judge scores are reported **only with
> their coverage**; `judge_coverage` is **1.00 in every arm**, with **0 judge
> failures**, including 0/199 and 0/500 at full scale.

### 4.1 Design

A one-variable ablation of the **context builder** (`answer.context_style` ∈
{`thg_v1`, `legacy`}). The two arms differ in **exactly 1 of 60 config keys**
(machine-verified in `analysis/_ablation_config_diff.json`); bank, system,
conversation, metrics, judge config, and judge workers are identical. Two
datasets, two sample sizes.

### 4.2 Primary result — the ablation is a NEGATIVE that replicates

**All six LoCoMo arms** (`thg_v2_postfix`, read-only frozen bank, 4,664 notes):

| bank | arm | n | EM | F1 | ROUGE-L | BERTScore-F1 | judge | judge fail |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `thg_v2` | thg_v1 | 20 | 0.3000 | 0.5762 | 0.5429 | 0.9090 | 0.7000 | 0/20 |
| `thg_v2` | legacy | 20 | 0.3000 | 0.5675 | 0.5342 | 0.9006 | 0.6500 | 0/20 |
| `thg_v2_postfix` | thg_v1 | 20 | 0.3500 | 0.5543 | 0.5209 | 0.8925 | 0.6500 | 0/20 |
| `thg_v2_postfix` | legacy | 20 | 0.3500 | 0.5093 | 0.4759 | 0.8937 | 0.7000 | 0/20 |
| **`thg_v2_postfix`** | **thg_v1** | **199** | **0.3065** | 0.4851 | 0.4717 | 0.8702 | **0.6181** | **0/199** |
| **`thg_v2_postfix`** | **legacy** | **199** | **0.3065** | 0.4935 | 0.4836 | 0.8685 | **0.6533** | **0/199** |

**Paired test, n=199** (both arms answer the same 199 questions, paired by `idx`):

```
both correct 115 | thg_v1 only 8 | legacy only 15 | neither 61
judge accuracy  thg_v1 0.6181 (123/199)   legacy 0.6533 (130/199)
discordant 23   exact two-sided binomial p = 0.2100   -> NOT significant
BERTScore  paired mean diff +0.0017, median 0.0000, 75/72/52 -> p = 0.8690
```

**LongMemEval replication, n=500** (0 judge failures in 500):

| arm | n | EM | F1 | ROUGE-L | BERTScore-F1 | judge |
|---|---:|---:|---:|---:|---:|---:|
| `thg_v1` | 500 | 0.2740 | 0.4166 | 0.4092 | 0.8833 | 0.5880 |
| `legacy` | 500 | 0.2680 | 0.4232 | 0.4131 | 0.8860 | 0.6140 |

```
both correct 236 | thg_v1 only 58 | legacy only 71 | neither 135
discordant 129   exact two-sided binomial p = 0.2907   -> NOT significant
```

**Read plainly.** EM ties exactly on LoCoMo; judge leans to `legacy` on both
datasets; BERTScore is flat. A +0.05 judge win at N=20 on the mixed-window bank
**reverses sign on the clean bank at both N**, and the F1/ROUGE-L tilt reverses at
N=199. **The context builder is not established as an improvement on answer
quality at any N measured.** We report this as the headline rather than bury it,
because a single-bank positive would not have survived a second bank.

**What the negative does NOT mean.** It is an ablation of *one rendering variable*,
not of the graph. It bounds the claim: the graph's value, if any, must be shown
through a graph-specific intervention (§4.4), not through context rendering.

### 4.3 Established graph-construction findings (the positive space)

These are measured, reproducible, and independent of the answer-quality null.

**(a) The entity channel is ~95% hub and ~19:1 noise.** Corpus-wide,
`hub_only_edge_fraction = 0.9474`: of 22,565 `same-entity` records, 21,378 are
hub-only and **1,187 discriminations** — a **19:1** noise-to-signal ratio.

**(b) The hub is a *class* of token, not one token.** LoCoMo's hubs are speaker
names (`caroline` 65%, `melanie` 56% fan-out); LongMemEval's is the role noun
`user` (67%). Two corpora reach ~65% by different routes, so **no hardcoded
stop-set generalises**; a frequency-based (IDF) mechanism is the only one that
covers both.

**(c) Fan-out capping is not the lever; IDF weighting is.** Capping reduced
`locomo_0000` from 46,084 to 1,792 stored edges (**25.7×**) while hub share moved
only 99.94% → 99.27% — a **0.67 pp** change from removing 96.1% of edge volume.
This is a negative-space finding: it rules out the obvious fix.

**(d) IDF weighting's reach is bounded.** It demonstrably promotes rare-sharing
peers where they exist (witness `00cb74d3`: rank #5 → #1, score 0.6589 → 3.8245,
displacing 4 hub-only peers), **but only 23/365 notes (6.3%)** have any rare peer
in their top-5, and the corpus top-5 split is unchanged at 1769/52. 6.3% ≈ 5.3%
of the channel carrying signal: two independent measurements agreeing. **Claim
bounded to "promotes rare peers where they exist", never "improves discriminative
ranking".**

### 4.4 Ablation matrix (component → status)

| # | Component ablated / tested | Variable | Status | Evidence |
|---|---|---|---|---|
| A1 | Context-builder style | `answer.context_style` | **NULL** (both datasets) | §4.2 |
| A2 | Supersession relation | `hyper_graph.supersession` | **OFF by default** (0/5 correct) | `supersession_audit.py` |
| A3 | Entity fan-out cap | `max_entity_links` | **Ineffective** (0.67 pp) | §4.3c |
| A4 | Hub IDF floor | `entity_idf_floor` | **Bounded** (6.3% reach) | §4.3d |
| A5 | Entity-node hygiene | `entities` vs `entities ∪ keywords` | designed, **not yet ablated** | roadmap P1-1 |
| A6 | Query-seeded PPR | `retriever.ppr_weight` | wired, default 0.15 | §2.5 |
| A7 | Bidirectional PPR | `retriever.ppr_reverse` | wired, default OFF | §2.5 |
| A8 | Relation-type weights | `relation_weights` | wired (`superseded_by` 1.3) | §2.5 |
| A9 | Tier-0 write gate | `write_gate.tau_high/tau_redund` | wired, default ON | §2.3 |
| A10 | Recovery loop | `recovery_enabled` | wired, default ON | §2.7 |

**A1, A3, A4 and A2 are the measured ones**; A5–A10 are engineered but await a
clean ablation run and are reported as such — not as established gains.

### 4.5 Per-category weakness (measured at large n)

LongMemEval oracle, `thg_v1` arm at n=500 — the weakness is an *inference-heavy*
one, consistent across the project:

| question_type | n | EM | judge |
|---|---:|---:|---:|
| multi-session | 133 | 0.1729 | **0.4586** |
| temporal-reasoning | 133 | 0.1353 | **0.4887** |
| knowledge-update | 78 | 0.3333 | 0.7051 |

Absolute dates help knowledge-update (judge 0.71) but do **not** rescue
multi-hop/temporal *reasoning* (judge < 0.50), which is the honest weak spot and
the target of A5/A6.

---

## 5 LIMITATIONS

1. **The context-builder ablation is null and replicates.** No answer-quality
   claim for the THG context style survives n=199 or n=500.
2. **Graph value is not yet isolated by a graph-specific ablation.** A6–A8 change
   retrieval directly but have not been run against matched controls; until they
   are, the graph's contribution to answer quality is **unproven**.
3. **Hub suppression is bounded** (6.3% reach) and must not be reported as a
   corpus-level improvement.
4. **Supersession is off by default** because measured precision was 0/5 — the
   mechanism is unsound as implemented, not merely low-value.
5. **Ingestion determinism depends on backend support** for per-call temperature
   and structured output; a backend that ignores both yields a non-reproducible
   bank, and the ablation loses its controls.

---

## 6 WHAT THIS LEAVES STANDING

The defensible positive core of ASEM-THG is **representation and cost**, not
answer-quality delta:

- **Temporal grounding at write time** — dates are absolute in the note, so
  difference-of-dates questions are computable from content (knowledge-update judge
  0.71).
- **O(1) LLM calls per session** for extraction, plus sub-millisecond LLM-free
  graph construction — a large cost reduction against per-note/per-pair linking.
- **Deterministic, dual-view graph** — reproducible banks, which is the
  precondition for *any* frozen-bank ablation.
- **A reproducible negative** that rules out context rendering as the lever and
  bounds the entity channel to 6.3% signal — narrowing the search space for the
  next iteration to multi-hop traversal (A5/A6).

---

## APPENDIX A — REPRODUCIBILITY

```powershell
# Environment (Windows host): memory-r1 conda env, explicit interpreter
$py = "C:\Users\Dat Pham\anaconda3\envs\memory-r1\python.exe"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"; $env:PYTHONPATH = "."

# A1 — context-builder ablation, full N=199 (LoCoMo10)
& $py scripts/run_static_eval.py --tag thg_v2_postfix `
    --input datasets/locomo/locomo10.json `
    --config configs/models/deepseek_v41_flash_thg_v1.yaml --systems ASEM-THG `
    --conversations locomo_0000 --metrics em f1 rougeL bertscore_f1 judge `
    --judge-config configs/models/judge_v41_flash.yaml --judge-workers 6 `
    --results-dir data/benchmarks/results/ws4_ablation/full199_postfix_thgv1
# legacy arm: identical except --config configs/models/deepseek_v41_flash.yaml

# Printed tables and tests
& $py analysis/ws4_ablation_table.py
& $py analysis/ws4_paired_significance.py
& $py analysis/longmemeval_analysis.py --tag-thgv1 thg_v1_lme --tag-legacy legacy_lme

# Graph diagnostics
& $py analysis/graph_metrics.py
& $py analysis/hub_only_corpus.py --root static/memory_banks/locomo10/thg_v2/ASEM-THG
& $py analysis/supersession_audit.py --bank static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000/asem_thg.sqlite
```

**Frozen revision under test:** `asem/hyper_graph.py` SHA-256
`342ECB0A3A57BF600EB5C74AF6D589C1673403EBBE17DBB961BE2015D7FBC2E3`, re-verified
unchanged before and after the arms. Test suite at the freeze: **345 passed, 1
skipped, 0 failed**.

**Banks:** `thg_v2` (4,579 notes / 73,862 links), `thg_v2_postfix` (4,664 / 73,640),
both built 10/10 conversations with 0 errors; LongMemEval oracle is per-question
(private haystacks, reset each question).
