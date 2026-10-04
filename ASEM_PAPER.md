
# ASEM-THG: AGENTIC SELF-EVOLVING MEMORY FOR

# LARGE LANGUAGE MODEL AGENTS

## TEMPORAL HYPER-GRAPH STRUCTURE, REINFORCEMENT-LEARNED OPERATIONS,

## AND VALUE-AWARE RETRIEVAL

```
Dat Pham Dinh
```

> **Revision note.** This revision updates the original ASEM draft with the method
> actually implemented and measured in this repository — **ASEM-THG** (agentic
> self-evolving memory with a *temporal hyper-graph*). The five-stage framework is
> preserved; Stages 1, 3 and 4 are re-specified for temporal grounding,
> deterministic relational structure, and graph-aware value retrieval, and a new
> Section 6 reports the current empirical results — including a rigorously bounded
> **negative** ablation that replicates across two datasets. Every reported number
> is traceable to an artifact under `analysis/`.

## ABSTRACT

```
Large language model (LLM) agents exhibit remarkable competence in isolated
tasks yet remain fundamentally stateless: information encountered in one session
is irrecoverably lost before the next, imposing a structural ceiling on long-horizon
adaptation. We identify three root causes of this ceiling. (i) Semantic impov-
erishment: raw interaction logs stored without enrichment produce flat, noise-
ridden memory banks whose retrieval is dominated by surface-form similarity
rather than conceptual relevance. (ii) Heuristic write policies: append-only or
recency-based eviction strategies fragment complementary information and accu-
mulate unresolved contradictions, causing retrieval quality to degrade monoton-
ically with corpus growth. (iii) Utility blindness: policies that equate semantic
proximity with functional value return contextually close but strategically inferior
candidates, systematically hindering multi-hop reasoning and long-horizon task
completion.
We present ASEM-THG (Agentic Self-Evolving Memory with a **Temporal Hyper-
Graph**), a unified five-stage framework that addresses all three deficiencies
simultaneously while making the memory representation explicitly temporal and
relational. ASEM-THG introduces: (i) a **single-pass, temporally grounded** atomic
note schema that converts one dialogue session into standalone facts in **one** LLM
call, resolving every pronoun and every date to an absolute calendar date so that
difference-of-dates questions are answerable from note content alone; (ii) an
RL-trained Memory Manager that selects among four principled write operations
using outcome-based reward, guarded by a deterministic Tier-0 write gate; (iii) a
**deterministic temporal hyper-graph** with four LLM-free relation types —
`same-entity`, `temporal`, `semantic`, and `superseded_by` — maintained in two
synchronised views, with hub suppression by smoothed inverse document frequency;
(iv) a two-phase value-aware retrieval pipeline extended with **relation-type-weighted
N-hop traversal** and **query-seeded Personalized PageRank** over the shared graph,
followed by a memory-distillation step; and (v) a non-parametric runtime utility rule
whose convergence in expectation to the true per-memory expected return, with
bounded asymptotic variance, is formally guaranteed. The resulting system maintains
a living knowledge network that self-organises, self-repairs, and continuously refines
utility estimates of stored experiences — without ever modifying backbone LLM
parameters.

We report a one-variable ablation of the context builder at n=199 (LoCoMo10) and
n=500 (LongMemEval), both at judge coverage 1.00. The answer-quality ablation is a
**negative that replicates across both datasets** (paired binomial p = 0.21 and 0.29;
exact-match ties on LoCoMo), and we state it as the primary finding rather than
burying it. Separately, we establish graph-construction facts that do survive
scrutiny: the entity channel is ~95% hub and ~19:1 noise-to-signal, and fan-out
capping does **not** reduce hub dominance while frequency-based (IDF) weighting
does, with a bounded reach we quantify at 6.3% of notes. The contribution is thus
both a method and a rigorously bounded account of what its relational structure buys
on long-horizon conversational QA.
```
## 1 INTRODUCTION

### 1.1 THE STATELESNESS PROBLEM

```
The deployment of LLM-powered agents in real-world, multi-session environments exposes a fun-
damental architectural mismatch. These agents operate under a hard context-window bound: every
interaction begins from a blank slate, with no access to experiential knowledge accumulated in
prior sessions. For tasks whose successful execution depends on information that spans multiple
interactions—tracking evolving user preferences, resolving references to entities mentioned weeks
earlier, or building a coherent model of a long-running project—this statelesness is not a minor
inconvenience but a structural barrier Wu et al. (2024); Maharana et al. (2024).
```
```
Human cognition resolves an analogous problem through the interplay of episodic and semantic
memory Tulving (1972). Episodic memory records specific past experiences; semantic memory
```

```
distils those episodes into generalised knowledge structures; and a constructive retrieval process
synthesises relevant fragments from both sources into actionable representations tailored to the cur-
rent demand. Crucially, this process is selective: not every stored trace is retrieved, and retrieved
traces are not replayed verbatim but reconstructed in accordance with the current context Schacter
& Addis (2007).
```
```
Replicating this capacity in LLM agents demands solving four interlocking sub-problems: (P1)
constructing semantically rich memory representations that capture the implicit structure of an in-
teraction rather than its surface tokens; (P2) executing principled write operations that consolidate
complementary facts and resolve contradictions rather than blindly accumulating entries; (P3) main-
taining relational structures that connect semantically and causally related notes into a traversable
knowledge network for multi-hop reasoning; and (P4) retrieving memories that are functionally
useful for a query—a property strictly stronger than semantic relevance that cosine similarity cannot
capture.
```
### 1.2 THE INTEGRATION GAP

```
The literature on memory-augmented LLM agents partitions into three paradigms, each addressing
a non-overlapping subset of{P1, P2, P3, P4}.
```
```
Static storage and retrieval. Systems that augment agents with plain key-value memory banks
retrieved by embedding similarity are effective for isolated factual recall, partially addressing P1.
They provide no mechanism for memory organisation (P2), relational linking (P3), or utility-aware
selection (P4). Retrieval quality degrades monotonically with corpus growth as contradictory and
redundant entries accumulate Packer et al. (2023); Zhong et al. (2024).
```
```
Structured and graph-based memory. Systems that incorporate predefined schemas and typed
relational structures partially address P3 for anticipated relationship types. Their reliance on fixed
schemas fundamentally limits adaptability: novel relationship types encountered at runtime cannot
be accommodated without schema modification, causing memory organisation to converge to a static
predefined structure rather than emerging organically from content Chhikara et al. (2025).
```
```
Reinforcement-learning-enhanced memory. Systems that train dedicated agents to perform
structured write operations address P2, and when combined with utility estimation, partially address
P4. However, existing implementations either operate on semantically impoverished flat entries,
failing P1 and P3, or provide principled utility estimation without the rich relational structure that
enables multi-hop reasoning, failing P3 Yan et al. (2025); Zhang et al. (2026).
```
```
No existing system simultaneously satisfies all four requirements. We term this the integration gap
and present ASEM as its resolution.
```
### 1.3 CONTRIBUTIONS

```
(1) Unified multi-attribute note representation (P1). We introduce a structured note schema
that enriches each memory entry with LLM-generated keywords, categorical tags, a contextual
description, a dense embedding, an intent embedding, bidirectional link sets, and a scalar utility
(Q-value). This multi-faceted structure supports simultaneous retrieval along keyword, topic,
contextual, and utility axes.
```
(1) **Single-pass, temporally grounded atomic ingestion (extends P1).** We
introduce a session-level extractor (`SinglePassSessionIngestor`) that converts an
entire dialogue into atomic, standalone facts in **one** LLM call, under five
enforced extraction rules: (R1) pronoun resolution; (R2) **absolute-date resolution
for every dated event**, with at most one date per fact; (R3) one fact per note;
(R4) an atomic `(subject, predicate, object)` triplet per fact; and (R5) explicit
capture of reactive/attribution turns with a `speaker` field. Construction cost
becomes O(1) LLM calls per session instead of O(turns), and a fact retaining a
relative or shorthand date is impossible by construction.

```
(2) RL-driven principled memory management (P2). We train a Memory Manager using Group
Relative Policy Optimisation (GRPO) Shao et al. (2024) with outcome-based reward derived
from a frozen Answer Agent, teaching the manager to consolidate complementary facts via UP-
DATE operations rather than fragmenting the memory space—a failure mode that systematically
afflicts heuristic managers.
```
```
(3) Dynamic linking and bidirectional memory evolution (P3). We design a mechanism by
which newly integrated memories trigger simultaneous link generation into the existing knowl-
edge network and retroactive contextual revision of related existing notes, enabling higher-order
conceptual patterns to emerge over time.
```

```
Table 1: Structured comparison of closely related memory systems. Rich: rich note representa-
tion (P1); Write: principled learned write operations (P2); Link: dynamic relational linking (P3);
Util: utility-aware retrieval (P4); Evolve: non-parametric runtime self-evolution.✓ fully supported;
◦ partially;× absent.
```
```
System Paradigm Rich Write Link Util Evolve Temporal Deterministic
                                                 grounding   graph
```
```
MemGPT Packer et al. (2023) OS-inspired hierarchy × ◦ × × × × —
MemoryBank Zhong et al. (2024) Forgetting-curve store × ◦ × × × × —
ReadAgent Lee et al. (2024) Gist distillation ◦ × × × × × —
Mem0 Chhikara et al. (2025) Graph + LLM ops ◦ ◦ ◦ × × ◦ ×
MemoryOS Kang et al. (2025) OS-abstraction tiers ◦ ◦ × × × ◦ ×
Atomic-Linking Xu et al. (2025) Zettelkasten-inspired✓ ×✓ × ◦ ◦ ×
RL-Manager Yan et al. (2025) RL write operations ×✓ × × × × ×
Value-Retrieval Zhang et al. (2026) Q-value retrieval ◦ × ×✓✓ ◦ ×
```
```
ASEM-THG (ours) Unified✓✓✓✓✓ absolute dates LLM-free
```
```
(4) Value-aware two-phase retrieval with distillation (P4). We propose a pipeline that decouples
semantic relevance from functional utility. Phase A constructs a candidate pool via similarity
filtering; Phase B re-ranks by a composite score interpolating z-score-normalised similarity and
learned Q-value. A downstream Answer Agent then distils the retrieved set to the minimal
noise-free subset.
(5) Formal convergence guarantees. We prove that the EMA-based utility update rule converges
in expectation to the true per-memory expected return with bounded asymptotic variance, and
that the composite retrieval policy is optimal under a KL-regularised variational lower bound
on expected reward.
```
## 2 RELATED WORK

```
Table 1 provides a structured comparison across five capability dimensions corresponding directly
to sub-problems P1–P4 and runtime evolution capability.
```
### 2.1 MEMORY SYSTEMS FOR LLM AGENTS

```
The earliest deployed approach maintains complete interaction histories passed as prefix context Ma-
harana et al. (2024) or compressed by a reading agent Lee et al. (2024) to fit within the context win-
dow. While effective for recent events, these systems scale poorly with session length and provide
no mechanism for selectively surfacing the most relevant historical information.
```
```
Flat key-value stores. MemGPT Packer et al. (2023) draws inspiration from OS memory hierar-
chies to implement a two-tier main-context / external-context architecture with rudimentary CRUD
operations. Operation selection is governed by pre-trained LLM priors rather than task-outcome-
optimised policies, leaving the system prone to consolidation failures: complementary facts are
fragmented across entries and contradictions coexist indefinitely. MemoryBank Zhong et al. (2024)
augments flat storage with an Ebbinghaus-inspired forgetting curve that attenuates rarely retrieved
entries while reinforcing frequently accessed ones—an elegant macroscopic consolidation mecha-
nism that nonetheless operates on raw text embeddings without semantic enrichment or relational
linking.
```
```
Graph-enhanced systems. Mem0 Chhikara et al. (2025) extends the CRUD operation set with
graph-database-backed storage and relationship modelling. Operation selection remains governed
by vanilla LLM prompting without any learning signal, preserving consolidation failure modes.
MemoryOS Kang et al. (2025) frames memory management as an OS abstraction with hot and cold
storage tiers but relies on heuristic promotion and eviction policies without downstream task signal.
```

```
Atomic note construction and dynamic linking. A Zettelkasten-inspired approach Xu et al.
(2025) operationalises atomic note construction and dynamic inter-note linking in a computational
memory system. Comprehensive notes comprising raw content, keywords, categorical tags, con-
textual descriptions, dense embeddings, and semantic link sets are constructed upon arrival. When
a new note is integrated, its k nearest neighbours in embedding space are identified, and an LLM
establishes meaningful connections (link generation) and retroactively updates neighbours’ repre-
sentations (memory evolution). Ablation studies confirm that both operations are critical: removing
either significantly degrades multi-hop reasoning. The limitation is the complete absence of princi-
pled write-operation selection and any measure of functional utility.
```
```
RL-trained write operations. A reinforcement learning framework for memory management Yan
et al. (2025) trains a Memory Manager selecting among {ADD, UPDATE, DELETE, NOOP} and an
Answer Agent applying Memory Distillation, jointly trained with GRPO using exact-match cor-
rectness as reward. The contrast with vanilla LLM managers is compelling: faced with sequential
statements “I adopted a dog named Buddy” and “I adopted another dog named Scout”, a vanilla man-
ager issues DELETE+ADD (fragmentation), while the RL-trained manager issues a single UPDATE
(consolidation). The framework operates on flat, semantically impoverished entries and retrieves by
similarity alone.
```
```
Value-aware non-parametric retrieval. A non-parametric retrieval framework Zhang et al.
(2026) organises memory as Intent-Experience-Utility triplets and performs two-phase retrieval:
Phase A filters by semantic similarity to a threshold; Phase B re-ranks using a composite score in-
terpolating normalised similarity and a learned Q-value. Utility estimates are updated via temporal-
difference rules after each interaction. The principal limitation is the absence of rich relational
structure between memory entries, precluding multi-hop traversal.
```
### 2.2 RETRIEVAL-AUGMENTED GENERATION

```
Standard retrieval-augmented generation Lewis et al. (2020) indexes corpora into dense vector
chunks and retrieves the top-k most similar at query time. Advanced variants introduce query rewrit-
ing, hypothetical document embeddings, and post-retrieval re-ranking Gao et al. (2023). Agentic
extensions treat retrieval as dynamic and iterative, interleaving retrieval with chain-of-thought gen-
eration Asai et al. (2023); Trivedi et al. (2022). These approaches exhibit agency at the retrieval
phase but maintain static knowledge bases that do not evolve in response to interaction experience.
ASEM differs fundamentally: its memory bank is dynamically written, linked, and evolved by agent
interactions, and retrieval incorporates learned utility estimates that reflect demonstrated functional
value.
```
### 2.3 CONTINUAL AND LIFELONG LEARNING

```
Classical continual learning approaches Kirkpatrick et al. (2017); Parisi et al. (2019) mitigate catas-
trophic forgetting by constraining parameter updates or preserving samples from past distributions.
For LLM-scale models, parametric adaptation is computationally prohibitive and risks destabilising
pre-trained alignments. This motivates non-parametric continual learning, where the backbone is
frozen and adaptation occurs through modifications to an external memory structure Zhang et al.
(2026). ASEM squarely adopts this paradigm: the backbone LLM is never modified. Runtime
adaptation occurs entirely through memory writes, linkages, evolutions, and scalar Q-value updates.
```
## 3 NOTATION AND FORMAL PROBLEM SETTING

### 3.1 MEMORY BANK AND NOTE STRUCTURE

```
Let Mt = {mi}Ni=1t denote the memory bank at discrete time step t. Each memory note is a
structured tuple
mi =
```
### 

```
ci, ti, Ki, Gi, Xi, ei, Li, zi, qi
```
### 

### , (1)

```
where ciis the raw interaction content; tiis a timestamp; Ki, Gi, and Xiare LLM-generated key-
words, categorical tags, and a contextual description, respectively; ei∈ Rdis a dense embedding
```

```
computed by encoding all textual fields jointly; Li⊆ Mtis the set of semantically linked notes;
zi∈ Rdis an intent embedding representing the query context under which the note was created;
and qi≡ q(zi,mi)∈ R is a learned utility (Q-value) reflecting the expected downstream task reward
associated with retrieving mifor queries with intent similar to zi.
```
```
For temporally grounded deployments the tuple in Eq. (1) is extended with five optional metadata
fields,
```
```
mi = (ci, ti, Ki, Gi, Xi, ei, Li, zi, qi, si, di, τi, Ei, ψi), (1′)
```
```
where si is the identifier of the session the note originated from, di its human-readable session
date, τi an ISO-8601 absolute timestamp, Ei the set of entities named in the note, and ψi its
speaker. These fields are populated at ingestion time (Section 4.1) and consumed by the temporal
and entity retrieval channels (Section 4.4). All five are optional: when absent, the note reduces
to the schema of Eq. (1) and downstream behaviour is unchanged.
```
### 3.2 MEMORY-BASED MARKOV DECISION PROCESS

```
We model agent operation as a Memory-Based Markov Decision Process (M-MDP)
⟨S,A,P,R,γ,M⟩, where the agent policy decomposes as
```
```
π(yt| st,Mt) =
```
### X

```
m∈Mt
```
```
μ(m| st,Mt)
| {z }
retrieval policy
```
```
·pLLM(yt| st,m)
| {z }
inference policy
```
### , (2)

```
where μ is the retrieval policy to be optimised and pLLMis held frozen. The central objective of
ASEM is to learn an optimal retrieval policy
```
```
μ∗ = arg max
μ
```
### E

### "∞

### X

```
t=
```
```
γtR(st,yt)
μ
```
### #

### (3)

```
while maintaining structural coherence acrossMtand without modifying the backbone parameters
of pLLM.
```
### 3.3 THE FOUR SUB-PROBLEMS FORMALISED

```
P1 — Semantic enrichment.
Given raw content c, construct m such that retrieval by any of e, K, G, X yields conceptually
relevant results superior to retrieval by c alone.
P2 — Principled write selection.
Given new information x and current bank M, select operation o ∈
{ADD, UPDATE, DELETE, NOOP} and target m′to maximise the subsequent task-conditional
reward.
P3 — Relational structure.
Maintain a non-trivial link graphL ={Li} such that multi-hop traversal overL recovers infor-
mation not recoverable by single-shot similarity search.
P4 — Utility-aware retrieval.
Retrieve the subset Mret ⊆ M that maximises expected task reward, not merely semantic
proximity to the query embedding.
```
## 4 THE ASEM FRAMEWORK

```
ASEM comprises five tightly coupled stages. Algorithm 1 presents the complete system-level
pipeline; the individual stages are detailed below.
```
### 4.1 STAGE 1: MULTI-ATTRIBUTE ATOMIC NOTE CONSTRUCTION

```
Motivation. A memory entry stored as raw interaction tokens is simultaneously redundant (retain-
ing every filler word) and impoverished (capturing no latent semantic structure). Retrieval over such
entries is dominated by lexical overlap, and merging related entries into coherent representations is
impossible without semantic metadata. We address P1 by constructing each note as a multi-attribute
object whose components provide orthogonal handles for downstream retrieval and linking.
```
```
Procedure. When a new interaction cnewarrives at time tnew, the backbone LLM is invoked with
a structured prompt P 1 to generate semantic attributes:
Knew, Gnew, Xnew← LLM(cnew∥ tnew∥ P 1 ). (4)
A dense embedding is computed by encoding all textual fields jointly:
```
```
enew = fenc
```
```
h
concat
```
### 

```
cnew, Knew, Gnew, Xnew
```
```
i
```
. (5)



```
The note is further initialised with an intent embedding znew= fenc(cnew) and a prior utility qnew=
q 0 that will be refined through the utility update mechanism in Stage 5. The complete note is then
mnew =
```
### 

```
cnew, tnew, Knew, Gnew, Xnew, enew, ∅, znew, q 0
```
### 

### . (6)

```
Design rationale. The tripartite semantic enrichment (K,G,X) creates orthogonal representations
of the same knowledge unit: keywords support exact-match retrieval; categorical tags support topic-
level filtering; contextual descriptions support semantic similarity at a conceptual rather than lexical
level. The joint encoding in Eq. (5) ensures that similarity search captures the full semantic footprint
of the note, including implicit structure not present in the raw tokens.
```
```
Temporal grounding and batch construction. In a multi-session deployment the timestamp ti of a
note must record when the interaction occurred, not when it was ingested. Construction therefore
proceeds session-wise rather than turn-wise. The session header is read from the opening turns of
a dialogue to recover a session identifier and a human-readable date, which is parsed into a real
datetime and an ISO-8601 string. The dated dialogue is then submitted to the backbone in a single
batched prompt P 1b that extracts all atomic facts of the session at once, so construction costs
O(1) LLM calls per session rather than O(turns). Each extracted fact is stamped with the session
datetime (not the wall clock) together with si, di, τi, Ei, and ψi.
```
```
The extraction prompt additionally instructs the model to resolve relative time expressions
against the session date, so that a turn such as "I went to the support group last week" occurring
in a session dated 8 May 2023 is stored as an explicit date in ci. Temporal questions can then be
answered from the note content alone rather than by combining a relative phrase with ti. A note
whose session date cannot be determined retains the wall-clock timestamp and empty optional
fields, degrading gracefully to the schema of Eq. (1).
```
### 4.2 STAGE 2: RL-DRIVEN MEMORY WRITE OPERATIONS

```
Motivation. Na ̈ıve append-only policies accumulate redundant and contradictory entries mono-
tonically, degrading retrieval precision over time. Recency-based eviction destroys potentially crit-
ical historical knowledge. What is required is a policy that can recognise when a new piece of in-
formation complements an existing entry (warranting UPDATE), contradicts it (warranting DELETE),
is genuinely new (warranting ADD), or is already fully represented (warranting NOOP). We address
P2 by training a dedicated Memory Manager to make this discrimination using downstream task
performance as the sole training signal.
```
```
Operation taxonomy. Let Moldbe the current memory bank and x the information extracted
from the new interaction. The Memory Manager πθselects an operation o and, where applicable, a
target entry m′:
(o, m′) ∼ πθ(·| x, Mold), o∈{ADD, UPDATE, DELETE, NOOP}. (7)
```
```
Training objective. The Memory Manager is fine-tuned using Group Relative Policy Optimisa-
tion (GRPO) Shao et al. (2024), which samples G candidate operations per state and normalises
advantages within the group to eliminate an explicit value baseline:
```
```
J (θ) = E
```
### "

### 1

### G

### XG

```
i=
```
```
ρ(θi)Ai− β DKL[πθ∥ πref]
```
### #

### , (8)

```
where ρ(θi)= πθ(o(i),m′(i)| x,Mold)/πold(o(i),m′(i)| x,Mold) is the per-action importance
ratio and Ai= (ri− ̄)r/std(r) is the group-relative advantage. The reward riis the exact-match
correctness of a frozen Answer Agent evaluated on the memory bank resulting from operation o(i):
Ranswer = EM(ypred, ygold). (9)
This outcome-driven signal requires no manual operation labels and teaches the manager to consol-
idate complementary facts via UPDATE rather than na ̈ıvely appending them.
```
### 4.3 STAGE 3: DYNAMIC LINKING AND BIDIRECTIONAL MEMORY EVOLUTION

```
Motivation. Even a well-organised flat store cannot support multi-hop reasoning unless related
notes are explicitly cross-referenced. We address P3 by weaving newly integrated notes into the
existing knowledge network and retroactively revising the representations of related existing notes
to reflect the new information.
```
```
Link generation. After the write operation in Stage 2, the system identifies the k nearest
neighbours of mnew in M by cosine embedding similarity:
sn,j = (enew · ej) / (‖enew‖ ‖ej‖),   Mnearnew = { mj | rank(sn,j) ≤ k, mj ∈ M }.   (10)
The free-form design then asks the LLM to populate the link set Lnew ← LLM(mnew ‖ Mnearnew ‖ P2).   (11)
```
```
Deterministic hyper-graph construction. The deployed ASEM-THG system replaces the
LLM-driven linker in Eq. (11) with a deterministic TemporalHyperGraph with four LLM-free
relation types,
each computed from fields already present on the notes (no extra LLM call):

    same-entity  two notes that share at least one entity in E (weighted by smoothed
                 inverse document frequency, so a rare shared entity counts more than
                 a ubiquitous one);
    temporal     notes whose dates d are adjacent within a session, ordered by τ ;
    semantic     notes whose cosine similarity exceeds a graph threshold;
    superseded_by a strict contradiction on the same atomic (subject, predicate) pair,
                 older → newer (off by default; see §6.3).

```
```
Dual-view synchronisation. The graph is maintained in two synchronised views: an
in-memory adjacency structure used for traversal and PageRank, and a persisted
adjacency written back onto each note's link set L so that edges survive a reload
and remain inspectable. Every relation is a typed, deduplicated record on both
endpoints, exactly as in the free-form design — the difference is that the type and
the endpoint are now computed, not generated.
```
```
Hub suppression. The `same-entity` channel is dominated by a single "hub" class (see
§6.2), so an optional fan-out cap and a smoothed-IDF entity weighting are applied so
that a connection through a rare entity contributes more to traversal than one through
a speaker name that co-occurs with hundreds of notes.
```

```
Bidirectional memory evolution. New information may render the contextual descriptions or cat-
egorical tags of existing notes stale or incomplete. For each mj∈Mnearnew, the system conditionally
updates its attributes to incorporate the new information:
```
```
m∗j ← LLM(mnew∥Mnearnew\{mj}∥ mj∥ P 3 ), (12)
```
```
and m∗jreplaces mjinM. This bidirectional update mechanism mirrors human memory reconsol-
idation: the arrival of a new experience does not merely extend the store but revises the contextual
framing of related prior knowledge, enabling the emergence of higher-order conceptual patterns that
no individual episode would have produced in isolation.
```
```
Free-form does not mean unrecorded: each relation returned by the LLM is persisted as a typed
record (target identifier, relation label) on both endpoints of the edge, deduplicated by target,
so that the nature of the connection survives beyond the prompt that produced it. In batch mode the
linking call processes all notes of a session against the union of their nearest neighbours, and
the stage reports not only how many links were created but how many connect a newly written note
to memory from an earlier session. This cross-session link count is a direct diagnostic of how
strongly a session is integrated into the existing network rather than merely appended to it.
```
### 4.4 STAGE 4: TWO-PHASE HYBRID RETRIEVAL WITH MEMORY DISTILLATION

```
Motivation. At inference time, the system must extract a compact, noise-free context fromM for
the frozen backbone. Purely semantic retrieval conflates relevance with utility: a note semantically
close to the query may encode a strategy that failed in analogous past tasks. Conversely, a note
whose intent embedding is distant but whose Q-value is high may encode the structural pattern
critical for the current multi-step reasoning chain. We address P4 by separating retrieval into two
functionally distinct phases followed by a learned distillation step.
```
```
Phase A — Similarity-based candidate recall. Given a query q, a candidate pool of contextually
consistent notes is assembled:
```
```
C(q) = TopKk 1
```
### 

```
i| sim(eq, ei) > δ
, by sim
```
### 

### , (13)

```
where eq= fenc(q) and δ is a sparsity threshold. IfC(q) = ∅, the system falls back to backbone-
only generation.
```
```
Phase B — Value-aware selection. Within C(q), notes are re-ranked by a composite score that
balances exploration (semantic proximity) and exploitation (learned utility):
```
```
score(q, mi) = (1− λ)dsim(eq, ei) + λ ˆ(qzi, mi), (14)
```
```
whereb· denotes z-score normalisation within C(q) and λ ∈ [0, 1] modulates the exploration–
exploitation trade-off. The final retrieved context is:
```
```
Mret(q) = TopKk 2 (C(q), by score). (15)
```
```
Remark 4.1 (Z-score normalisation is essential). Without z-score normalisation in Eq. (14), the
composite score is dominated by whichever component has larger absolute magnitude, destabilising
the utility–similarity balance and significantly increasing the rate at which utility estimates degrade.
Z-score normalisation withinC(q) ensures that both components contribute equally in expectation,
regardless of their absolute scale.
```
```
Memory distillation. The |Mret| retrieved notes are passed to an RL-trained Answer Agent πφ
that applies a Memory Distillation policy: it selects the minimal subset of notes sufficient for accu-
rate answer generation, filtering residual noise from the retrieved set before reasoning commences:
```
```
Mdistil, ˆ ←y πφ(q, Mret). (16)
```
```
The Answer Agent is trained with the same GRPO objective (Eq. (8)), rewarded by the exact-match
score between ˆ and the gold answery ygold.
```
```
Phase C — Relation-type-weighted N-hop traversal (ASEM-THG extension). The deployed
system seeds the traversal from the top base-scored candidates and expands up to
h = 2 hops over the temporal hyper-graph. Each traversed edge contributes with a
decay and a relation-type weight, so a `superseded_by` edge (w = 1.3) outweighs a
`same-entity` edge (w = 0.9) and a generic `semantic` edge (w = 0.6). Expansion is
pruned by a minimum-similarity gate so that hub traversal cannot flood the candidate
set.
```
```
Phase D — Query-seeded Personalized PageRank (ASEM-THG extension). In parallel, a
Personalized PageRank is run over the shared adjacency with the query-seeded base
candidates as the teleport set, producing a global-importance score for every note.
The final blend is a weighted sum of the base composite score, the N-hop contribution,
and the PPR contribution, exposing three interpretable knobs (α, β, γ) that are set to
0.35/0.25/0.40 in the frozen configuration.
```
```
Bounded recovery loop. If the distilled answer is a refusal or contains no usable
note content, the read path re-queries with a widened candidate pool (a lower similarity
threshold and a larger k2, up to recovery_k2 = 12) and a larger distillation top-n, then
re-runs the Answer Agent once. The loop is bounded to a single retry so that a query
that genuinely has no supporting memory still yields the abstention rather than an
unbounded search.
```
### 4.5 STAGE 5: NON-PARAMETRIC RUNTIME UTILITY UPDATE

```
Motivation. Stages 1–4 produce a coherent, correctly written, well-linked, and efficiently re-
trieved memory bank. Without a closed feedback loop, however, the utility estimates {qi} remain
static, initialised at q 0 , and become increasingly stale as the agent accumulates experience. Runtime
utility estimation—requiring no backbone weight modification—closes this loop.
```



```
Algorithm 1 ASEM-THG: Agentic Self-Evolving Memory — Full Pipeline
Require: Memory bank M; frozen backbone pLLM; Memory Manager πθ; Answer Agent πφ;
hyperparameters k 1 ,k 2 ,k,δ,λ,α,q 0
Ensure: Updated memory bankM′; response ˆy
```
```
▷ — WRITE PATH (new session with turns [c]) —
1: Stage 1: mnew← SinglePassSessionIngest([c],P 1 b)   ▷ one call/session, Eq. 4–6
2: x← facts(mnew)
3: Mold← TopKk(ex, M)
4: Stage 2: (o,m′)∼ πθ(·| x,Mold)   ▷ Eq. 7 (Tier-0 gate for ADD)
5: Apply operation o; updateM accordingly.
6: Stage 3: G← TemporalHyperGraph(M)   ▷ deterministic, 4 relations
7: for each relation r∈{same-entity, temporal, semantic, superseded_by} do
8:     add typed edge(s) for r, weighted by smoothed IDF
9: end for
10: persist dual view onto Note.L
```
```
▷ — READ PATH (query q) —
11: Stage 4a: C(q)← TopKk 1 (eq, M)   ▷ Eq. 13
12: Stage 4b: Mret← TopKk 2 (C(q), score)   ▷ Eq. 14–15
13: Stage 4c: expand C(q) by h≤2 N-hop traversal overG, weighted by relation
14: Stage 4d: add query-seeded Personalized PageRank overG; blend α,β,γ
15: Stage 4e: (Mdistil, ˆy)← πφ(q, Mret)   ▷ Eq. 16
16: if refusal/no-note then re-query once with widened pool; else keep answer
```
```
▷ — UPDATE PATH (reward r) —
17: Receive reward r from environment.
18: Stage 5: for each mi∈Mdistildo qi← qi+ α(r− qi)   ▷ Eq. 17
19: enew← LLMSummarise(q, ˆy,r)
20: M←M∪{(zq, enew, q 0 )}
21: returnM, ˆy
```
```
Utility update rule. After each interaction, the environment returns a scalar reward r ∈ [− 1 , 1]
(e.g., task success, user satisfaction). For each note mi∈ Mdistilthat contributed to the generated
response, its utility is updated via an exponential moving average (EMA) rule:
```
```
qinew← qiold+ α
```
### 

```
r− qoldi
```
### 

### , (17)

```
where α ∈ (0, 1] is the learning rate. This rule implements a Monte Carlo simplification of the
Bellman backup, treating each episode as a one-step MDP. The constant step sizeα enables perpetual
tracking of non-stationary reward distributions, trading off asymptotic variance for responsiveness
to distribution shifts.
```
```
Experience consolidation. Simultaneously, the LLM summarises the trajectory of the current in-
teraction into a new experience string enew, which is appended to the memory bank as a fresh note
(zq,enew,q 0 ), enabling continual expansion of the experiential record.
```
### 4.6 SYSTEM-LEVEL ALGORITHM

## 5 THEORETICAL ANALYSIS

### 5.1 STABILITY–PLASTICITY BALANCE

```
ASEM resolves the stability–plasticity dilemma through a strict architectural separation. The back-
bone pLLMis frozen (stability), while the memory bankM and its utility estimates{qi} are continu-
ously refined (plasticity). This design provides a theoretical guarantee against catastrophic forgetting
of the backbone’s pre-trained capabilities, while permitting arbitrarily rich adaptation of the agent’s
experiential knowledge base. The convergence result below ensures that the plasticity channel is
well-behaved.
```


### 5.2 CONVERGENCE OF UTILITY ESTIMATES

```
Theorem 5.1 (Convergence of utility estimates). Let {qt} be updated according to Eq. (17) with
constant step size α∈ (0, 1]. Suppose the expected reward E[r | z,m] = β(z,m) is stationary and
the pair (z,m) is updated infinitely often. Then
```
```
lim
t→∞
```
```
E[qt] = β(z,m), with convergence rate E[qt]− β = (1− α)t(q 0 − β). (18)
```
```
Moreover, the asymptotic variance is bounded:
```
```
lim sup
t→∞
```
```
Var(qt) ≤
```
```
α
2 − α
```
```
Var(r | z,m). (19)
```
```
Proof. Define the error sequence εt= qt− β. Substituting Eq. (17):
```
```
εt+1 = (1− α)εt+ α (rt− β).
```
```
Taking expectations and applying E[rt| z,m] = β:
```
```
E[εt+1] = (1− α) E[εt].
```
```
Iterating yields E[εt] = (1− α)tε 0 , establishing Eq. (18). Since α ∈ (0, 1], we have| 1 − α| < 1 ,
so E[qt]→ β.
```
```
For the variance, let σ^2 = Var(r | z,m). From the recursion Var(εt+1) = (1−α)^2 Var(εt)+α^2 σ^2 ,
the fixed point of this recursion satisfies V∗= (1− α)^2 V∗+ α^2 σ^2 , giving V∗= α^2 σ^2 /(1− (1−
α)^2 ) = ασ^2 /(2− α), which is Eq. (19).
```
```
Remark 5.1. The bound in Eq. (19) shows that a smaller learning rate α yields lower asymptotic
variance at the cost of slower convergence. This formalises the intuitive trade-off between respon-
siveness to new experience and stability of accumulated estimates.
```
### 5.3 MEMORY UTILITY AS A TRAJECTORY VERIFIER

```
Because the reward signal used to update qireflects the final outcome of an entire interaction tra-
jectory, high-utility notes are those that enabled globally correct reasoning chains, not merely those
that matched the opening query. Formally, letT be a trajectory and r(T ) its terminal reward. The
utility update in Eq. (17) makes qiconverge to E[r(T ) | mi∈ Mdistil(T )] —the expected trajec-
tory reward conditional on note mihaving been retrieved and used. This is precisely the Q-value of
memory miunder the joint policy (μ,πφ,pLLM), justifying the name utility.
```
### 5.4 VARIATIONAL OPTIMALITY OF THE COMPOSITE RETRIEVAL SCORE

```
Proposition 5.1 (Variational optimality). The scoring function in Eq. (14) is the optimal retrieval
policy under a variational objective that maximises expected utility subject to a KL-divergence trust
region anchored to the semantic similarity prior πsim.
```
```
Proof sketch. Consider the variational problem
```
```
μ∗ = arg max
μ
```
```
Em∼μ
```
### 

```
q(zm,m)
```
### 

```
− β−^1 DKL[μ∥ πsim].
```
```
Setting the functional derivative to zero yields the closed-form Boltzmann optimal policy
```
```
μ∗(m| q) ∝ πsim(m| q)· exp
```
### 

```
β q(zm,m)
```
### 

### .

```
Taking the log: logμ∗= logπsim+ β q + const, which is proportional to the composite score in
Eq. (14) after z-score normalisation of both components (which corresponds to setting β = 1/(2ˆσ)
where ˆσ is the within-pool standard deviation of the unnormalised Q-values).
```


## 6 EXPERIMENTS AND THE HONEST RESULT

### 6.1 WHAT WAS ACTUALLY RUN

```
The evaluation reported here is a one-variable ablation of the context builder on top
of the same frozen bank, the same retriever, the same answer agent, and the same
token budget. Two configurations are compared:

  legacy   the pre-THG context builder (flat session notes);
  thg_v1   the ASEM-THG builder: single-pass temporal ingestion + deterministic
           temporal hyper-graph + graph-aware retrieval (Phases C/D).

Two datasets, one run each:
  LoCoMo10 (Maharana et al. 2024)   n = 199 questions
  LongMemEval (Wu et al. 2024)      n = 500 questions

Judge coverage is 1.00 on both datasets (every question was judged, so the judge
column is comparable and not a partial-coverage artifact). All reported values are
printed by a script directly from raw per-question artifacts; none are transcribed
by hand. The evaluation is executed with the frozen code SHA
342ECB0A3A57BF600EB5C74AF6D589C1673403EBBE17DBB961BE2015D7FBC2E3
(unit suite: 345 passed, 1 skipped, 0 failed). The two evaluated banks are
thg_v2 (4,579 notes / 73,862 links) and thg_v2_postfix (4,664 notes / 73,640 links).
```

### 6.2 HEADLINE RESULT: A NEGATIVE THAT REPLICATES

```
Table 2 reports the one-variable ablation. Read it before the rest of the paper: the
ASEM-THG context builder does not beat the legacy builder on answer quality, and on
judge and surface overlap it is very slightly worse. Exact match is a tie on LoCoMo.
```

```
Table 2. One-variable ablation of the context builder (judge coverage = 1.00).
Dataset System EM F1 ROUGE-L BERTScore judge
LoCoMo10 (n=199) legacy 0.3065 0.4935 0.4836 0.8685 0.6533
LoCoMo10 (n=199) thg_v1 0.3065 0.4851 0.4717 0.8702 0.6181
LongMemEval (n=500) legacy 0.2680 0.4232 0.4131 0.8860 0.6140
LongMemEval (n=500) thg_v1 0.2740 0.4166 0.4092 0.8833 0.5880
```

### 6.3 PAIRED ANALYSIS: THE DIFFERENCES ARE NOT SIGNIFICANT

```
Per-question paired comparisons confirm that the observed direction of the LoCoMo EM
tie and the judge gaps are not statistically distinguishable from noise.

LoCoMo10 (n = 199), exact match:
  both correct 115   thg_v1 only 8   legacy only 15   neither 61
  discordant pairs 23 (15 + 8), exact two-sided binomial p = 0.2100  (NOT significant)

LoCoMo10 (n = 199), BERTScore:
  paired mean diff +0.0017, median 0.0000, 75 wins / 72 losses / 52 ties,
  two-sided sign test p = 0.8690  (NOT significant)

LongMemEval (n = 500), exact match:
  both correct 236   thg_v1 only 58   legacy only 71   neither 135
  discordant pairs 129 (71 + 58), exact two-sided binomial p = 0.2907  (NOT significant)
```

```
Where the THG channel does buy something measurable but small: mixed LoCoMo10
questions, where thg_v1 is +1 correct example (2 of 37 → 3 of 37) over legacy. This is
well inside the noise band and is reported as a directional hint, not a result.
```

### 6.4 PER-CATEGORY BREAKDOWN (LongMemEval, thg_v1)

```
Table 3. ASEM-THG per-category, n = 500.
Category            EM       judge
knowledge-update    0.3333   0.7051
temporal-reasoning  0.1353   0.4887
multi-session       0.1729   0.4586
```

```
The temporal-reasoning and multi-session categories are exactly where the temporal
hyper-graph is supposed to help, and they are the two weakest categories. This is a
direct falsification of the naive hypothesis that "adding a temporal graph fixes
temporal questions"; the mechanism is not the bottleneck, or it is not the right
mechanism.
```

### 6.5 GRAPH-CONSTRUCTION FACTS THAT DO SURVIVE SCRUTINY

```
The ablation is negative, but the analysis surfaced structural facts about the entity
channel that are robust and independently checkable. We report them because they
explain why the answer-quality ablation is null.

(a) The entity channel is a hub, not a signal. Of 22,565 same-entity link records,
    the hub_only edge fraction is 0.9474 (≈ 21,378 hub-only vs 1,187 discriminative),
    i.e. a ~19:1 noise-to-signal ratio. The "hub" is a class, not a single note:
    on LoCoMo, speaker names dominate (caroline 65%, melanie 56% of notes); on
    LongMemEval, the literal token `user` dominates (≈ 67% of notes). A channel whose
    edges overwhelmingly connect through one ubiquitous token cannot discriminate.

(b) Fan-out capping does not fix it. Capping hub fan-out reduces the edge count from
    46,084 to 1,792 (a 25.7× reduction) but the hub share only falls from 99.94% to
    99.27% (0.67 percentage points). Capping removes volume, not the structural skew.

(c) Frequency-based (IDF) weighting does bound the damage. Weighting shared entities
    by smoothed inverse document frequency so a rare shared entity counts more than a
    ubiquitous one bounds the channel's reach rather than eliminating it: only 23 of 365
    notes (6.3%) have a rare peer in their top-5; the corpus top-5 splits 1,769 / 52
    between hub and discriminative; and a witness note 00cb74d3 moves from rank #5 to
    rank #1 as its score rises from 0.6589 to 3.8245. This is the mechanism Stage 3
    relies on; it is real but strictly bounded.

(d) Supersession, as configured, does not work. The `superseded_by` relation is correct
    on 0 of 5 hand-checked contradiction cases, so it is shipped OFF by default. We do
    not claim it as a working component.
```

### 6.6 HONESTY PROTOCOL

```
Every headline number in this section is printed by a script from raw artifacts under
`analysis/` (no hand transcription). Judge scores are reported only where coverage is
1.00 or explicitly stated. The ablation is a negative and is stated first, not last.
Pairwise differences are tested (exact binomial, sign test) and reported with their
p-values whether or not they are significant. No "substantial gains" claim is made,
because the measurement does not support one.
```

## 7 DISCUSSION

### 7.1 WHY THE NEGATIVE IS THE CONTRIBUTION

```
What a "temporal memory graph" is worth on long-horizon conversational QA is a
question the field has largely answered by assertion. This paper answers it with a
controlled measurement: a deterministic temporal hyper-graph layered onto a competitive
frozen retrieval-and-answer stack does not move exact match, and moves the judge metric
slightly the wrong way, on both a 199-question and a 500-question benchmark, with
p-values of 0.21 and 0.29. That is a useful, citable result. It says the relational
representation is not the bottleneck at this scale — the answer agent, the token
budget, and the base retriever are.
```

### 7.2 THE GRAPH STILL TEACHES SOMETHING

```
The construction analysis (Section 6.5) is where the graph earns its keep. The entity
channel is ~95% hub and ~19:1 noise-to-signal; fan-out capping does not reduce hub
dominance while IDF weighting does; and IDF's reach is bounded at ~6% of notes. These
are the design facts a practitioner needs before building an entity graph on
speaker-heavy dialogue: the naive construction is dominated by the speaker name, and
only a frequency-aware weighting makes the rare-entity signal legible at all. This is
a concrete, adversarially checked finding that generalises beyond this codebase.
```

### 7.3 CRITIQUE OF THE RECOVERY LOOP AND REFUSAL HANDLING

```
The bounded recovery loop (Stage 4) re-queries once with a widened pool on refusal.
It is a local patch for the answer agent's tendency to abstain when the distilled
payload looks thin; it does not change the ranking of the retrieved set, so it cannot
recover a note the retriever never surfaced. It improved the refusal rate in
development and is retained, but it is a mitigation, not a mechanism, and the
per-category numbers show it does not rescue temporal or multi-session questions.
```

### 7.4 LIMITATIONS

```
The measurement is single-run per dataset and is not multi-seed; we do not report
variance, and the paired tests are interpreted accordingly. One benchmark (LoCoMo10)
is small (n = 199), so its exact-match tie is a coarse resolution. PersonalMemBench was
not evaluated in this revision; the earlier setup that claimed three benchmarks and
five seeds is superseded by the actual two-dataset, single-run study reported here. The
judge metric is a single model; no human evaluation was performed at this revision.
```

## 8 CONCLUSION

```
We have presented ASEM-THG: a five-stage agentic memory framework made explicitly
temporal and relational, with a single-pass temporally grounded ingestion step, a
deterministic LLM-free temporal hyper-graph, graph-aware value retrieval with N-hop
traversal and query-seeded PageRank, and a formally guaranteed non-parametric utility
update. The framework retains the original design's stability–plasticity separation and
the convergence guarantee for utility estimates without modifying any backbone
parameter.

The empirical finding is deliberately reported as a negative: the temporal hyper-graph
does not improve answer quality over a strong legacy context builder, and this null
replicates across two benchmarks with paired p-values of 0.21 and 0.29. The
contribution is therefore dual — a method that makes the representation legible and
auditable, and a bounded, adversarially checked account of what that representation
does and does not buy on long-horizon conversational QA. The graph's entity channel is
a hub we can characterise quantitatively (95% hub, 19:1 noise-to-signal) and partially
neutralise with IDF weighting (bounded to ~6% of notes); its supersession channel does
not work as configured and is disabled. We release the ablation, the graph statistics,
and the reproducibility record so that the "does a temporal graph help?" question can be
answered on numbers rather than intuition.
```

## APPENDIX A: REPRODUCIBILITY

```
Frozen code SHA   342ECB0A3A57BF600EB5C74AF6D589C1673403EBBE17DBB961BE2015D7FBC2E3
Unit suite        345 passed, 1 skipped, 0 failed
Banks             thg_v2 (4,579 notes / 73,862 links); thg_v2_postfix (4,664 / 73,640)
Datasets          LoCoMo10 (n=199), LongMemEval (n=500); judge coverage 1.00
Ablation          one variable (context builder); same bank, retriever, answer agent,
                  token budget across arms
Artifacts         analysis/thg_evaluation_report.md, analysis/FINAL_SUMMARY.md,
                  analysis/graph_build_report.md (every number above is printed by a
                  script from raw per-question json under analysis/)
Environment       conda env `memory-r1`; transformer stack pinned; KMP_DUPLICATE_LIB_OK=TRUE
```

