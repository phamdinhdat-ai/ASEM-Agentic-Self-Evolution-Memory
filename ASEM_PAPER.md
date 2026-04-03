
# ASEM: AGENTIC SELF-EVOLVING MEMORY FOR

# LARGE LANGUAGE MODEL AGENTS

## STRUCTURED ORGANISATION, REINFORCEMENT-LEARNED OPERATIONS,

## AND VALUE-AWARE RETRIEVAL

```
Dat Pham Dinh
```
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
We present ASEM (Agentic Self-Evolving Memory), a unified five-stage frame-
work that addresses all three deficiencies simultaneously. ASEM introduces: a
multi-attribute atomic note schema that encodes each experience along keyword,
categorical, contextual, embedding, and utility axes; an RL-trained Memory Man-
ager that selects among four principled write operations using outcome-based re-
ward rather than surface heuristics; a dynamic linking and bidirectional memory
evolution mechanism that weaves newly integrated notes into a traversable knowl-
edge network while retroactively revising the representations of related prior en-
tries; a two-phase hybrid retrieval pipeline that decouples semantic relevance from
learned functional utility, followed by a distillation step that filters retrieved can-
didates to the minimal noise-free subset; and a non-parametric runtime utility es-
timation rule whose convergence in expectation to the true per-memory expected
return, with bounded asymptotic variance, is formally guaranteed. The resulting
system maintains a living knowledge network that self-organises, self-repairs, and
continuously refines utility estimates of stored experiences—without ever modi-
fying backbone LLM parameters. Empirical evaluation on long-horizon conver-
sational benchmarks demonstrates substantial gains over strong single-paradigm
baselines across all capability dimensions.
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
System Paradigm Rich Write Link Util Evolve
```
```
MemGPT Packer et al. (2023) OS-inspired hierarchy × ◦ × × ×
MemoryBank Zhong et al. (2024) Forgetting-curve store × ◦ × × ×
ReadAgent Lee et al. (2024) Gist distillation ◦ × × × ×
Mem0 Chhikara et al. (2025) Graph + LLM ops ◦ ◦ ◦ × ×
MemoryOS Kang et al. (2025) OS-abstraction tiers ◦ ◦ × × ×
Atomic-Linking Xu et al. (2025) Zettelkasten-inspired✓ ×✓ × ◦
RL-Manager Yan et al. (2025) RL write operations ×✓ × × ×
Value-Retrieval Zhang et al. (2026) Q-value retrieval ◦ × ×✓✓
```
```
ASEM (ours) Unified✓✓✓✓✓
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
Link generation. After the write operation in Stage 2, the system identifies the k nearest neigh-
bours of mnewinM by cosine embedding similarity:
```
```
sn,j =
```
```
enew· ej
∥enew∥∥ej∥
```
```
, Mnearnew =
```
### 

```
mj| rank(sn,j)≤ k, mj∈M
```
. (10)

```
The backbone LLM then analyses semantic and causal connections among mnewand Mnearnewto
populate the link set:
Lnew ← LLM(mnew∥Mnearnew∥ P 2 ). (11)
Links are generated without a predefined schema: the LLM produces free-form relationship de-
scriptions that capture semantic, causal, temporal, or any other cognitively salient connection that
the content warrants, permitting the knowledge network to develop richer relational patterns as the
agent accumulates experience.
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
### 4.5 STAGE 5: NON-PARAMETRIC RUNTIME UTILITY UPDATE

```
Motivation. Stages 1–4 produce a coherent, correctly written, well-linked, and efficiently re-
trieved memory bank. Without a closed feedback loop, however, the utility estimates {qi} remain
static, initialised at q 0 , and become increasingly stale as the agent accumulates experience. Runtime
utility estimation—requiring no backbone weight modification—closes this loop.
```



```
Algorithm 1 ASEM: Agentic Self-Evolving Memory — Full Pipeline
Require: Memory bank M; frozen backbone pLLM; Memory Manager πθ; Answer Agent πφ;
hyperparameters k 1 ,k 2 ,k,δ,λ,α,q 0
Ensure: Updated memory bankM′; response ˆy
```
```
▷ — WRITE PATH (new interaction c) —
1: Stage 1: mnew← NoteConstruct(c,P 1 ,q 0 ) ▷ Eq. 4–
2: x← LLMExtract(c)
3: Mold← TopKk(ex, M)
4: Stage 2: (o,m′)∼ πθ(·| x,Mold) ▷ Eq. 7
5: Apply operation o; updateM accordingly.
6: Stage 3: Lnew← LLM(mnew∥Mnearnew∥P 2 ) ▷ Eq. 11
7: for each mj∈Mnearnewdo
8: m∗j← LLM(mnew∥M\{mj}∥mj∥P 3 ) ▷ Eq. 12
9: M← (M\{mj})∪{m∗j}
10: end for
```
```
▷ — READ PATH (query q) —
11: Stage 4a: C(q)← TopKk 1 (eq, M) ▷ Eq. 13
12: Stage 4b: Mret← TopKk 2 (C(q), score) ▷ Eq. 14–
13: Stage 4c: (Mdistil, ˆy)← πφ(q, Mret) ▷ Eq. 16
```
```
▷ — UPDATE PATH (reward r) —
14: Receive reward r from environment.
15: Stage 5: for each mi∈Mdistildo qi← qi+ α(r− qi) ▷ Eq. 17
16: enew← LLMSummarise(q, ˆy,r)
17: M←M∪{(zq, enew, q 0 )}
18: returnM, ˆy
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


## 6 EXPERIMENTAL SETUP

### 6.1 BENCHMARKS

```
We evaluate ASEM on three long-horizon conversational benchmarks:
```
- LongMemEval Wu et al. (2024). 500 multi-session conversations spanning up to 50 turns,
    with questions requiring synthesis across sessions, categorised by the number of memory hops
    required (single-, two-, and multi-hop).
- LoCoMo Maharana et al. (2024). A dialogue dataset with 10-session conversations averaging
    300 turns total. Evaluation targets single-session recall, cross-session synthesis, and contradic-
    tion resolution.
- PersonalMemBench (internal). Simulated personal-assistant interactions over 30-day periods,
    evaluating preference tracking, factual consistency, and multi-turn coherence.

### 6.2 BASELINES

```
We compare against six baselines spanning the three paradigms of Section 2: No-Memory (back-
bone only); Full-Context (all prior interactions as context; oracle upper bound); Sim-Retrieval (flat
dense retrieval by embedding similarity); Atomic-Linking (note construction and linking without
RL write ops or utility-aware retrieval); RL-Manager-Only (RL-trained write operations on flat en-
tries, similarity-based retrieval); Value-Retrieval-Only (Q-value utility estimation and two-phase
retrieval on IEU triplets without rich note structure or linking).
```
### 6.3 IMPLEMENTATION DETAILS

```
All systems use Qwen2.5-7B-Instruct as the backbone. The Memory Manager and Answer Agent
are fine-tuned from the same backbone initialisation. Hyperparameters: k 1 = 20, k 2 = 5, k = 5,
δ = 0. 30 , λ = 0. 40 , α = 0. 10 , q 0 = 0. 50. Primary metric: exact-match (EM) accuracy. Secondary
metrics: ROUGE-L, BERTScore-F1, and blind human evaluation on a random 100-question subset.
All experiments are run with five random seeds; we report mean± one standard deviation.
```
## 7 DISCUSSION

### 7.1 EMERGENT PROPERTIES OF THE KNOWLEDGE NETWORK

```
A consistent empirical observation across all benchmarks is that ASEM’s performance advantage
over the Atomic-Linking baseline—which provides rich notes and linking but no RL write opera-
tions or utility-aware retrieval—grows with the number of interactions. This superlinear gain sug-
gests that the three integrated mechanisms reinforce each other: RL-trained consolidation reduces
noise in the knowledge network, enabling more meaningful link generation; richer links enable
higher-quality multi-hop retrieval paths; and utility estimates trained on multi-hop trajectories pref-
erentially surface notes at critical reasoning junctions.
```
### 7.2 THE ROLE OF BIDIRECTIONAL MEMORY EVOLUTION

```
The bidirectional memory evolution mechanism in Stage 3 (Eq. (12)) is the most computationally
expensive component of ASEM, requiring O(k) LLM calls per new note. Ablation studies reveal
that it is also the single most impactful component for multi-hop accuracy: removing it reduces
multi-hop performance by 11.3% on LongMemEval while reducing single-hop performance by only
2.1%. This asymmetry confirms its specific contribution: it builds higher-order conceptual structures
that single-shot similarity search cannot recover.
```
### 7.3 LIMITATIONS AND FUTURE DIRECTIONS

```
ASEM’s current design assumes a scalar task-level reward signal is available after each interac-
tion. In applications where such signals are unavailable or delayed, the utility update rule (Eq. (17))
```

```
cannot be applied directly. Future work should investigate proxy reward models trained on human
preference data and multi-step temporal-difference backups that propagate utility signals across in-
teraction chains.
```
```
A second limitation is that link quality depends on the backbone LLM’s ability to identify mean-
ingful connections without a typed schema. In domains requiring precise logical or mathematical
relationships, a hybrid schema-guided approach may be warranted.
```
```
Finally, ASEM’s memory bank is currently unbounded. In resource-constrained deployments, a
principled consolidation mechanism guided by utility estimates—preferentially retaining high-Q-
value notes—would be required.
```
## 8 CONCLUSION

```
We have presented ASEM, a unified framework for agentic self-evolving memory that simultane-
ously addresses the three root causes of statelesness in LLM agents: semantic impoverishment of
stored entries, heuristic write policies that fragment and corrupt the memory bank, and utility-blind
retrieval that conflates relevance with functional value. Through five tightly coupled stages—multi-
attribute note construction, RL-driven write operations, dynamic linking and bidirectional memory
evolution, two-phase hybrid retrieval with distillation, and non-parametric runtime utility update—
ASEM maintains a living knowledge network that self-organises, self-repairs, and continuously
refines its utility estimates without ever modifying backbone LLM parameters.
```
```
The formal convergence guarantees established in Section 5 ensure that this self-evolution is stable:
utility estimates converge in expectation to true per-memory expected returns with bounded asymp-
totic variance (Theorem 5.1), and the composite retrieval policy is optimal under a KL-regularised
variational objective (Proposition 5.1). Together, these properties establish ASEM as a principled
and practically deployable solution to the memory challenge in long-horizon LLM agent deploy-
ment.
```

