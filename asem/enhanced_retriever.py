"""Enhanced hybrid retrieval with global graph semantics and typed graph ranking.

Extends the base HybridRetriever with:

* **Global semantic signals**: Louvain community detection + PageRank centrality
* **Typed heterogeneous Personalized PageRank**: one walk whose transition matrix
  carries each edge's relation type and whose reset distribution carries the
  entity nodes the query names (G2)
* **Intent-grounded Q-values**: Query-to-z similarity gates utility relevance
* **Hybrid scoring**: ``alpha*local + beta*global + gamma*utility``

The original phases A+B are preserved (ANN recall + z-score composite re-rank).
Phase C is a single typed PageRank (:meth:`EnhancedHybridRetriever._graph_rank`),
which replaced the earlier pair of ad-hoc channels — a bounded relation-weighted
BFS and a type-blind PPR append — that disagreed about what an edge means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import Note
from .retriever import HybridRetriever

_log = get_logger("S4.enhanced_retriever")


@dataclass
class EnhancedHybridRetriever(HybridRetriever):
    """Hybrid retrieval augmented with global graph structure.

    Extends ``HybridRetriever`` (preserving all existing fields / Phase A+B logic)
    and adds graph-based re-ranking, community-aware boosting, a typed
    heterogeneous Personalized PageRank over the hyper-graph, and intent-grounded
    utility scoring.

    New knobs (in addition to inherited ones):
        alpha: Weight for local semantic similarity (default 0.35).
        beta:  Weight for global graph score (default 0.25).
        gamma: Weight for learned utility (default 0.40).
        base_weight: Weight on the base Phase A+B ordering in Phase D (0.60).
        multi_hop_topn: How many base winners seed the graph walk (default 2).
        relation_weights: Per-relation transition weights for the walk and the
            base retriever's link traversal (single source of truth).
        ppr_edge_weights: Override for the walk's transition table (None = derive
            from ``relation_weights``).
        ppr_damping: PageRank teleport factor (default 0.85).
        entity_seed_weight: Share of reset mass given to query-named entities.
        entity_seed_topn: How many query entities may seed the walk (default 5).
        graph_rank_topn: How many walk-ranked notes may enter the pool (default 20).
        use_graph_rank: Master switch for the walk (default True).
        community_boost: Multiplier for same-community notes (default 1.2).
        enable_intent_q: Gate Q-values by query-to-z similarity (default True).
        enable_global_semantics: Toggle community + PageRank (default True).
    """

    # ── Hybrid weights ──────────────────────────────────────────────────
    alpha: float = 0.35   # local semantic
    beta: float = 0.25    # global graph
    gamma: float = 0.40   # learned utility

    # Weight on the base Phase A+B ordering (z-scored rank) in the Phase D
    # re-rank. The remainder (1 − base_weight) is split across alpha/beta/gamma.
    # Keeping this high ensures the graph/utility terms refine — rather than
    # overwrite — the retrieval order, which the old flat min-max blend did.
    base_weight: float = 0.60

    # ── Graph walk seeding ──────────────────────────────────────────────
    # How many base winners seed the walk (`multi_hop_topn * 2` seeds are drawn),
    # and the relevance floor for a walk-ranked note. `multi_hop_min_score` is
    # retained for the legacy base-retriever traversal score floor.
    multi_hop_topn: int = 2
    multi_hop_min_sim: float = 0.30
    multi_hop_min_score: float = 0.0
    # Legacy BFS knobs, retained so existing configs still load. `max_hops` and
    # `hop_decay` no longer bound the walk (PageRank has no hop limit); they are
    # read by the base retriever's own link traversal.
    max_hops: int = 2
    hop_decay: float = 0.7

    # ── Relation-type-aware traversal ───────────────────────────────────
    # Weights applied per relation type.  Strong relations
    # (contradicts/extends/causal) carry more information than weak ones
    # (same-topic/temporal/semantic), so traversed neighbors reached via strong
    # edges are preferred. ONE table drives BOTH the base retriever's link
    # traversal and the G2 graph walk's transition matrix — see
    # `_resolve_edge_weights`.
    relation_weights: Dict[str, float] = field(default_factory=lambda: {
        "contradicts": 1.2,
        "extends": 1.1,
        "causal": 1.1,
        "superseded_by": 1.3,   # THG versioning: a newer fact replaces an older one
        "same-entity": 0.9,     # THG entity co-occurrence: corroboration, not a typed relation
        "same-topic": 0.8,
        "temporal": 0.7,
        "semantic": 0.6,
        "linked": 1.0,  # legacy links (unknown type)
    })
    # If set, traversal only follows edges whose relation is in this set.
    filter_relation_types: Optional[Set[str]] = None
    # Weight applied to a traversed edge whose relation is not in the table
    # above. Kept neutral (1.0) to match ``retriever._RELATION_BOOST``; the old
    # 0.5 penalty silently down-ranked legitimate unknown-typed links.
    unknown_relation_weight: float = 1.0

    # ── Global semantics ────────────────────────────────────────────────
    community_boost: float = 1.2
    enable_intent_q: bool = True
    enable_global_semantics: bool = True

    # ── Cached graph state (recomputed when bank changes) ───────────────
    _graph_hash: int = field(default=-1, init=False, repr=False)
    _community_map: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _pagerank: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    # ── Query-seeded Personalized PageRank ──────────────────────────────
    # The graph channel above uses GLOBAL PageRank, which is query-independent:
    # it rewards notes that are central to the whole bank, not notes that are
    # reachable from the evidence the query already found. ``ppr_weight`` adds a
    # QUERY-CONDITIONAL graph signal — PPR seeded on the Phase A+B winners —
    # which is the mechanism that surfaces the *other half* of a multi-hop fact
    # (the premise note that shares no query terms with the question).
    # Set to 0.0 to reproduce the previous behaviour exactly.
    ppr_weight: float = 0.15
    ppr_topk: int = 20
    # GB4: bidirectional personalization. `personalized_pagerank` walks OUTWARD
    # from the seeds, so a note that merely POINTS AT the seed set — a premise
    # the seed depends on — could never be surfaced at any rank, however strong
    # the dependency. `ppr_reverse` adds the reversed walk as a second channel,
    # which is what makes "what does this evidence rest on?" reachable.
    #
    # Defaults preserve today's behaviour exactly: `ppr_reverse=False` means the
    # reverse channel contributes nothing, and `ppr_weight=0.0` still reproduces
    # the pre-W3 ranking byte-for-byte.
    ppr_reverse: bool = False
    ppr_reverse_weight: float = 0.5

    # ── G2: typed heterogeneous PPR ─────────────────────────────────────
    # Per-RELATION transition weights for the walk. `None` derives the table
    # from `relation_weights` (see `_resolve_edge_weights`), which is what wires
    # the typed table — until now applied only to the bounded BFS — into the
    # PageRank transition matrix, where it governs EVERY path rather than one
    # hop. Set explicitly to ablate a single relation's weight.
    ppr_edge_weights: Optional[Dict[str, float]] = None
    # PageRank teleport factor. 0.85 is the networkx default and the value the
    # forward walk has always used; exposed so an ablation can move it.
    ppr_damping: float = 0.85
    # Entity-anchored seeding (HippoRAG 2's reset scheme, but over TYPED edges).
    # Seeding only on the base winners confines the walk to notes dense
    # retrieval already found, so a bridging premise that shares no query terms
    # can never be reached. Seeding the entity nodes the QUESTION names gives the
    # walk a second, query-side entry point. `entity_seed_weight` is the share of
    # reset mass they receive (0.0 reproduces the note-only seeding exactly).
    entity_seed_weight: float = 0.5
    entity_seed_topn: int = 5
    # Cap on how many walk-ranked notes enter the candidate pool, and an optional
    # relevance floor for them. The floor defaults OFF because gating the append
    # only removes candidates, and an ungated append cannot lower answer-window
    # recall — it can only add. (The G1 candidate-level "gain", 0.9695 vs 0.9594,
    # was a pool-size artifact: scoring over the WHOLE candidate list, ~27 notes
    # vs ~15. Scored over the top-8 the answer prompt actually reads, the two
    # arms are identical, 0.9289. The floor is retained for the BFS case, whose
    # relevance-blind append is what it was written for.)
    graph_rank_topn: int = 20
    graph_rank_min_sim: float = 0.0
    # Master switch: `false` drops the walk entirely, leaving Phase D's
    # similarity/utility/global terms — the "graph is dark" rung of the ladder.
    use_graph_rank: bool = True

    # G3: multi-hop query decomposition. When set and it fires, the retriever
    # retrieves each sub-question INDEPENDENTLY and fuses the pools, because the
    # G3 probes showed the missing hop's evidence is usually absent from the pool
    # for the original query — no re-ranking of that pool can recover it, but a
    # different query can (the oracle recovered 16/16). ``None`` = no planning.
    query_planner: Optional[Any] = None
    # RRF weight of each SUB-question's list relative to the original query's
    # (which is always 1.0). 1.0 interleaves them, so a sub-question's top note
    # reaches the answer window even when the original query already filled it —
    # the "crowded out" failure the mechanism probe measured.
    planner_fuse_weight: float = 1.0

    # ── G3b: session-diverse answer window ──────────────────────────────
    # A multi-hop question names evidence in two sessions, but the answer window
    # is the TOP `answer.max_context_notes` of the returned pool. Ranking by
    # similarity alone fills that window with the one session the question's
    # words match best, so the second premise never reaches the model — even when
    # retrieval DID find it. The probe measured this directly: on LoCoMo cat-1,
    # widening the pool alone made all_sessions WORSE (0.4074 -> 0.3704), while
    # widening + taking the best-ranked note per session reached 0.7037 against
    # an oracle ceiling of 0.8148, at ZERO LLM cost. The LLM query planner, by
    # contrast, moved nothing (0.4074). So the fix is selection, not a new query.
    #
    # `false` (default) reproduces the pre-G3b ranking byte-for-byte.
    session_diversity: bool = False

    # The corpus graph lives on ``pipeline.hyper_graph`` for ASEM-THG. It is
    # optional so plain ASEM / FastASEM runs keep working unchanged.
    hyper_graph: Optional[Any] = None
    # note_id -> PPR score from the most recent query. Populated by
    # ``_graph_rank`` and consumed by the Phase D blend.
    ppr_scores: Dict[str, float] = field(default_factory=dict, init=False, repr=False)
    # GB4: note_id -> score from the REVERSED walk, kept as a SEPARATE dict (not
    # merged into `ppr_scores`) so an ablation can tell the two channels apart
    # and `ppr_scores` keeps its existing meaning for any existing consumer.
    ppr_scores_reverse: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Override: retrieve()
    # ------------------------------------------------------------------

    def retrieve(self, query: str, M: MemoryBank) -> List[Note]:
        """Enhanced retrieval: base RRF, one typed graph walk, global re-rank.

        When a :class:`~asem.query_planner.QueryPlanner` is attached and fires
        for a multi-hop query, the sub-questions are retrieved INDEPENDENTLY and
        their pools fused here, before the graph walk and Phase D. The fusion is
        what lets a sub-question's evidence reach the answer window even when
        the original query's own results already filled it.

        G3b: the returned pool is then optionally reordered so its FRONT spans
        sessions (`session_diversity`). Applied here, at the single exit, so both
        the planner and the plain path get it, and so it acts on the pool the
        answer window actually reads.
        """
        notes = self._retrieve_dispatch(query, M)
        if self.session_diversity and notes:
            notes = self._diversify_sessions(notes)
        return notes

    def _retrieve_dispatch(self, query: str, M: MemoryBank) -> List[Note]:
        """Route to the planner path or the plain core path."""
        # ── G3: multi-hop decomposition (optional) ──────────────────────
        # Each sub-question is retrieved with the planner switched OFF, so a
        # sub-question cannot recurse into further planning.
        planner = self.query_planner
        if planner is None:
            return self._retrieve_core(query, M)

        subs = planner.plan(query)
        if len(subs) <= 1:
            return self._retrieve_core(query, M)

        pools = [self._retrieve_core(sq, M) for sq in subs]
        fused = self._fuse_subquestions(pools, subs)
        self.stats["planner_fired"] = True
        self.stats["planner_subquestions"] = len(subs)
        self.stats["planner_pool_sizes"] = [len(p) for p in pools]
        self.stats.update(planner.stats)
        self.stats["total_retrieved"] = len(fused)
        return fused

    def _diversify_sessions(self, notes: List[Note]) -> List[Note]:
        """Reorder so the FRONT of the pool spans sessions, keeping rank order.

        The answer window is the top ``answer.max_context_notes`` of this list.
        A multi-hop question's evidence lives in two sessions; ranking by
        similarity alone fills the window with the one session the question's
        words match best and the second premise never reaches the model. This
        walks the ranked list once, emitting the best-ranked note of each session
        the first time that session appears, then appends the remainder in rank
        order.

        Order is preserved WITHIN the emitted set (a session's first appearance
        is its best note, and sessions are emitted in the order their best note
        ranks), so this is a stable reordering, not a re-scoring: the top note is
        unchanged and no note's relative order among same-session peers is
        disturbed. Notes with no ``session_id`` are treated as their own
        singleton groups and keep their rank position.

        Only the FRONT is spread; the tail is unchanged, so a caller that reads
        the whole pool sees the same notes in a session-aware order.
        """
        seen: Set[str] = set()
        head: List[Note] = []
        tail: List[Note] = []
        for i, note in enumerate(notes):
            sid = getattr(note, "session_id", None)
            # A note without a session id cannot crowd another session out, so
            # it never claims a slot: it stays in the tail at its rank.
            if sid is None:
                tail.append(note)
                continue
            key = str(sid)
            if key in seen:
                tail.append(note)
            else:
                seen.add(key)
                head.append(note)
        out = head + tail
        self.stats["session_diversity"] = True
        self.stats["session_diversity_sessions"] = len(seen)
        self.stats["session_diversity_head"] = len(head)
        return out

    def _retrieve_core(self, query: str, M: MemoryBank) -> List[Note]:
        """The pre-G3 retrieval path, unchanged. Planner-free by construction."""
        self.stats = {}

        # Run base Phase A+B (ANN + composite z-score re-rank)
        base_results = super().retrieve(query, M)
        if not base_results:
            return []

        e_q = self.backend.embed(query)

        # Refresh graph metrics if bank has changed
        if self.enable_global_semantics:
            self._refresh_graph(M)

        # Shared dedupe set for the graph phase below. Initialised here (not
        # inside the branch) so the Phase D blend cannot hit an unbound name.
        seen_ids = {n.id for n in base_results}

        # ── Phase C (G2): ONE typed graph walk ──────────────────────
        # This replaces the previous pair of ad-hoc channels — a bounded BFS
        # (`_multi_hop_expand`) and an outward PPR append (`_ppr_expand`) — with
        # a single heterogeneous PageRank whose transition matrix carries the
        # RELATION TYPE of every edge and whose reset distribution carries the
        # entity nodes the question names. The two old channels disagreed about
        # what a graph edge means (BFS weighted by relation, PPR ignored type
        # entirely) and each appended candidates on its own relevance rules, so
        # their combined effect was not attributable to either. See `_graph_rank`.
        #
        # Both switches must be on: `use_graph_rank` enables the walk itself,
        # `ppr_weight` is how much Phase D trusts its ranking. Either at zero
        # leaves Phase D's similarity/utility/global terms alone, which is the
        # "graph is dark" ablation rung and, at `ppr_weight=0.0`, reproduces the
        # pre-W3 ranking byte-for-byte.
        if self.use_graph_rank and self.ppr_weight > 0.0:
            for n in self._graph_rank(query, e_q, base_results, M):
                if n.id not in seen_ids:
                    base_results.append(n)
                    seen_ids.add(n.id)

        # ── Phase D (NEW): Global graph re-ranking ──────────────────
        if self.enable_global_semantics and self._community_map:
            n = len(base_results)
            # Preserve the base ordering: base_results[0] is best from Phase A+B.
            # ``base_rank`` = descending rank score in [0, 1].
            base_rank = np.array(
                [1.0 - (i / max(n - 1, 1)) for i in range(n)], dtype=float
            )

            local_sims = np.array(
                [self._cosine(e_q, note.e) for note in base_results], dtype=float
            )

            # Intent-grounded utility, gated by query-to-intent similarity.
            q_vals = np.array([note.q for note in base_results], dtype=float)
            if self.enable_intent_q:
                intent_sims = np.array(
                    [self._cosine(e_q, note.z) for note in base_results], dtype=float
                )
                q_eff = q_vals * intent_sims
            else:
                q_eff = q_vals
            # If every note has the same q (e.g. all 0.5 before Stage-5 warming),
            # the utility term carries no signal — drop it rather than let a
            # z-scored constant inject noise / a flat global term dominate.
            q_has_signal = float(q_eff.std()) > 1e-6

            global_scores = np.array(
                [
                    self._compute_global_score(
                        note, base_results[0] if base_results else None
                    )
                    for note in base_results
                ],
                dtype=float,
            )

            local_z = self._zscore(local_sims)
            global_z = self._zscore(global_scores)
            rank_z = self._zscore(base_rank)
            q_z = self._zscore(q_eff) if q_has_signal else np.zeros(n)

            # Query-seeded PPR score per note, 0 for notes the walk did not rank.
            # A z-score over mostly-zeros is degenerate, so a missing score is
            # encoded as "below the ranked set" via min-max normalisation instead.
            if self.ppr_scores:
                raw_ppr = np.array([self.ppr_scores.get(note.id, 0.0) for note in base_results], dtype=float)
                hi = float(raw_ppr.max()) if raw_ppr.size else 0.0
                ppr_z = (raw_ppr / hi) if hi > 0 else np.zeros(n)
            else:
                ppr_z = np.zeros(n)

            # GB4: fold the REVERSED walk into the SAME term, combined by
            # `max`. Taking the max rather than a sum is deliberate: the two
            # walks answer different questions ("what follows from the
            # evidence" vs "what the evidence rests on") and a note strong in
            # exactly one of them should be promoted, not diluted by being
            # absent from the other. `ppr_reverse_weight` scales the reverse
            # channel so a run can trust it less than the forward walk.
            if self.ppr_reverse and self.ppr_scores_reverse:
                raw_rev = np.array(
                    [self.ppr_scores_reverse.get(note.id, 0.0) for note in base_results],
                    dtype=float,
                )
                hi_rev = float(raw_rev.max()) if raw_rev.size else 0.0
                ppr_rev_z = (
                    (raw_rev / hi_rev) * float(self.ppr_reverse_weight)
                    if hi_rev > 0
                    else np.zeros(n)
                )
                ppr_z = np.maximum(ppr_z, ppr_rev_z)
                self.stats["phase_d_ppr_reverse_active"] = bool(ppr_rev_z.any())

            # Renormalize hybrid weights so the base ordering always retains
            # primacy, then distribute the remainder across the graph terms.
            w_ppr = self.ppr_weight if ppr_z.any() else 0.0
            w_graph = self.alpha + self.beta + (self.gamma if q_has_signal else 0.0) + w_ppr
            w_graph = w_graph if w_graph > 0 else 1.0
            a = self.base_weight
            scale = (1.0 - a) / w_graph

            scored = []
            for i, note in enumerate(base_results):
                hybrid = (
                    a * rank_z[i]
                    + scale * self.alpha * local_z[i]
                    + scale * self.beta * global_z[i]
                    + scale * self.gamma * q_z[i]
                    + scale * w_ppr * ppr_z[i]
                )
                scored.append((float(hybrid), note))

            scored.sort(key=lambda x: x[0], reverse=True)
            base_results = [n for _, n in scored]
            self.stats["phase_d_q_has_signal"] = bool(q_has_signal)
            self.stats["phase_d_ppr_active"] = bool(ppr_z.any())

        self.stats["total_retrieved"] = len(base_results)
        return base_results

    # ------------------------------------------------------------------
    # G3: multi-hop sub-question fusion
    # ------------------------------------------------------------------

    def _fuse_subquestions(
        self, pools: List[List[Note]], subs: List[str]
    ) -> List[Note]:
        """Fuse per-sub-question retrieval pools by reciprocal rank fusion.

        The original query is ``subs[0]`` and is weighted 1.0; each further
        sub-question is weighted ``planner_fuse_weight``. RRF is used because it
        combines RANK positions rather than scores, so a sub-question's list
        (produced by a different query, hence a different score scale) is not
        drowned out by the original's — which is the whole point: a note the
        original query ranked 200th but a sub-question ranked 1st must be able
        to reach the answer window.

        Output order is by fused score, so a sub-question's top note interleaves
        with the original's rather than sitting behind it. Ties break on the
        original query's rank, then on the note id for determinism.
        """
        k = max(int(self.rrf_k), 1)
        weights = [1.0] + [float(self.planner_fuse_weight)] * (len(pools) - 1)

        scores: Dict[str, float] = {}
        first_seen: Dict[str, int] = {}   # index of the first pool that held it
        by_id: Dict[str, Note] = {}
        for pi, (pool, w) in enumerate(zip(pools, weights)):
            for rank, note in enumerate(pool):
                if note.id not in by_id:
                    by_id[note.id] = note
                    first_seen[note.id] = pi
                scores[note.id] = scores.get(note.id, 0.0) + w / (k + rank + 1)

        ordered = sorted(
            by_id.values(),
            key=lambda n: (-scores[n.id], first_seen[n.id], n.id),
        )
        return ordered

    # ------------------------------------------------------------------
    # G2: typed heterogeneous Personalized PageRank
    # ------------------------------------------------------------------

    def _resolve_edge_weights(self) -> Tuple[Dict[str, float], Optional[float]]:
        """The relation -> transition-weight table the walk uses, and its default.

        Returns ``(table, default_weight)``. ``ppr_edge_weights`` wins when set,
        so a single relation can be ablated from the config. Otherwise the table
        is DERIVED from ``relation_weights`` — the typed table that until now
        only the bounded BFS consulted. Deriving rather than duplicating means
        the two views of "how much does a ``superseded_by`` edge matter" can
        never drift apart, which is the failure the old split (BFS weighted, PPR
        uniform) produced.

        ``default_weight`` is ``None`` — an unnamed relation keeps the weight the
        graph built it with, so the semantic edges' cosine-similarity weights
        survive and only the named relations are re-weighted. ``None`` is also
        what a caller gets for a relation neither table names, matching the
        BFS's neutral treatment of unknown-typed links. When
        ``filter_relation_types`` is set the default becomes ``0.0``, which turns
        the table into an ALLOW-LIST: an unlisted relation carries no transition
        mass, reproducing the BFS's hard edge filter as a soft one.
        """
        if self.ppr_edge_weights is not None:
            table = dict(self.ppr_edge_weights)
        else:
            table = dict(self.relation_weights)
            # `entity_cooccurrence` is the note<->entity edge every note carries.
            # It must stay weak: it is the highest-degree edge in the graph (each
            # note touches all its entity nodes), so weighting it like a typed
            # relation would let the walk drift to hub entities instead of facts.
            table.setdefault("entity_cooccurrence", 0.5)
        default: Optional[float] = None
        if self.filter_relation_types:
            allowed = set(self.filter_relation_types)
            table = {rel: w for rel, w in table.items() if rel in allowed}
            default = 0.0
        return table, default

    def _query_entities(self, query: str) -> List[str]:
        """Entity nodes the QUESTION names, normalised the way the graph keys them.

        This is HippoRAG 2's query-side entry point: the walk's reset mass starts
        on the entities the question mentions, not only on the notes dense
        retrieval happened to find, which is what lets a bridging premise that
        shares no query terms with the question be reached at all.

        Capitalised-but-not-ALLCAPS tokens are the candidate names, filtered of
        question/function words (the same ``_MASK_KEEP`` vocabulary the masked
        query channel uses) and of anything the bank's entity index does not
        know. ``TemporalHyperGraph._entity_keys`` lowercases, so the seeds are
        lowercased here to match the graph's node ids.

        The function-word filter is what keeps this from degenerating: without
        it "What did Caroline say" seeds the entity "what", which sits on every
        note in the bank and would swamp the reset distribution.
        """
        from .retriever import _MASK_KEEP

        # `_MASK_KEEP` is capitalised ("What", "Did"), but a question can spell a
        # function word mid-sentence in lower case ("What did Caroline do in
        # Hawaii?"). Compare case-insensitively, or "did"/"do"/"in" survive the
        # filter and are dropped only incidentally by the graph-membership test
        # below — which stops working the moment such a token becomes an entity
        # key. The mask must do its own job.
        mask = {w.lower() for w in _MASK_KEEP}
        raw = re.findall(r"\b[A-Za-z][a-z0-9_-]+\b", query or "")
        seeds: List[str] = []
        for token in raw:
            if token.isupper() or token.lower() in mask:
                continue
            key = token.strip().lower()
            if key:
                seeds.append(key)
        # Only entities the graph actually holds can carry reset mass. The graph
        # exposes its entity index as `_entity_nodes`; when the attribute is
        # absent the filter is skipped and `_ppr_personalization` does the
        # validation instead (it drops any seed the walked graph does not have).
        graph = self.hyper_graph
        known = getattr(graph, "_entity_nodes", None)
        if known is not None:
            seeds = [s for s in seeds if s in known]
        return list(dict.fromkeys(seeds))[: self.entity_seed_topn]

    def _graph_rank(
        self,
        query: str,
        query_embedding: np.ndarray,
        base_results: List[Note],
        M: MemoryBank,
    ) -> List[Note]:
        """One typed PageRank over the hyper-graph; returns notes to ADD to the pool.

        Three things distinguish this from the ``_ppr_expand`` it replaces:

        1. **Typed transitions.** The edge weights come from
           :meth:`_resolve_edge_weights`, so the walk's mass follows
           ``superseded_by`` differently from ``semantic`` — the graph's type
           information now governs every path, not just the first hop of a BFS.
        2. **Entity-anchored reset.** Seeds are the base winners AND the entity
           nodes the question names (:meth:`_query_entities`), which is the only
           way the walk reaches a note that dense retrieval missed entirely.
        3. **One score vector, both directions.** The forward and (optional)
           reverse walks are recorded separately for Phase D, and their union is
           what gets fetched into the pool — so a premise the seed merely POINTS
           AT is admitted alongside what follows FROM it.

        The walk's output feeds Phase D, which re-ranks on similarity, utility
        and global centrality; this widens recall and supplies the query-conditional
        graph signal, it does not impose a final order.
        """
        self.ppr_scores = {}
        self.ppr_scores_reverse = {}
        if self.hyper_graph is None:
            return []
        seeds = [n.id for n in base_results[: max(self.multi_hop_topn * 2, 1)]]
        entity_seeds = self._query_entities(query)
        if not seeds and not entity_seeds:
            return []

        edge_weights, default_edge_weight = self._resolve_edge_weights()
        common = dict(
            top_k=self.ppr_topk,
            edge_weights=edge_weights,
            default_edge_weight=default_edge_weight,
            damping=self.ppr_damping,
            entity_seeds=entity_seeds or None,
            entity_seed_weight=self.entity_seed_weight,
        )
        try:
            ranked = self.hyper_graph.personalized_pagerank(seeds, **common)
        except Exception as exc:  # noqa: BLE001 — never fail retrieval over a graph walk
            _log.debug("Graph walk skipped: {}", exc)
            return []

        # Record the full ranking (seeds included) for the Phase D blend.
        self.ppr_scores = {nid: float(score) for nid, score in ranked}

        reverse_ranked: List[Tuple[str, float]] = []
        if self.ppr_reverse:
            try:
                reverse_ranked = self.hyper_graph.personalized_pagerank(
                    seeds, directions="in", **common
                )
                self.ppr_scores_reverse = {
                    nid: float(score) for nid, score in reverse_ranked
                }
            except Exception as exc:  # noqa: BLE001
                _log.debug("Reverse graph walk skipped: {}", exc)
                self.ppr_scores_reverse = {}

        self.stats["graph_rank_seeds"] = len(seeds)
        self.stats["graph_rank_entity_seeds"] = len(entity_seeds)
        self.stats["graph_rank_ranked"] = len(ranked)

        known = {n.id for n in base_results}
        # The union of both walks, in rank order, deduped. A premise note must
        # actually enter the pool before Phase D can rank it.
        wanted = list(dict.fromkeys(
            [nid for nid, _ in ranked] + [nid for nid, _ in reverse_ranked]
        ))
        wanted = [nid for nid in wanted if nid not in known][: self.graph_rank_topn]
        if not wanted:
            self.stats["graph_rank_added"] = 0
            return []
        try:
            fetched = M.get_notes_by_ids(wanted) if hasattr(M, "get_notes_by_ids") else []
        except Exception as exc:  # noqa: BLE001
            _log.debug("Graph walk fetch failed: {}", exc)
            return []

        by_id = {n.id: n for n in fetched}
        added: List[Note] = []
        for nid in wanted:  # preserve rank order, not the fetch order
            note = by_id.get(nid)
            if note is None or note.id in known:
                continue
            if self.graph_rank_min_sim > 0.0:
                if self._cosine(query_embedding, note.e) < self.graph_rank_min_sim:
                    continue
            added.append(note)
            known.add(note.id)
        self.stats["graph_rank_added"] = len(added)
        _log.debug(
            "Graph walk | seeds={}  entity_seeds={}  ranked={}  added={}",
            len(seeds), len(entity_seeds), len(ranked), len(added),
        )
        return added

    # ------------------------------------------------------------------
    # Global graph semantics
    # ------------------------------------------------------------------

    def _refresh_graph(self, M: MemoryBank) -> None:
        """Recompute community map and PageRank when the bank has changed."""
        current_hash = hash(tuple(sorted(n.id for n in M.list_notes())))
        if current_hash == self._graph_hash:
            return
        self._graph_hash = current_hash

        # Build networkx graph
        from .visualizer import MemoryGraphBuilder
        builder = MemoryGraphBuilder(M)
        graph = builder.build_graph()

        if graph.number_of_nodes() == 0:
            self._community_map = {}
            self._pagerank = {}
            return

        # Community detection (Louvain via networkx)
        try:
            import networkx as nx
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(graph, seed=42)
            self._community_map = {}
            for cid, community in enumerate(communities):
                for node_id in community:
                    self._community_map[node_id] = cid
            _log.debug(
                "Louvain communities | nodes={}  communities={}",
                graph.number_of_nodes(),
                len(communities),
            )
        except Exception:
            self._community_map = {}

        # PageRank
        try:
            import networkx as nx
            self._pagerank = dict(nx.pagerank(graph, alpha=0.85))
        except Exception:
            self._pagerank = {}

    def _compute_global_score(self, note: Note, top_note: Optional[Note]) -> float:
        """Compute graph-based score for a note (0.0–1.0)."""
        score = 0.5  # neutral baseline

        # Community boost: same community as the top-ranked note
        if top_note is not None and self._community_map:
            c_note = self._community_map.get(note.id)
            c_top = self._community_map.get(top_note.id)
            if c_note is not None and c_top is not None and c_note == c_top:
                score *= self.community_boost

        # PageRank centrality boost
        if self._pagerank:
            pr = self._pagerank.get(note.id, 0.0)
            # Normalize PageRank to [0, 1] using max value in the graph
            max_pr = max(self._pagerank.values()) if self._pagerank else 1.0
            pr_norm = pr / max_pr if max_pr > 0 else 0.0
            score = 0.5 * score + 0.5 * pr_norm

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(value: float, lo: float, hi: float) -> float:
        """Clamp and normalize a value to [0, 1]."""
        if hi == lo:
            return 0.5
        clamped = max(lo, min(hi, value))
        return (clamped - lo) / (hi - lo)

    @staticmethod
    def _zscore(values: np.ndarray) -> np.ndarray:
        """Z-score normalize a vector; returns zeros when variance is 0.

        Used to put local similarity, global graph score and utility on the
        same (standardized) footing so no single term dominates by magnitude.
        """
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return arr
        mu = float(arr.mean())
        sd = float(arr.std())
        if sd <= 1e-9:
            return np.zeros_like(arr)
        return (arr - mu) / sd
