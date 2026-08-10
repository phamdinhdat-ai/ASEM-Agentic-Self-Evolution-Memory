"""Enhanced hybrid retrieval with global graph semantics and true multi-hop traversal.

Extends the base HybridRetriever with:

* **Global semantic signals**: Louvain community detection + PageRank centrality
* **True N-hop traversal**: Recursive BFS with configurable depth and decay factor
* **Intent-grounded Q-values**: Query-to-z similarity gates utility relevance
* **Hybrid scoring**: ``alpha*local + beta*global + gamma*utility``

The original phases A+B are preserved (ANN recall + z-score composite re-rank).
Phase C is upgraded from 1-hop to configurable N-hop with graph-aware scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

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
    and adds graph-based re-ranking, community-aware boosting, true N-hop
    traversal with decay, and intent-grounded utility scoring.

    New knobs (in addition to inherited ones):
        alpha: Weight for local semantic similarity (default 0.35).
        beta:  Weight for global graph score (default 0.25).
        gamma: Weight for learned utility (default 0.40).
        max_hops: Maximum depth for multi-hop link traversal (default 2).
        hop_decay: Decay factor per hop, applied as ``decay ** hop`` (default 0.7).
        community_boost: Multiplier for same-community notes (default 1.2).
        enable_intent_q: Gate Q-values by query-to-z similarity (default True).
        enable_global_semantics: Toggle community + PageRank (default True).
    """

    # ── Hybrid weights ──────────────────────────────────────────────────
    alpha: float = 0.35   # local semantic
    beta: float = 0.25    # global graph
    gamma: float = 0.40   # learned utility

    # ── Multi-hop ───────────────────────────────────────────────────────
    max_hops: int = 2
    hop_decay: float = 0.7
    multi_hop_topn: int = 5

    # ── Relation-type-aware traversal ───────────────────────────────────
    # Weights applied to link traversal scores per relation type.  Strong
    # relations (contradicts/extends/causal) carry more information than
    # weak ones (same-topic/temporal/semantic), so traversed neighbors
    # reached via strong edges are preferred.
    relation_weights: Dict[str, float] = field(default_factory=lambda: {
        "contradicts": 1.2,
        "extends": 1.1,
        "causal": 1.1,
        "same-topic": 0.8,
        "temporal": 0.7,
        "semantic": 0.6,
        "linked": 1.0,  # legacy links (unknown type)
    })
    # If set, traversal only follows edges whose relation is in this set.
    filter_relation_types: Optional[Set[str]] = None

    # ── Global semantics ────────────────────────────────────────────────
    community_boost: float = 1.2
    enable_intent_q: bool = True
    enable_global_semantics: bool = True

    # ── Cached graph state (recomputed when bank changes) ───────────────
    _graph_hash: int = field(default=-1, init=False, repr=False)
    _community_map: Dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _pagerank: Dict[str, float] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------
    # Override: retrieve()
    # ------------------------------------------------------------------

    def retrieve(self, query: str, M: MemoryBank) -> List[Note]:
        """Enhanced retrieval with global graph semantics + true multi-hop."""
        self.stats = {}

        # Run base Phase A+B (ANN + composite z-score re-rank)
        base_results = super().retrieve(query, M)
        if not base_results:
            return []

        e_q = self.backend.embed(query)

        # Refresh graph metrics if bank has changed
        if self.enable_global_semantics:
            self._refresh_graph(M)

        # ── Phase C-Enhanced: True multi-hop traversal ──────────────
        if self.enable_link_traversal and self.max_hops > 0:
            linked = self._multi_hop_expand(e_q, base_results, M)
            seen_ids = {n.id for n in base_results}
            for n in linked:
                if n.id not in seen_ids:
                    base_results.append(n)
                    seen_ids.add(n.id)
            self.stats["multi_hop_added"] = len(linked)
            self.stats["max_hops"] = self.max_hops

        # ── Phase D (NEW): Global graph re-ranking ──────────────────
        if self.enable_global_semantics and self._community_map:
            scored = []
            for note in base_results:
                local = self._cosine(e_q, note.e)

                # Intent-grounded Q-value
                if self.enable_intent_q:
                    intent_sim = self._cosine(e_q, note.z)
                    q_effective = note.q * intent_sim
                else:
                    q_effective = note.q

                # Global graph score
                global_score = self._compute_global_score(
                    note, base_results[0] if base_results else None
                )

                # Hybrid composite
                hybrid = (
                    self.alpha * self._norm(local, 0.0, 1.0)
                    + self.beta * global_score
                    + self.gamma * self._norm(q_effective, 0.0, 1.0)
                )
                scored.append((hybrid, note))

            scored.sort(key=lambda x: x[0], reverse=True)
            base_results = [n for _, n in scored]

        self.stats["total_retrieved"] = len(base_results)
        return base_results

    # ------------------------------------------------------------------
    # True multi-hop expansion (BFS with decay)
    # ------------------------------------------------------------------

    def _multi_hop_expand(
        self,
        query_embedding: np.ndarray,
        seed_notes: List[Note],
        M: MemoryBank,
    ) -> List[Note]:
        """BFS traversal of the link graph up to ``max_hops`` depth."""
        visited: Set[str] = {n.id for n in seed_notes}
        frontier: Set[str] = {n.id for n in seed_notes[: self.link_traversal_topn]}
        all_candidates: List[Tuple[float, Note]] = []

        for hop in range(1, self.max_hops + 1):
            next_frontier: Set[str] = set()
            decay = self.hop_decay ** hop

            # Batch-lookup all frontier neighbors
            frontier_notes = M.get_notes_by_ids(list(frontier))
            for note in frontier_notes:
                if not note.L:
                    continue
                # Relation-type-aware edge filtering + weighting
                edge_map = {l.target_id: l.relation for l in note.L}
                if self.filter_relation_types:
                    edge_map = {
                        tid: rel for tid, rel in edge_map.items()
                        if rel in self.filter_relation_types
                    }
                if not edge_map:
                    continue
                linked = M.get_notes_by_ids(list(edge_map.keys()))
                for neighbor in linked:
                    if neighbor.id in visited:
                        continue
                    visited.add(neighbor.id)
                    next_frontier.add(neighbor.id)

                    sim = self._cosine(query_embedding, neighbor.e)
                    q_eff = neighbor.q
                    if self.enable_intent_q:
                        q_eff *= self._cosine(query_embedding, neighbor.z)

                    rel_weight = self.relation_weights.get(
                        edge_map.get(neighbor.id, ""), 0.5
                    )
                    score = sim * (0.5 + 0.5 * q_eff) * decay * rel_weight
                    all_candidates.append((score, neighbor))

            frontier = next_frontier
            if not frontier:
                break

        all_candidates.sort(key=lambda x: x[0], reverse=True)
        added = [note for _, note in all_candidates[: self.multi_hop_topn]]
        _log.debug(
            "Multi-hop | hops={}  candidates_found={}  added={}",
            self.max_hops,
            len(all_candidates),
            len(added),
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
