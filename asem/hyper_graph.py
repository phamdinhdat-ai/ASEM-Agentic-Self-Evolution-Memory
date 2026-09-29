"""Temporal Hyper-Graph Memory (THG) representation and indexing.

The graph carries four deterministic (LLM-free) relation types:

1. ``same-entity``   -- note <-> note, bridged by the entity nodes they share
2. ``temporal``      -- chronological neighbours inside one session
3. ``semantic``      -- ``cosine(e_i, e_j) >= semantic_tau``
4. ``superseded_by`` -- the same (subject, predicate) seen again in a newer fact

Two views of the same graph are kept side by side:

* ``self._graph`` -- an in-memory ``networkx`` graph with explicit ENTITY nodes,
  which is what :meth:`personalized_pagerank` walks;
* the persisted note-to-note projection on ``Note.L``, which is what the
  retriever traverses and the only view that survives the process boundary
  between the ingest phase and the evaluation phase.

Every relation is written to BOTH views on purpose. A relation that lived only
in the nx graph would be silently lost at eval time -- which is how an earlier
version of this module ended up persisting just two of its four edge types.

All emitted edges are SYMMETRIC. The retriever's link traversal
(``HybridRetriever._traverse_links``) only follows outgoing links from a seed,
so a one-directional edge is invisible whenever the seed is the older endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is a hard dep in practice
    nx = None

from .logging_utils import get_logger
from .note import LinkRecord, Note

_log = get_logger("THG.graph")

# Relation labels. `temporal` and `same-entity` deliberately reuse the labels the
# deterministic FastASEM weaver already emits, so the shared retriever weights
# apply to both systems.
_REL_ENTITY = "same-entity"
_REL_TEMPORAL = "temporal"
_REL_SEMANTIC = "semantic"
_REL_SUPERSEDED = "superseded_by"


@dataclass
class Triplet:
    """Atomic fact triplet: (subject, predicate, object, timestamp)."""

    subject: str
    predicate: str
    object: str
    timestamp: str


class TemporalHyperGraph:
    """Deterministic Temporal Hyper-Graph for associative memory traversal."""

    def __init__(
        self,
        semantic_tau: float = 0.70,
        max_entity_links: int = 5,
        max_semantic_links: int = 5,
    ) -> None:
        self.semantic_tau = semantic_tau
        # Fan-out caps. A dominant entity (the conversation's own speaker name,
        # "Caroline") is shared by most notes, so an uncapped entity channel
        # would link every note to every other note and collapse the graph into
        # a clique. Peers are ranked by embedding similarity before the cut.
        self.max_entity_links = max_entity_links
        self.max_semantic_links = max_semantic_links

        self._graph = nx.Graph() if nx is not None else None
        self._fact_nodes: Dict[str, Note] = {}
        self._entity_nodes: Set[str] = set()
        self._entity_index: Dict[str, Set[str]] = {}  # note_id -> normalised entity keys
        self._subj_pred_index: Dict[Tuple[str, str], str] = {}  # (subj, pred) -> latest note id
        # Ids of notes whose ``L`` was extended as the *peer* of a link, so the
        # caller can re-persist them (they may pre-date this ingest call).
        self._touched: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_note(self, note: Note, triplet: Optional[Triplet] = None) -> List[LinkRecord]:
        """Insert ``note`` and return the links originating from it.

        The returned records are NOT applied to ``note.L`` -- the caller owns
        that, so it can decide the persistence order. Mirror edges pointing back
        at ``note`` are applied in place to the peer notes; collect those with
        :meth:`pop_touched_peers` and re-persist them.
        """
        self._fact_nodes[note.id] = note
        entity_keys = self._entity_keys(note)
        self._entity_index[note.id] = entity_keys

        created_links: List[LinkRecord] = []

        if self._graph is not None:
            self._graph.add_node(note.id, type="fact", date=note.session_date or "")
            for key in sorted(entity_keys):
                self._entity_nodes.add(key)
                self._graph.add_node(key, type="entity")
                self._graph.add_edge(note.id, key, relation="entity_cooccurrence", weight=1.0)

        # 1. same-entity peers (the note-to-note projection of the entity nodes)
        for peer_id, _sim in self._entity_peers(note, entity_keys):
            created_links.append(LinkRecord(target_id=peer_id, relation=_REL_ENTITY))
            self._link_back(peer_id, note.id, _REL_ENTITY)
            if self._graph is not None:
                self._graph.add_edge(note.id, peer_id, relation=_REL_ENTITY, weight=1.0)

        # 2. Deterministic temporal versioning
        if triplet and triplet.subject and triplet.predicate:
            sp_key = (self._norm(triplet.subject), self._norm(triplet.predicate))
            old_id = self._subj_pred_index.get(sp_key)
            if old_id and old_id != note.id and old_id in self._fact_nodes:
                created_links.append(LinkRecord(target_id=old_id, relation=_REL_SUPERSEDED))
                self._link_back(old_id, note.id, _REL_SUPERSEDED)
                if self._graph is not None:
                    self._graph.add_edge(
                        note.id, old_id, relation=_REL_SUPERSEDED, weight=1.5
                    )
                _log.debug("THG versioning | {} supersedes {}", note.id[:8], old_id[:8])
            self._subj_pred_index[sp_key] = note.id

        # 3. Semantic similarity (top matches above tau, capped)
        for other_id, sim in self._semantic_peers(note):
            created_links.append(LinkRecord(target_id=other_id, relation=_REL_SEMANTIC))
            self._link_back(other_id, note.id, _REL_SEMANTIC)
            if self._graph is not None:
                self._graph.add_edge(note.id, other_id, relation=_REL_SEMANTIC, weight=sim)

        return created_links

    def add_temporal_sequence(self, note_ids: List[str]) -> int:
        """Symmetric chronological edges between consecutive notes of a session.

        Returns the number of edges created.
        """
        created = 0
        for i in range(len(note_ids) - 1):
            a_id, b_id = note_ids[i], note_ids[i + 1]
            if a_id not in self._fact_nodes or b_id not in self._fact_nodes:
                continue
            self._link_both(a_id, b_id, _REL_TEMPORAL)
            if self._graph is not None:
                self._graph.add_edge(a_id, b_id, relation=_REL_TEMPORAL, weight=1.2)
            created += 1
        return created

    def pop_touched_peers(self) -> List[Note]:
        """Peer notes whose ``L`` was extended; the caller re-persists them."""
        peers = [self._fact_nodes[i] for i in self._touched if i in self._fact_nodes]
        self._touched = set()
        return peers

    def hydrate(self, notes: Iterable[Note]) -> int:
        """Load already-persisted notes into the graph WITHOUT creating links.

        The hyper-graph lives only in memory, but the notes it links live in
        SQLite. A resumed ingest (one that continues an interrupted conversation
        instead of rebuilding it from scratch) opens a bank that already holds
        thousands of notes, and every new note would otherwise be linked only to
        other new notes — silently producing a disconnected graph.

        Hydrating first restores the ``same-entity`` and ``semantic`` channels
        across the resume boundary, because both are computed by scanning
        ``_fact_nodes`` / ``_entity_index``.

        Known limitation: the ``(subject, predicate) -> note_id`` index used for
        ``superseded_by`` versioning is NOT rebuilt, because the triplet is not
        persisted on the note. Notes hydrated here can therefore be superseded
        only by a later hydrated note, never by a newly ingested one. Pass
        ``subj_pred`` to supply the triplets when the caller has them.
        """
        count = 0
        for note in notes:
            self._fact_nodes[note.id] = note
            self._entity_index[note.id] = self._entity_keys(note)
            if self._graph is not None:
                self._graph.add_node(note.id, type="fact", date=note.session_date or "")
            count += 1
        return count

    def get_note(self, note_id: str) -> Optional[Note]:
        return self._fact_nodes.get(note_id)

    def clear(self) -> None:
        """Forget every node and edge (used when the backing bank is reset)."""
        self._graph = nx.Graph() if nx is not None else None
        self._fact_nodes = {}
        self._entity_nodes = set()
        self._entity_index = {}
        self._subj_pred_index = {}
        self._touched = set()

    def __len__(self) -> int:
        return len(self._fact_nodes)

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    def _entity_peers(self, note: Note, entity_keys: Set[str]) -> List[Tuple[str, float]]:
        """Existing notes sharing >= 1 entity, best embedding match first."""
        if not entity_keys:
            return []
        scored: Dict[str, float] = {}
        for other_id, other in self._fact_nodes.items():
            if other_id == note.id:
                continue
            if not (self._entity_index.get(other_id) or set()) & entity_keys:
                continue
            scored[other_id] = self._cosine(note.e, other.e)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[: max(0, self.max_entity_links)]

    def _semantic_peers(self, note: Note) -> List[Tuple[str, float]]:
        """Existing notes at/above ``semantic_tau``, most similar first."""
        if note.e is None:
            return []
        similar: List[Tuple[str, float]] = []
        for other_id, other in self._fact_nodes.items():
            if other_id == note.id or other.e is None:
                continue
            sim = self._cosine(note.e, other.e)
            if sim >= self.semantic_tau:
                similar.append((other_id, sim))
        similar.sort(key=lambda item: item[1], reverse=True)
        return similar[: max(0, self.max_semantic_links)]

    # ------------------------------------------------------------------
    # Link bookkeeping
    # ------------------------------------------------------------------

    def _link_back(self, peer_id: str, src_id: str, relation: str) -> None:
        """Apply the mirror edge ``peer -> src`` in place, and mark the peer."""
        peer = self._fact_nodes.get(peer_id)
        if peer is None:
            return
        if not any(
            link.target_id == src_id and link.relation == relation for link in peer.L
        ):
            peer.L.append(LinkRecord(target_id=src_id, relation=relation))
        self._touched.add(peer_id)

    def _link_both(self, a_id: str, b_id: str, relation: str) -> None:
        """Symmetric edge between two notes already known to the graph."""
        a = self._fact_nodes.get(a_id)
        if a is None:
            return
        if not any(
            link.target_id == b_id and link.relation == relation for link in a.L
        ):
            a.L.append(LinkRecord(target_id=b_id, relation=relation))
        self._touched.add(a_id)
        self._link_back(b_id, a_id, relation)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_keys(note: Note) -> Set[str]:
        keys = {TemporalHyperGraph._norm(e) for e in (note.entities or [])}
        keys |= {TemporalHyperGraph._norm(k) for k in (note.K or [])}
        keys.discard("")
        return keys

    @staticmethod
    def _norm(text: str) -> str:
        return str(text or "").strip().lower()

    @staticmethod
    def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype="float32").reshape(-1)
        b = np.asarray(b, dtype="float32").reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def personalized_pagerank(self, seed_note_ids: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Run Personalized PageRank (PPR) over hyper-graph from seed notes."""
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return [(nid, 1.0) for nid in seed_note_ids[:top_k]]

        valid_seeds = [nid for nid in seed_note_ids if self._graph.has_node(nid)]
        if not valid_seeds:
            return []

        personalization = {nid: (1.0 / len(valid_seeds)) for nid in valid_seeds}
        try:
            pr_scores = nx.pagerank(self._graph, alpha=0.85, personalization=personalization, max_iter=100)
            # Filter only fact nodes
            fact_scores = [(nid, score) for nid, score in pr_scores.items() if nid in self._fact_nodes]
            fact_scores.sort(key=lambda x: x[1], reverse=True)
            return fact_scores[:top_k]
        except Exception as exc:
            _log.warning("PPR computation fallback: {}", exc)
            return [(nid, 1.0) for nid in valid_seeds[:top_k]]

