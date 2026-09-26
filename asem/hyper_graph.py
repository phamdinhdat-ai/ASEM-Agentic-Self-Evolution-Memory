"""Temporal Hyper-Graph Memory (THG) representation and indexing.

Constructs a graph of Fact Nodes & Entity Nodes with 4 deterministic edge types:
1. Entity Co-occurrence: Connects Fact -> Entity
2. Temporal Sequence: Connects Fact(t_i) -> Fact(t_{i+1})
3. Semantic Similarity: Connects Fact(i) -> Fact(j) when Cosine >= tau
4. Temporal Versioning: Connects Fact_old -> Fact_new via 'superseded_by' when
   (Subject, Predicate) is updated with a newer timestamp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
try:
    import networkx as nx
except ImportError:
    nx = None

from .logging_utils import get_logger
from .note import LinkRecord, Note

_log = get_logger("THG.graph")


@dataclass
class Triplet:
    """Atomic fact triplet: (subject, predicate, object, timestamp)."""
    subject: str
    predicate: str
    object: str
    timestamp: str


class TemporalHyperGraph:
    """Deterministic Temporal Hyper-Graph for associative memory traversal."""

    def __init__(self, semantic_tau: float = 0.70) -> None:
        self.semantic_tau = semantic_tau
        self._graph = nx.Graph() if nx is not None else None
        self._fact_nodes: Dict[str, Note] = {}
        self._entity_nodes: Set[str] = set()
        self._subj_pred_index: Dict[Tuple[str, str], str] = {}  # (subj, pred) -> latest_note_id

    def add_note(self, note: Note, triplet: Optional[Triplet] = None) -> List[LinkRecord]:
        """Insert note into hyper-graph and return deterministically generated links."""
        self._fact_nodes[note.id] = note
        created_links: List[LinkRecord] = []

        if self._graph is not None:
            self._graph.add_node(note.id, type="fact", date=note.session_date or "")

        # 1. Entity Co-occurrence Edges
        entities = list(dict.fromkeys((note.entities or []) + (note.K or [])))
        for ent in entities:
            ent_clean = str(ent).strip().lower()
            if not ent_clean:
                continue
            self._entity_nodes.add(ent_clean)
            if self._graph is not None:
                self._graph.add_node(ent_clean, type="entity")
                self._graph.add_edge(note.id, ent_clean, relation="entity_cooccurrence", weight=1.0)

        # 2. Deterministic Temporal Versioning / Superseded_by
        if triplet and triplet.subject and triplet.predicate:
            sp_key = (triplet.subject.lower().strip(), triplet.predicate.lower().strip())
            if sp_key in self._subj_pred_index:
                old_id = self._subj_pred_index[sp_key]
                if old_id != note.id and old_id in self._fact_nodes:
                    created_links.append(LinkRecord(target_id=old_id, relation="superseded_by"))
                    if self._graph is not None:
                        self._graph.add_edge(note.id, old_id, relation="superseded_by", weight=1.5)
                    _log.debug("THG versioning | note {} supersedes {}", note.id[:8], old_id[:8])
            self._subj_pred_index[sp_key] = note.id

        # 3. Semantic Similarity Edges
        if note.e is not None:
            for other_id, other_note in self._fact_nodes.items():
                if other_id == note.id or other_note.e is None:
                    continue
                sim = float(np.dot(note.e, other_note.e) / (np.linalg.norm(note.e) * np.linalg.norm(other_note.e) + 1e-9))
                if sim >= self.semantic_tau:
                    created_links.append(LinkRecord(target_id=other_id, relation="semantic"))
                    if self._graph is not None:
                        self._graph.add_edge(note.id, other_id, relation="semantic", weight=sim)

        return created_links

    def add_temporal_sequence(self, note_ids: List[str]) -> None:
        """Connect chronologically sequential notes within a session."""
        if not note_ids or self._graph is None:
            return
        for i in range(len(note_ids) - 1):
            n1, n2 = note_ids[i], note_ids[i + 1]
            if n1 in self._fact_nodes and n2 in self._fact_nodes:
                self._graph.add_edge(n1, n2, relation="temporal_sequence", weight=1.2)

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

