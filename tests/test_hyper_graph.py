"""Unit tests for TemporalHyperGraph."""

from __future__ import annotations

from datetime import datetime
import numpy as np
import pytest

from asem.hyper_graph import TemporalHyperGraph, Triplet
from asem.note import Note


def _make_test_note(nid: str, text: str, entities: list, vec: np.ndarray) -> Note:
    return Note(
        id=nid,
        c=text,
        t=datetime(2024, 1, 1),
        K=entities,
        G=["tag"],
        X=text,
        e=vec,
        L=[],
        z=vec,
        q=0.5,
        entities=entities,
    )


def test_hyper_graph_deterministic_linking() -> None:
    graph = TemporalHyperGraph(semantic_tau=0.7)

    v1 = np.asarray([1.0, 0.0], dtype=float)
    v2 = np.asarray([0.95, 0.05], dtype=float)  # High similarity with v1

    n1 = _make_test_note("n1", "Caroline visited Hawaii", ["Caroline", "Hawaii"], v1)
    n2 = _make_test_note("n2", "Caroline went to Hawaii again", ["Caroline", "Hawaii"], v2)

    # Add n1 with Triplet (Caroline, visited, Hawaii)
    t1 = Triplet("Caroline", "visited", "Hawaii", "2023-05-01")
    links1 = graph.add_note(n1, t1)
    assert len(links1) == 0

    # Add n2 with Triplet (Caroline, visited, Hawaii) - should detect semantic & versioning
    t2 = Triplet("Caroline", "visited", "Hawaii", "2023-05-10")
    links2 = graph.add_note(n2, t2)

    # Should contain semantic link or superseded_by link
    rel_types = {link.relation for link in links2}
    assert "superseded_by" in rel_types or "semantic" in rel_types


def test_hyper_graph_pagerank() -> None:
    graph = TemporalHyperGraph()
    v = np.asarray([1.0, 0.0], dtype=float)

    n1 = _make_test_note("n1", "Fact 1", ["Alex"], v)
    n2 = _make_test_note("n2", "Fact 2", ["Alex"], v)

    graph.add_note(n1)
    graph.add_note(n2)
    graph.add_temporal_sequence(["n1", "n2"])

    top_nodes = graph.personalized_pagerank(["n1"], top_k=2)
    assert len(top_nodes) > 0
    ids = [nid for nid, score in top_nodes]
    assert "n1" in ids

