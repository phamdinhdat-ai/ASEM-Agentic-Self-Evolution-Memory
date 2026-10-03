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


def test_all_four_relation_types_reach_the_persisted_projection() -> None:
    """Every edge type must be emitted as a LinkRecord, not just kept in the nx graph.

    The evaluation phase runs in a separate process and only sees ``Note.L``, so a
    relation that lives exclusively in ``_graph`` is lost.
    """
    graph = TemporalHyperGraph(semantic_tau=0.7)
    v = np.asarray([1.0, 0.0], dtype=float)
    entities = ["Caroline", "Hawaii"]

    n1 = _make_test_note("n1", "Caroline visits Hawaii", entities, v)
    graph.add_note(n1, Triplet("Caroline", "visits", "Hawaii", "2023-05-01"))

    n2 = _make_test_note("n2", "Caroline visited Hawaii in May", entities, v)
    links = graph.add_note(n2, Triplet("Caroline", "visits", "Hawaii", "2023-05-02"))

    assert {"same-entity", "semantic", "superseded_by"} <= {l.relation for l in links}

    n3 = _make_test_note("n3", "Caroline left Hawaii in June", entities, v)
    graph.add_note(n3, Triplet("Caroline", "left", "Hawaii", "2023-06-01"))
    assert graph.add_temporal_sequence(["n1", "n2", "n3"]) == 2
    assert any(l.relation == "temporal" for l in n1.L)


def test_links_are_symmetric_and_touched_peers_are_reported() -> None:
    """Mirror edges are written onto the peer, and the peer is reported for re-persist."""
    graph = TemporalHyperGraph(semantic_tau=0.7)
    v = np.asarray([1.0, 0.0], dtype=float)

    n1 = _make_test_note("n1", "Caroline likes tea", ["Caroline"], v)
    graph.add_note(n1)
    assert graph.pop_touched_peers() == []  # nothing to mirror for the first note

    n2 = _make_test_note("n2", "Caroline likes coffee", ["Caroline"], v)
    graph.add_note(n2)

    assert any(l.target_id == "n2" for l in n1.L), n1.L
    touched = {n.id for n in graph.pop_touched_peers()}
    assert "n1" in touched
    assert graph.pop_touched_peers() == []  # draining clears the buffer


def test_entity_fanout_is_capped() -> None:
    """A dominant shared entity must not turn the graph into a clique."""
    graph = TemporalHyperGraph(
        semantic_tau=2.0, max_entity_links=2, max_semantic_links=0
    )
    v = np.asarray([1.0, 0.0], dtype=float)

    for i in range(6):
        peer = _make_test_note(f"p{i}", f"fact {i}", ["Shared"], v)
        graph.add_note(peer)

    links = graph.add_note(_make_test_note("new", "new fact", ["Shared"], v))
    assert sum(1 for l in links if l.relation == "same-entity") == 2
    assert all(l.relation == "same-entity" for l in links)


def test_versioning_requires_a_usable_triplet() -> None:
    """A missing subject/predicate must not overwrite the version chain holder."""
    graph = TemporalHyperGraph()
    v = np.asarray([1.0, 0.0], dtype=float)

    graph.add_note(_make_test_note("n1", "Caroline lives in Sweden", ["Caroline"], v))
    links = graph.add_note(
        _make_test_note("n2", "Caroline lives in Norway", ["Caroline"], v)
    )
    # Neither note carried a triplet, so nothing may claim to supersede the other.
    assert not any(l.relation == "superseded_by" for l in links)

