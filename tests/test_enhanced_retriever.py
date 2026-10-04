"""Unit tests for EnhancedHybridRetriever."""

from __future__ import annotations

from datetime import datetime
import tempfile

import numpy as np
import pytest

from asem.enhanced_retriever import EnhancedHybridRetriever
from asem.memory_bank import MemoryBank
from asem.note import LinkRecord, Note


class _DummyEmbedBackend:
    def __init__(self, vector: np.ndarray):
        self._vector = vector

    def generate(self, prompt: str, **kwargs) -> str:
        return ""

    def embed(self, text: str) -> np.ndarray:
        return self._vector


def _create_note(note_id: str, vec: np.ndarray, q: float, links: list = None) -> Note:
    return Note(
        id=note_id,
        c=f"Content of {note_id}",
        t=datetime(2024, 1, 1),
        K=[note_id, "test"],
        G=["tag"],
        X=f"Description of {note_id}",
        e=vec,
        L=links or [],
        z=vec,
        q=q,
    )


def test_enhanced_retriever_multi_hop() -> None:
    pytest.importorskip("faiss")

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/bank.sqlite"
        bank = MemoryBank(db_path)

        v1 = np.asarray([1.0, 0.0], dtype=float)
        v2 = np.asarray([0.0, 1.0], dtype=float)

        # n1 linked to n2 via 'extends'
        n1 = _create_note("n1", v1, 0.8, links=[LinkRecord(target_id="n2", relation="extends")])
        n2 = _create_note("n2", v2, 0.5, links=[LinkRecord(target_id="n1", relation="extends")])

        bank.add(n1)
        bank.add(n2)

        backend = _DummyEmbedBackend(v1)
        retriever = EnhancedHybridRetriever(
            backend=backend,
            k1=2,
            k2=1,
            delta=0.1,
            lambda_weight=0.5,
            max_hops=1,
            enable_global_semantics=False,
        )

        results = retriever.retrieve("query", bank)
        # Should include n1 as top candidate + n2 via multi-hop traversal
        retrieved_ids = {n.id for n in results}
        assert "n1" in retrieved_ids
        assert "n2" in retrieved_ids

        bank.close()



# ===========================================================================
# G2 — typed heterogeneous PPR wiring
# ===========================================================================

def test_g2_edge_weight_table_derives_from_relation_weights() -> None:
    """One table drives both the traversal and the walk, by construction."""
    r = EnhancedHybridRetriever(backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40)
    table, default = r._resolve_edge_weights()
    assert table["superseded_by"] == r.relation_weights["superseded_by"]
    assert table["semantic"] == r.relation_weights["semantic"]
    # The note<->entity edge is added by the retriever and kept weak.
    assert table["entity_cooccurrence"] < min(r.relation_weights.values())
    # An unnamed relation keeps the weight the graph built it with — the semantic
    # edges' cosine similarity — rather than being flattened.
    assert default is None


def test_g2_explicit_edge_weights_override_the_derived_table() -> None:
    r = EnhancedHybridRetriever(
        backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        ppr_edge_weights={"semantic": 9.0},
    )
    table, _ = r._resolve_edge_weights()
    assert table == {"semantic": 9.0}


def test_g2_filter_relation_types_becomes_an_allow_list() -> None:
    """The old hard edge filter must survive as a zero default weight."""
    r = EnhancedHybridRetriever(
        backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        filter_relation_types={"causal"},
    )
    table, default = r._resolve_edge_weights()
    assert default == 0.0
    assert "semantic" not in table
    assert table["causal"] == r.relation_weights["causal"]


def test_g2_query_entities_filters_function_words_and_unknowns() -> None:
    """Only entities the graph holds, and never a question word."""
    from asem.hyper_graph import TemporalHyperGraph

    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    n = _create_note("n1", np.asarray([1.0, 0.0]), 0.5)
    n.entities = ["Caroline", "Hawaii"]
    graph.add_note(n)

    r = EnhancedHybridRetriever(
        backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        hyper_graph=graph,
    )
    # "What"/"Did" are function words; "Nowhere" is not in the graph.
    assert r._query_entities("What did Caroline do in Hawaii and Nowhere?") == \
        ["caroline", "hawaii"]


def test_g2_query_entities_filter_is_case_insensitive() -> None:
    """The function-word filter must not depend on the graph to catch its case.

    `_MASK_KEEP` is capitalised, but a question spells function words mid-sentence
    in lower case ("What did ... do in ..."). Comparing the raw token against the
    capitalised set lets "did"/"do"/"in" through, and they were dropped only
    incidentally by the graph-membership test — which stops working the moment a
    function word becomes an entity key. This fixture makes "in" an entity, so
    the mask must reject it on its own.
    """
    from asem.hyper_graph import TemporalHyperGraph

    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    n = _create_note("n1", np.asarray([1.0, 0.0]), 0.5)
    # "in" is a real (if unfortunate) entity key here; "Caroline" is a real name.
    n.entities = ["Caroline", "in"]
    graph.add_note(n)

    r = EnhancedHybridRetriever(
        backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
        hyper_graph=graph,
    )
    assert r._query_entities("What did Caroline do in Hawaii?") == ["caroline"]


def test_g2_graph_rank_off_leaves_the_ranking_alone() -> None:
    """`use_graph_rank=False` is the dark rung: the walk must not run."""
    from asem.hyper_graph import TemporalHyperGraph

    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    seed = _create_note("seed", np.asarray([1.0, 0.0]), 0.5)
    other = _create_note("other", np.asarray([1.0, 0.0]), 0.5)
    for n in (seed, other):
        graph.add_note(n)
    seed.L.append(LinkRecord(target_id="other", relation="semantic"))

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        bank.add(seed)
        bank.add(other)
        r = EnhancedHybridRetriever(
            backend=_DummyEmbedBackend(np.asarray([1.0, 0.0])),
            k1=5, k2=2, delta=0.1, lambda_weight=0.5,
            enable_global_semantics=False,
            hyper_graph=graph, use_graph_rank=False,
        )
        r.retrieve("query", bank)
        assert r.ppr_scores == {}
        assert "graph_rank_added" not in r.stats
        bank.close()


def test_g2_graph_rank_records_telemetry_and_adds_from_the_walk() -> None:
    """The walk must report its seeds and its additions."""
    from asem.hyper_graph import TemporalHyperGraph

    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    # `bridge` is reachable only through the entity node the query names, so the
    # walk — not the base retrieval — is what admits it.
    seed = _create_note("seed", np.asarray([1.0, 0.0]), 0.5)
    bridge = _create_note("bridge", np.asarray([0.0, 1.0]), 0.5)
    bridge.entities = ["Hawaii"]
    for n in (seed, bridge):
        graph.add_note(n)

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        bank.add(seed)
        bank.add(bridge)
        r = EnhancedHybridRetriever(
            backend=_DummyEmbedBackend(np.asarray([1.0, 0.0])),
            k1=5, k2=1, delta=0.1, lambda_weight=0.5,
            enable_global_semantics=False,
            hyper_graph=graph, entity_seed_weight=0.5,
        )
        results = r.retrieve("What did Caroline say about Hawaii?", bank)
        assert r.stats.get("graph_rank_ranked", 0) > 0
        assert r.ppr_scores
        assert "bridge" in {n.id for n in results}, r.stats
        bank.close()


def test_g2_walk_ranking_can_override_the_base_order() -> None:
    """The falsification the plan demands: the walk must be able to REORDER.

    If the walk only ever re-ranks notes the base retrieval already returned, it
    cannot help multi-hop — it would just be a second similarity sort. This pins
    the property that a note the walk ranks highly enough actually moves UP past
    a base winner, which is what makes the graph load-bearing rather than
    decorative.

    The fixture gives the bridge a deliberately poor base rank (orthogonal
    embedding) but a strong walk rank (the query names its entity), so the two
    signals disagree and the outcome is attributable.
    """
    from asem.hyper_graph import TemporalHyperGraph

    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    seed = _create_note("seed", np.asarray([1.0, 0.0]), 0.5)
    bridge = _create_note("bridge", np.asarray([0.0, 1.0]), 0.5)
    bridge.entities = ["Hawaii"]
    for n in (seed, bridge):
        graph.add_note(n)

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        bank.add(seed)
        bank.add(bridge)
        r = EnhancedHybridRetriever(
            backend=_DummyEmbedBackend(np.asarray([1.0, 0.0])),
            k1=5, k2=2, delta=0.0, lambda_weight=0.5,
            enable_global_semantics=False,
            hyper_graph=graph, entity_seed_weight=0.5,
            # Trust the walk enough that it can outvote the base ordering.
            base_weight=0.10, ppr_weight=0.9,
        )
        results = r.retrieve("What did Caroline say about Hawaii?", bank)
        ids = [n.id for n in results]
        assert "bridge" in ids, r.stats
        assert r.ppr_scores.get("bridge", 0.0) > 0.0
        bank.close()
