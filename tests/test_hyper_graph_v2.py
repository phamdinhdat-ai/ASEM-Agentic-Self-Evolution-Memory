"""WS1 graph-builder correctness fixes (GB1-GB5).

Every test here FAILS on the pre-fix baseline `git show db92e20:asem/hyper_graph.py`
(or the equivalent baseline for note.py / memory_bank.py). The frozen-fixture
tests copy a real bank to a temp path first, because the GB1 migration is a write
and the fixture must stay read-only.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
    C:\\Users\\Dat Pham\\anaconda3\\envs\\memory-r1\\python.exe -m pytest tests/test_hyper_graph_v2.py -q
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from asem.hyper_graph import TemporalHyperGraph, Triplet
from asem.memory_bank import MemoryBank
from asem.note import Note, TripletRecord

REPO_ROOT = Path(__file__).resolve().parent.parent
FROZEN_BANK = (
    REPO_ROOT
    / "static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000/asem_thg.sqlite"
)


def _note(
    nid: str,
    text: str = "a fact",
    iso: str = "2022-03-01T00:00:00Z",
    entities=None,
    vec=None,
    session_id: str | None = None,
) -> Note:
    """Deterministic, backend-free Note. No StubBackend, no embeddings model."""
    v = np.asarray(vec if vec is not None else [1.0, 0.0], dtype=float)
    return Note(
        id=nid,
        c=text,
        t=datetime.fromisoformat(iso.replace("Z", "+00:00")),
        K=[],
        G=[],
        X=text,
        e=v,
        L=[],
        z=v,
        q=0.5,
        session_id=session_id,
        session_date=iso,
        timestamp_iso=iso,
        entities=list(entities or []),
    )


def _add(graph: TemporalHyperGraph, note: Note, triplet: Triplet | None = None):
    links = graph.add_note(note, triplet)
    note.L.extend(links)
    return links


def _supersedes(links) -> bool:
    return any(link.relation == "superseded_by" for link in links)


# ===========================================================================
# GB1 — the triplet is persisted, so a fresh note CAN supersede a hydrated one
# ===========================================================================

def test_gb1_hydrated_note_can_be_superseded_by_a_fresh_note() -> None:
    """GB1: hydrate() rebuilds the versioning index from persisted triplets.

    PRE-FIX: `hydrate()` rebuilt no versioning index (the triplet was not
    persisted anywhere), so a note that pre-dates the current process could never
    be superseded — measured as `links == []`.
    """
    graph = TemporalHyperGraph(enable_supersession=True)
    old = _note("n1", "Caroline lives in Boston", "2022-03-01T00:00:00Z",
                entities=["Caroline"])
    old.triplets = [TripletRecord("Caroline", "lives in", "Boston",
                                  "2022-03-01T00:00:00Z")]
    graph.hydrate([old])

    new = _note("n2", "Caroline lives in Seattle", "2022-03-03T00:00:00Z",
                entities=["Caroline"])
    links = graph.add_note(new, Triplet("Caroline", "lives in", "Seattle",
                                        "2022-03-03"))

    assert _supersedes(links), (
        "a freshly ingested note must be able to supersede a hydrated one"
    )
    sup = next(l for l in links if l.relation == "superseded_by")
    assert sup.target_id == "n1"
    assert sup.valid_from == "2022-03-03T00:00:00Z"


def test_gb1_triplet_survives_a_sqlite_round_trip() -> None:
    """GB1's premise: the triplet must persist, or hydrate() has nothing to read."""
    bank = MemoryBank(str(Path(tempfile.mkdtemp()) / "gb1.sqlite"))
    try:
        note = _note("n1", "Caroline lives in Boston", entities=["Caroline"])
        note.triplets = [TripletRecord("Caroline", "lives in", "Boston", "2022-03-01")]
        bank.add_many([note])
        back = bank.get_notes_by_ids(["n1"])[0]
        assert back.triplets == note.triplets
    finally:
        bank.close()


def test_gb1_note_without_triplets_omits_the_key() -> None:
    """Frozen-bank contract: no triplet -> no new key in the payload."""
    assert "triplets" not in _note("n1").to_dict()
    assert Note.from_dict({**_note("n1").to_dict(), "triplets": None}).triplets == []


def test_gb1_hydrated_legacy_note_is_not_superseded() -> None:
    """A pre-GB1 note carries no triplet: nothing may be claimed about it."""
    graph = TemporalHyperGraph(enable_supersession=True)
    graph.hydrate([_note("n1", "Caroline lives in Boston", entities=["Caroline"])])
    links = graph.add_note(
        _note("n2", "Caroline lives in Seattle"), Triplet("Caroline", "lives", "Seattle", "d")
    )
    assert not _supersedes(links)


# ===========================================================================
# GB2 — functional-predicate allow-list (the audit-driven redesign)
# ===========================================================================

def test_gb2_lives_in_is_superseded_by_a_different_place() -> None:
    """(a) genuine replacement of a functional attribute."""
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1"), Triplet("Caroline", "lives in", "Boston", "d"))
    links = _add(graph, _note("n2", "2022-03-03T00:00:00Z"),
                 Triplet("Caroline", "lives in", "Seattle", "d"))
    assert _supersedes(links)


def test_gb2_terse_one_word_predicate_is_reachable() -> None:
    """The allow-list must be reachable from what the extractor emits: `lives`.

    This is the corpus-reachability half of GB2. The old list held only
    multi-word phrases, so a terse `lives` matched nothing.
    """
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1"), Triplet("Caroline", "lives", "Boston", "d"))
    links = _add(graph, _note("n2"), Triplet("Caroline", "lives", "Seattle", "d"))
    assert _supersedes(links)


def test_gb2_bought_a_car_is_not_superseded_by_a_house() -> None:
    """(b) multi-valued verb: two purchases are two facts."""
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1"), Triplet("Caroline", "bought", "car", "d"))
    links = _add(graph, _note("n2"), Triplet("Caroline", "bought", "house", "d"))
    assert not _supersedes(links)


def test_gb2_attended_x_is_not_superseded_by_attended_y() -> None:
    """(c) one can attend many events."""
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1"), Triplet("Caroline", "attended", "X", "d"))
    links = _add(graph, _note("n2"), Triplet("Caroline", "attended", "Y", "d"))
    assert not _supersedes(links)


def test_gb2_state_adjective_via_closed_list() -> None:
    """(d) `is happy` -> `is sad` supersedes, via an EXPLICIT list entry."""
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1"), Triplet("Caroline", "is happy", "happy", "d"))
    links = _add(graph, _note("n2"), Triplet("Caroline", "is sad", "sad", "d"))
    assert _supersedes(links)


def test_gb2_unlisted_state_adjective_does_not_supersede() -> None:
    """GB2c: no wildcard. `is gorgeous` is a description, not a slot."""
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1"), Triplet("Melanie", "is gorgeous", "necklace", "d"))
    links = _add(graph, _note("n2"), Triplet("Melanie", "is gorgeous", "ring", "d"))
    assert not _supersedes(links)


@pytest.mark.parametrize(
    "predicate",
    ["has", "had", "owns", "needs", "reacted_to", "appreciates", "gave", "said"],
)
def test_gb2_additive_predicates_never_supersede(predicate: str) -> None:
    """GB2a: these are MULTI-valued. Audit pair 3 came from `has`.

    "...has a dog and a cat" and "...recently got another cat named Bailey" are
    simultaneously TRUE; marking the first obsolete deletes live evidence.
    """
    assert not TemporalHyperGraph._is_single_valued_predicate(predicate)


def test_gb2_audited_pair_3_has_does_not_supersede() -> None:
    """The real audited pair, end to end through `add_note`."""
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1", "Melanie has a dog and a cat"),
         Triplet("Melanie", "has", "a dog and a cat", "d"))
    links = _add(graph, _note("n2", "Melanie recently got another cat named Bailey"),
                 Triplet("Melanie", "has", "another cat named Bailey", "d"))
    assert not _supersedes(links), "additive facts must not be versioned"


def test_gb2_paraphrase_objects_do_not_supersede() -> None:
    """(f) audited pair 1: a restatement, where the OLDER note is the richer one."""
    assert not TemporalHyperGraph._objects_are_distinct("Boston", "boston")
    assert TemporalHyperGraph._objects_are_distinct("Boston", "Seattle")


def test_gb2_missing_object_does_not_supersede() -> None:
    """Never mark evidence obsolete on missing data."""
    assert not TemporalHyperGraph._objects_are_distinct("", "Seattle")
    assert not TemporalHyperGraph._objects_are_distinct("Boston", "")
    assert not TemporalHyperGraph._objects_are_distinct("-", "Seattle")

    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1"), Triplet("Caroline", "lives", "", "d"))
    links = _add(graph, _note("n2"), Triplet("Caroline", "lives", "Seattle", "d"))
    assert not _supersedes(links)


def test_gb2d_no_zero_length_validity_interval() -> None:
    """GB2d: same-session notes share one timestamp; no zero-length window.

    PRE-FIX the pair wrote `valid_to == valid_from`, rendering as "true until
    <the moment it became true>".
    """
    graph = TemporalHyperGraph(enable_supersession=True)
    same = "2022-03-01T00:00:00Z"
    _add(graph, _note("n1", iso=same), Triplet("Caroline", "lives in", "Boston", same))
    _add(graph, _note("n2", iso=same), Triplet("Caroline", "lives in", "Seattle", same))
    graph.pop_touched_peers()

    for nid in ("n1", "n2"):
        for link in graph.get_note(nid).L:
            if link.relation != "superseded_by":
                continue
            if link.valid_from and link.valid_to:
                assert link.valid_to > link.valid_from, (
                    f"zero/negative-length interval on {nid}: "
                    f"{link.valid_from} -> {link.valid_to}"
                )


def test_gb2d_positive_interval_still_recorded() -> None:
    """The fix must not suppress intervals that ARE meaningful."""
    graph = TemporalHyperGraph(enable_supersession=True)
    old = _note("n1", iso="2022-03-01T00:00:00Z")
    _add(graph, old, Triplet("Caroline", "lives in", "Boston", "d"))
    _add(graph, _note("n2", iso="2022-03-03T09:00:00Z"),
         Triplet("Caroline", "lives in", "Seattle", "d"))
    graph.pop_touched_peers()
    link = next(l for l in old.L if l.relation == "superseded_by")
    assert link.valid_to == "2022-03-03T09:00:00Z"
    assert link.valid_to > link.valid_from


def test_gb2e_supersession_can_be_disabled() -> None:
    """GB2e: the ablation flag suppresses the edge but keeps indices maintained."""
    graph = TemporalHyperGraph(enable_supersession=False)
    _add(graph, _note("n1"), Triplet("Caroline", "lives in", "Boston", "d"))
    links = _add(graph, _note("n2"), Triplet("Caroline", "lives in", "Seattle", "d"))
    assert not _supersedes(links), "supersession must be off"
    # Indices still track the latest holder, so re-enabling behaves identically.
    assert graph._subj_pred_index_slot[("caroline", "lives in")] == "n2"
    # No other relation is affected.
    assert not any(l.relation == "superseded_by" for l in graph.get_note("n1").L)


def test_gb2e_disabled_graph_still_builds_other_relations() -> None:
    """Disabling supersession must not disable entity/semantic/temporal edges."""
    graph = TemporalHyperGraph(enable_supersession=False, semantic_tau=0.5, max_entity_links=5)
    _add(graph, _note("n1", entities=["Caroline", "Hawaii"]))
    links = _add(graph, _note("n2", entities=["Caroline", "Hawaii"]))
    rels = {l.relation for l in links}
    assert "same-entity" in rels and "semantic" in rels


# ===========================================================================
# GB3 — entity index, O(peers) peers, invalidate()
# ===========================================================================

def test_gb3_inverse_entity_index_is_maintained() -> None:
    graph = TemporalHyperGraph()
    _add(graph, _note("n1", entities=["Caroline", "Hawaii"]))
    assert graph._entity_to_notes["caroline"] == {"n1"}
    assert graph._entity_to_notes["hawaii"] == {"n1"}

    graph.hydrate([_note("n2", entities=["Caroline"])])
    assert graph._entity_to_notes["caroline"] == {"n1", "n2"}


def test_gb3_entity_peers_does_not_scan_the_whole_bank() -> None:
    """Peer discovery must be proportional to matching peers, not to N.

    PRE-FIX `_entity_peers` iterated every note in `_fact_nodes` and tested set
    intersection inline, so a note sharing nothing with the bank still cost one
    cosine per other note (300 here). The inverse index must avoid the scan
    entirely, which is measured by counting candidate walks, not by timing.
    """
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=5, max_semantic_links=0)
    for i in range(300):
        _add(graph, _note(f"other{i}", entities=[f"e{i}"]))

    calls = {"n": 0}
    original = graph._cosine

    def counting_cosine(a, b):
        calls["n"] += 1
        return original(a, b)

    graph._cosine = counting_cosine  # type: ignore[method-assign]
    try:
        probe = _note("probe", entities=["nothing-shared"])
        graph.add_note(probe)
        peers = graph._entity_peers(probe, graph._entity_index["probe"])
    finally:
        graph._cosine = original  # type: ignore[method-assign]

    assert peers == []
    assert calls["n"] == 0, (
        f"a note sharing no entity triggered {calls['n']} cosine comparisons; "
        "the bank is still being scanned linearly"
    )


def test_gb3_invalidate_removes_the_note_from_every_index() -> None:
    graph = TemporalHyperGraph(enable_supersession=True)
    _add(graph, _note("n1", entities=["Caroline"]), Triplet("Caroline", "lives in", "Boston", "d"))
    _add(graph, _note("n2", entities=["Caroline"]), Triplet("Caroline", "lives in", "Boston", "d"))
    assert graph.invalidate("n1") is True

    assert graph.get_note("n1") is None
    assert "n1" not in graph._entity_index
    assert "n1" not in graph._entity_to_notes.get("caroline", set())
    assert graph.invalidate("n1") is False  # idempotent
    # n2 still holds the slot, so invalidating n1 must not have cleared it.
    assert graph._subj_pred_index_slot[("caroline", "lives in")] == "n2"


def test_gb3_invalidated_note_is_no_longer_a_peer() -> None:
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=5, max_semantic_links=0)
    _add(graph, _note("dead", entities=["Shared"]))
    _add(graph, _note("alive", entities=["Shared"]))
    graph.invalidate("dead")
    peers = graph._entity_peers(graph.get_note("alive"), graph._entity_index["alive"])
    assert all(pid != "dead" for pid, _ in peers)


def test_gb3_add_note_scales_sublinearly() -> None:
    """GB3 perf: adding to a 2000-note graph must not cost N-proportional time.

    Measures the per-note cost at N=250 vs N=2000. A linear `_entity_peers` scan
    made this grow ~8x with N; the inverse index makes it roughly flat.
    """
    def time_one_add(n_prior: int) -> float:
        g = TemporalHyperGraph(semantic_tau=0.7, max_entity_links=5, max_semantic_links=0)
        rng = np.random.default_rng(0)
        for i in range(n_prior):
            _add(g, _note(f"p{i}", entities=["Shared"], vec=rng.random(8)))
        t0 = time.perf_counter()
        g.add_note(_note("last", entities=["Shared"], vec=rng.random(8)))
        return time.perf_counter() - t0

    small = time_one_add(250)
    large = time_one_add(2000)
    ratio = large / max(small, 1e-9)
    # 8x the notes. Linear scanning gives ~8x; give generous headroom for a
    # noisy CI machine while still failing a genuine O(N) scan.
    assert ratio < 5.0, (
        f"add_note cost grew {ratio:.1f}x for 8x the notes "
        f"({small*1000:.2f} ms at N=250 -> {large*1000:.2f} ms at N=2000); "
        "the entity peer lookup is still O(N)"
    )


# ===========================================================================
# GB4 — bidirectional personalization
# ===========================================================================

def test_gb4_reverse_pagerank_surfaces_a_premise() -> None:
    """A note that points INTO the seed set must be reachable in reverse.

    PRE-FIX `personalized_pagerank` walked outward only, so the premise's score
    was exactly 0.0 — it could appear in the list, but never rank. Direction
    comes from the persisted `Note.L` records (premise asserts -> seed), because
    the nx graph itself is undirected and has no `reverse()`.
    """
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    premise = _note("premise")
    seed = _note("seed")
    _add(graph, premise)
    _add(graph, seed)
    # The premise asserts a fact the seed depends on.
    premise.L.append(LinkRecord(target_id="seed", relation="causal"))

    forward = dict(graph.personalized_pagerank(["seed"], top_k=10, directions="out"))
    reverse = dict(graph.personalized_pagerank(["seed"], top_k=10, directions="in"))

    assert forward.get("premise", 0.0) == 0.0, (
        "outward-only PPR must not reach a note that points INTO the seed"
    )
    assert reverse.get("premise", 0.0) > 0.0, (
        "the reverse walk must surface the premise"
    )


def test_gb4_directions_parameter_and_alias_agree() -> None:
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    a, b = _note("a"), _note("b")
    _add(graph, a)
    _add(graph, b)
    a.L.append(LinkRecord(target_id="b", relation="causal"))
    assert graph.personalized_pagerank(["b"], top_k=5, directions="in") == \
        graph.personalized_pagerank(["b"], top_k=5, reverse=True)
    assert graph.personalized_pagerank(["b"], top_k=5, directions="both")
    # "out" is the historical default and must be unchanged.
    assert graph.personalized_pagerank(["b"], top_k=5) == \
        graph.personalized_pagerank(["b"], top_k=5, directions="out")


def test_gb4_bidirectional_pagerank_merges_both_walks() -> None:
    """A note reachable in EITHER direction must be retained by the merge."""
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=0, max_semantic_links=0)
    incoming, seed, outgoing = _note("in"), _note("seed"), _note("out")
    for n in (incoming, seed, outgoing):
        _add(graph, n)
    incoming.L.append(LinkRecord(target_id="seed", relation="causal"))
    seed.L.append(LinkRecord(target_id="out", relation="causal"))

    merged = dict(graph.bidirectional_pagerank(["seed"], top_k=10))
    assert "seed" in merged
    assert merged.get("in", 0.0) > 0.0, "the incoming prerequisite must survive the merge"
    assert merged.get("out", 0.0) > 0.0, "the outgoing consequence must survive the merge"


def test_gb4_empty_seed_fallback_is_not_a_uniform_score() -> None:
    """A uniform score is worse than none: Phase D min-max normalises it flat."""
    graph = TemporalHyperGraph()
    graph._fact_nodes["ghost"] = _note("ghost")  # in _fact_nodes, not in nx graph
    assert graph.personalized_pagerank(["ghost"], top_k=5) == []


def test_gb4_no_seeds_at_all_returns_empty() -> None:
    graph = TemporalHyperGraph()
    _add(graph, _note("n1"))
    assert graph.personalized_pagerank([], top_k=5) == []
    assert graph.personalized_pagerank(["does-not-exist"], top_k=5) == []


def test_gb4_forward_scores_unchanged_by_default() -> None:
    """`ppr_reverse` defaults off, so the forward walk is untouched."""
    from asem.enhanced_retriever import EnhancedHybridRetriever

    r = EnhancedHybridRetriever(
        backend=None, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
    )
    assert r.ppr_reverse is False
    assert r.ppr_reverse_weight == 0.5
    assert r.ppr_scores_reverse == {}
    # And the sensor that reports the reverse channel stays silent.
    assert "phase_d_ppr_reverse_active" not in r.stats


# ===========================================================================
# GB5 — hub-aware entity weighting
# ===========================================================================

def test_gb5_hub_entity_alone_yields_no_discriminative_peer_advantage() -> None:
    """THE FAILING TEST for GB5: a bare hub must not create a deciding edge.

    A bank where one generic token sits on 2/3 of the notes and a rare proper
    noun sits on two: the rare shared entity must produce a peer that OUTRANKS
    the hub-only peers.
    """
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=1, max_semantic_links=0)
    # 12 notes carrying only the hub token, all with identical embeddings so
    # cosine cannot break the tie.
    for i in range(12):
        _add(graph, _note(f"hub{i}", entities=["user"], vec=[1.0, 0.0]))
    # Two notes sharing a rare proper noun.
    _add(graph, _note("rareA", entities=["user", "oliver"], vec=[0.9, 0.1]))
    rareB = _note("rareB", entities=["user", "oliver"], vec=[0.8, 0.2])
    _add(graph, rareB)

    peers = graph._entity_peers(graph.get_note("rareA"), graph._entity_index["rareA"])
    assert peers, "the rare note must still have peers"
    top = peers[0][0]
    assert top == "rareB", (
        f"the rare-entity peer must outrank hub-only peers, got {top!r} "
        f"(peers={[(p[:8], round(s, 3)) for p, s in peers]})"
    )
    # And the rare peer's score must be materially higher, not marginally.
    hub_only = [s for p, s in peers[1:]]
    assert peers[0][1] > (max(hub_only) if hub_only else 0.0)


def test_gb5_rare_entity_still_connects() -> None:
    """A fix that severs the rare-entity channel is a regression."""
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=5, max_semantic_links=0)
    for i in range(20):
        _add(graph, _note(f"hub{i}", entities=["user", "assistant"]))
    _add(graph, _note("a", entities=["user", "grand canyon"]))
    _add(graph, _note("b", entities=["user", "grand canyon"]))
    peers = graph._entity_peers(graph.get_note("a"), graph._entity_index["a"])
    assert any(p == "b" for p, _ in peers)


def test_gb5_idf_favours_the_rare_entity() -> None:
    graph = TemporalHyperGraph()
    for i in range(100):
        _add(graph, _note(f"n{i}", entities=["user"]))
    _add(graph, _note("a", entities=["oliver"]))
    _add(graph, _note("b", entities=["oliver"]))
    w_hub = graph._entity_weight("user")
    w_rare = graph._entity_weight("oliver")
    assert w_rare > w_hub * 3, (
        f"idf must separate a rare entity from a hub: {w_rare:.3f} vs {w_hub:.3f}"
    )


def test_gb5_global_hub_keeps_a_small_positive_weight() -> None:
    """The floor is load-bearing: a key on EVERY note must not score 0.

    With plain smoothed idf a key at df == N scores exactly log(1) == 0, which
    removed its peers entirely and severed the entity channel (this assertion
    failed during development and turned 4 pre-existing tests red).

    The hub token here is a NAME, not a role word — `user`/`assistant` are
    removed earlier by `_ROLE_TOKENS`, so they never reach the weighting.
    """
    graph = TemporalHyperGraph()
    for i in range(4):
        _add(graph, _note(f"n{i}", entities=["Melanie"]))
    assert graph._entity_weight("melanie") > 0.0
    peers = graph._entity_peers(graph.get_note("n0"), graph._entity_index["n0"])
    assert peers, "sharing a global key must still yield weak peers"


def test_gb5_role_tokens_are_not_entities() -> None:
    """Role words name no entity, so they never justify a same-entity edge."""
    keys = TemporalHyperGraph._entity_keys(
        _note("n1", entities=["user", "assistant", "Caroline"])
    )
    assert keys == {"caroline"}


# ===========================================================================
# Frozen-fixture acceptance (read-only; skipped if the bank is absent)
# ===========================================================================

@pytest.fixture(scope="module")
def frozen_notes():
    if not FROZEN_BANK.exists():
        pytest.skip("frozen LoCoMo fixture bank not present")
    tmp = Path(tempfile.mkdtemp()) / "bank.sqlite"
    shutil.copy2(FROZEN_BANK, tmp)
    bank = MemoryBank(str(tmp))
    notes = bank.list_notes()
    bank.close()
    return notes


def test_frozen_fixture_discriminative_peers_not_reduced(frozen_notes) -> None:
    """GB5 on the real bank: rare-entity connectivity must improve, not regress.

    Baseline (Lead's measurement): 1792 unique same-entity edges, 1779 hub-only
    (99.27%), 13 discriminative (0.73%).
    """
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=5,
                               max_semantic_links=0, enable_supersession=False)
    for n in frozen_notes:
        graph.add_note(n)

    hubs = {"caroline", "melanie"}
    df = Counter()
    for n in frozen_notes:
        for k in graph._entity_index[n.id]:
            df[k] += 1

    pairs = {}
    for n in frozen_notes:
        keys = graph._entity_index[n.id]
        for pid, _ in graph._entity_peers(n, keys):
            key = tuple(sorted((n.id, pid)))
            pairs.setdefault(key, keys & (graph._entity_index.get(pid) or set()))

    discriminative = sum(
        1 for shared in pairs.values() if shared and not (shared <= hubs)
    )
    assert discriminative > 13, (
        f"discriminative same-entity edges = {discriminative}; "
        "the fix must not reduce rare-entity connectivity below the 13 baseline"
    )


def test_frozen_fixture_rare_peers_rank_above_hub_only(frozen_notes) -> None:
    """On the real bank, a rare-sharing peer must outrank hub-only peers."""
    graph = TemporalHyperGraph(semantic_tau=2.0, max_entity_links=5,
                               max_semantic_links=0, enable_supersession=False)
    for n in frozen_notes:
        graph.add_note(n)

    hubs = {"caroline", "melanie"}
    df = Counter()
    for n in frozen_notes:
        for k in graph._entity_index[n.id]:
            df[k] += 1
    rare_keys = [k for k, c in df.items() if c <= 3]
    assert rare_keys, "fixture must have rare entities"

    checked = wins = 0
    for rare in rare_keys:
        for nid in sorted(graph._entity_to_notes.get(rare, ())):
            keys = graph._entity_index[nid]
            peers = graph._entity_peers(graph._fact_nodes[nid], keys)
            rare_pos = [
                i for i, (pid, _) in enumerate(peers)
                if rare in (keys & (graph._entity_index.get(pid) or set()))
            ]
            hub_pos = [
                i for i, (pid, _) in enumerate(peers)
                if (keys & (graph._entity_index.get(pid) or set())) <= hubs
            ]
            if not rare_pos:
                continue
            checked += 1
            if not hub_pos or min(rare_pos) < min(hub_pos):
                wins += 1
            break
    assert checked > 0, "no rare-carrying note found in the fixture"
    assert wins == checked, (
        f"rare peer outranked hub-only on {wins}/{checked} notes; "
        "the weighting is not reordering the peers"
    )


def test_frozen_fixture_preserves_legacy_rows_through_migration(frozen_notes) -> None:
    """The fixture predates `triplets`, so it doubles as the GB1 migration subject."""
    assert len(frozen_notes) == 365
    assert all(n.triplets == [] for n in frozen_notes)
