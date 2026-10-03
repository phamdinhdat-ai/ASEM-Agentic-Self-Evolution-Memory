"""Tests for graph temporal structure: versioning keys, intervals, entity hygiene.

These pin the three graph-level changes that support temporal reasoning:

* **object-aware versioning** — a changed object on a single-valued predicate
  invalidates the previous fact, while a different object on a multi-valued verb
  does not;
* **bi-temporal validity** — superseded pairs carry `valid_from`/`valid_to`, and
  those intervals survive serialisation;
* **entity hygiene** — only named entities drive the `same-entity` channel, not
  topic keywords.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from asem.hyper_graph import TemporalHyperGraph, Triplet
from asem.note import LinkRecord, Note


def _note(nid: str, text: str, iso: str, entities=None, keywords=None) -> Note:
    return Note(
        id=nid,
        c=text,
        t=datetime.fromisoformat(iso.replace("Z", "+00:00")),
        K=list(keywords or []),
        G=[],
        X=text,
        e=np.zeros(3),
        L=[],
        z=np.zeros(3),
        q=0.5,
        session_date=iso,
        timestamp_iso=iso,
        entities=list(entities or []),
    )


def _add(graph: TemporalHyperGraph, note: Note, triplet: Triplet | None = None):
    """Add a note and apply the returned links to ``note.L`` (caller owns that)."""
    links = graph.add_note(note, triplet)
    note.L.extend(links)
    return links


# ---------------------------------------------------------------------------
# Object-aware versioning
# ---------------------------------------------------------------------------

def test_replacement_on_single_valued_predicate_supersedes() -> None:
    """`lives in X` -> `lives in Y`: the subject can only live in one place."""
    graph = TemporalHyperGraph()
    old = _note("n1", "Caroline lives in Boston", "2022-03-01T00:00:00Z")
    new = _note("n2", "Caroline lives in Seattle", "2022-03-03T09:00:00Z")
    _add(graph, old, Triplet("Caroline", "lives in", "Boston", "2022-03-01"))
    links = _add(graph, new, Triplet("Caroline", "lives in", "Seattle", "2022-03-03"))

    assert any(link.relation == "superseded_by" for link in links), (
        "a changed single-valued attribute must invalidate the previous value"
    )


def test_multi_valued_predicate_does_not_supersede() -> None:
    """"bought a car" and "bought a house" are two facts, not two versions."""
    graph = TemporalHyperGraph()
    _add(graph, _note("n1", "Caroline bought a car", "2022-03-01T00:00:00Z"),
         Triplet("Caroline", "bought", "car", "2022-03-01"))
    links = _add(graph, _note("n2", "Caroline bought a house", "2022-03-02T00:00:00Z"),
                 Triplet("Caroline", "bought", "house", "2022-03-02"))

    assert not any(link.relation == "superseded_by" for link in links), (
        "multi-valued verbs must not mark unrelated facts obsolete"
    )


def test_restated_identical_triplet_supersedes() -> None:
    """The same (subject, predicate, object) seen again is a re-assertion."""
    graph = TemporalHyperGraph()
    _add(graph, _note("n1", "Caroline lives in Boston", "2022-03-01T00:00:00Z"),
         Triplet("Caroline", "lives in", "Boston", "2022-03-01"))
    links = _add(graph, _note("n2", "Caroline lives in Boston", "2022-03-02T00:00:00Z"),
                 Triplet("Caroline", "lives in", "Boston", "2022-03-02"))

    assert any(link.relation == "superseded_by" for link in links)


def test_note_never_supersedes_itself() -> None:
    graph = TemporalHyperGraph()
    note = _note("n1", "Caroline lives in Boston", "2022-03-01T00:00:00Z")
    links = _add(graph, note, Triplet("Caroline", "lives in", "Boston", "2022-03-01"))
    assert not any(link.relation == "superseded_by" for link in links)


def test_missing_triplet_creates_no_supersession() -> None:
    graph = TemporalHyperGraph()
    _add(graph, _note("n1", "a fact", "2022-03-01T00:00:00Z"))
    links = _add(graph, _note("n2", "another fact", "2022-03-02T00:00:00Z"))
    assert not any(link.relation == "superseded_by" for link in links)


# ---------------------------------------------------------------------------
# Bi-temporal validity
# ---------------------------------------------------------------------------

def test_supersession_records_a_validity_interval() -> None:
    """The replaced fact ends when the replacement starts; the new one is open."""
    graph = TemporalHyperGraph()
    old = _note("n1", "Caroline lives in Boston", "2022-03-01T00:00:00Z")
    new = _note("n2", "Caroline lives in Seattle", "2022-03-03T09:00:00Z")
    _add(graph, old, Triplet("Caroline", "lives in", "Boston", "2022-03-01"))
    _add(graph, new, Triplet("Caroline", "lives in", "Seattle", "2022-03-03"))
    graph.pop_touched_peers()  # the mirror edge on the OLD note is applied here

    old_link = next(l for l in old.L if l.relation == "superseded_by")
    assert old_link.valid_to == "2022-03-03T09:00:00Z", "old fact must END when replaced"
    assert old_link.valid_from == "2022-03-01T00:00:00Z"

    new_link = next(l for l in new.L if l.relation == "superseded_by")
    assert new_link.valid_from == "2022-03-03T09:00:00Z"
    assert new_link.valid_to is None, "the current fact's interval is still open"


def test_legacy_link_serialisation_is_unchanged() -> None:
    """A link with no interval must not gain null fields (frozen banks rely on it)."""
    legacy = LinkRecord(target_id="a", relation="temporal")
    assert legacy.to_dict() == {"target_id": "a", "relation": "temporal"}
    assert LinkRecord.from_dict(legacy.to_dict()) == legacy
    assert not legacy.has_interval


def test_interval_link_round_trips() -> None:
    link = LinkRecord(
        target_id="b", relation="superseded_by",
        valid_from="2022-03-01T00:00:00Z", valid_to="2022-03-03T00:00:00Z",
    )
    assert link.has_interval
    assert LinkRecord.from_dict(link.to_dict()) == link


def test_link_from_bare_string_and_dict_forms() -> None:
    """Both legacy payload shapes must still load (no interval -> None)."""
    from_str = LinkRecord.from_dict("note-7")
    assert from_str.target_id == "note-7" and from_str.valid_from is None
    from_dict = LinkRecord.from_dict({"target_id": "x", "relation": "causal"})
    assert from_dict.relation == "causal" and from_dict.valid_to is None


# ---------------------------------------------------------------------------
# Entity hygiene
# ---------------------------------------------------------------------------

def test_shared_keywords_do_not_create_entity_links() -> None:
    """Topic keywords are the semantic/BM25 channel's job, not the entity graph's."""
    graph = TemporalHyperGraph()
    a = _note("n1", "note A", "2022-03-01T00:00:00Z",
              entities=["Caroline"], keywords=["shopping", "travel", "tired"])
    b = _note("n2", "note B", "2022-03-02T00:00:00Z",
              entities=["Melanie"], keywords=["shopping", "travel", "tired"])
    _add(graph, a)
    links = _add(graph, b)
    assert not any(link.relation == "same-entity" for link in links)


def test_shared_real_entity_still_links() -> None:
    graph = TemporalHyperGraph()
    _add(graph, _note("n1", "note A", "2022-03-01T00:00:00Z", entities=["Caroline"]))
    links = _add(graph, _note("n2", "note B", "2022-03-02T00:00:00Z", entities=["Caroline"]))
    assert any(link.relation == "same-entity" for link in links)


# ---------------------------------------------------------------------------
# Single-valued predicate allow-list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("predicate", ["lives in", "works at", "is married to", "is", "was happy"])
def test_single_valued_predicates_recognised(predicate: str) -> None:
    assert TemporalHyperGraph._is_single_valued_predicate(predicate)


@pytest.mark.parametrize("predicate", ["bought", "attended", "visited", "studied", ""])
def test_multi_valued_predicates_rejected(predicate: str) -> None:
    assert not TemporalHyperGraph._is_single_valued_predicate(predicate)


def test_clear_resets_all_indices() -> None:
    """`clear()` must drop the new slot index too, or a reset bank keeps versioning."""
    graph = TemporalHyperGraph()
    _add(graph, _note("n1", "a", "2022-03-01T00:00:00Z"),
         Triplet("Caroline", "lives in", "Boston", "d"))
    graph.clear()
    links = _add(graph, _note("n2", "b", "2022-03-02T00:00:00Z"),
                 Triplet("Caroline", "lives in", "Seattle", "d"))
    assert not any(link.relation == "superseded_by" for link in links)
