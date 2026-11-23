"""Semantic contracts for dated relations and adversarial source checks."""

from datetime import date, datetime
import json

from asem.care import SourceTurn
from asem.care.temporal_bridge import CareTemporalBridge
from asem.care.temporal_relations import (
    TemporalClaim, TemporalProofGuard, TemporalRelationGraph, TimeSpan,
    event_time, extract_high_confidence,
)
from scripts.run_care_locomo import select_questions
from scripts.run_care_temporal_locomo import select_paired_questions


def turn(source_id, said_at, speaker, text, ordinal=0):
    return SourceTurn(source_id, "session_1", said_at, speaker, text, ordinal)


class SequenceBackend:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        return json.dumps(next(self.answers))


def test_explicit_update_respects_valid_time_and_preserves_history():
    paris = turn("S1", "1 January 2024", "Alex", "I live in Paris.")
    hanoi = turn("S2", "1 June 2024", "Alex", "I moved to Hanoi; I live in Hanoi.", 1)
    graph = TemporalRelationGraph([paris, hanoi])
    graph.add(TemporalClaim("C1", "S1", "Alex", "residence", "Paris", "I live in Paris.",
                            said_at=date(2024, 1, 1), valid=TimeSpan(date(2024, 1, 1)),
                            functional=True))
    graph.add(TemporalClaim("C2", "S2", "Alex", "residence", "Hanoi", "I live in Hanoi.",
                            said_at=date(2024, 6, 1), valid=TimeSpan(date(2024, 6, 1)),
                            explicit_update=True, functional=True))
    assert [c.value for c in graph.as_of("Alex", "residence", date(2024, 5, 1)).claims] == ["Paris"]
    assert [c.value for c in graph.as_of("Alex", "residence", date(2024, 7, 1)).claims] == ["Hanoi"]
    assert any(e.relation == "supersedes" and e.source == "C2" and e.target == "C1"
               for e in graph.edges)


def test_later_statement_alone_does_not_overwrite_conflicting_claim():
    a = turn("S1", "1 January 2024", "Alex", "I live in Paris.")
    b = turn("S2", "1 June 2024", "Alex", "I live in Hanoi.", 1)
    graph = TemporalRelationGraph([a, b])
    graph.add(TemporalClaim("C1", "S1", "Alex", "residence", "Paris", a.text,
                            valid=TimeSpan(date(2024, 1, 1)), functional=True))
    graph.add(TemporalClaim("C2", "S2", "Alex", "residence", "Hanoi", b.text,
                            valid=TimeSpan(date(2024, 6, 1)), functional=True))
    assert graph.as_of("Alex", "residence", date(2024, 7, 1)).status == "conflict"
    assert not any(e.relation == "supersedes" for e in graph.edges)
    assert graph.as_of("Alex", "residence", date(2023, 1, 1)).status == "unknown"


def test_update_edges_do_not_depend_on_ingest_order():
    old = turn("S1", "1 January 2024", "Alex", "I live in Paris.")
    new = turn("S2", "1 June 2024", "Alex", "I moved to Hanoi.")
    graph = TemporalRelationGraph([old, new])
    graph.add(TemporalClaim("C2", "S2", "Alex", "residence", "Hanoi", new.text,
                            said_at=date(2024, 6, 1), valid=TimeSpan(date(2024, 6, 1)),
                            explicit_update=True, functional=True))
    graph.add(TemporalClaim("C1", "S1", "Alex", "residence", "Paris", old.text,
                            said_at=date(2024, 1, 1), valid=TimeSpan(date(2024, 1, 1)),
                            functional=True))
    assert [c.value for c in graph.as_of("Alex", "residence", date(2024, 7, 1)).claims] == ["Hanoi"]


def test_known_at_does_not_use_a_later_recorded_update():
    old = turn("S1", "1 January 2024", "Alex", "I live in Paris.")
    new = turn("S2", "1 June 2024", "Alex", "I moved to Hanoi.")
    graph = TemporalRelationGraph([old, new])
    graph.add(TemporalClaim("C1", "S1", "Alex", "residence", "Paris", old.text,
                            said_at=date(2024, 1, 1), recorded_at=datetime(2024, 1, 2),
                            valid=TimeSpan(date(2024, 1, 1)), functional=True))
    graph.add(TemporalClaim("C2", "S2", "Alex", "residence", "Hanoi", new.text,
                            said_at=date(2024, 6, 1), recorded_at=datetime(2024, 6, 2),
                            valid=TimeSpan(date(2024, 6, 1)), explicit_update=True,
                            functional=True))
    may_snapshot = graph.as_of("Alex", "residence", date(2024, 7, 1),
                               known_at=datetime(2024, 5, 31))
    assert [claim.value for claim in may_snapshot.claims] == ["Paris"]
    july_snapshot = graph.as_of("Alex", "residence", date(2024, 7, 1),
                                known_at=datetime(2024, 7, 1))
    assert [claim.value for claim in july_snapshot.claims] == ["Hanoi"]


def test_relative_event_uses_source_utterance_date():
    source = turn("S1", "21 August, 2023", "Evan",
                  "I started lifting weights one year ago.")
    claims = extract_high_confidence(source)
    assert len(claims) == 1
    assert claims[0].relation == "start_activity"
    assert claims[0].occurred == TimeSpan(date(2022, 8, 21), date(2022, 8, 22))
    assert event_time("one year ago", None) is None


def test_guard_checks_relation_degree_and_event_scope_without_fabricating_facts():
    favorite = turn("S1", "", "John", "My favorite character is Aragorn.")
    holiday = turn("S2", "", "Tim", "During Thanksgiving we watch Home Alone.")
    guard = TemporalProofGuard()
    assert guard.check("Who is John's least favorite character?", "Aragorn", [favorite]).status == "incompatible"
    assert guard.check("What does John like?", "Aragorn", [favorite]).status == "unknown"
    assert guard.check("What movie does Tim watch at Halloween?", "Home Alone", [holiday]).status == "incompatible"
    assert guard.check("What movie does Tim watch at Halloween?", "another movie", [holiday]).status == "unknown"
    halloween = turn("S3", "", "Tim", "During Halloween we watch Home Alone.")
    assert guard.check("What movie does Tim watch at Halloween?", "Home Alone",
                       [holiday, halloween]).status == "supported"
    mixed = turn("S4", "", "Tim",
                 "During Halloween we watch The Addams Family. During Thanksgiving we watch Home Alone.")
    assert guard.check("What movie does Tim watch at Halloween?", "Home Alone", [mixed]).status == "incompatible"


def test_opt_in_bridge_retries_a_rejected_answer_and_caps_reader_calls():
    turns = [turn("S1", "", "John", "My favorite character is Aragorn."),
             turn("S2", "", "John", "My least favorite character is Sauron.", 1)]
    backend = SequenceBackend([
        {"answer": "Aragorn", "source_ids": ["S1"], "search_queries": []},
        {"answer": "Sauron", "source_ids": ["S2"], "search_queries": []},
    ])
    result = CareTemporalBridge(turns).answer("Who is John's least favorite character?", backend)
    assert result["answer"] == "Sauron"
    assert result["escalated"] is True
    assert result["first_pass"]["invalid_reason"] == "temporal_preference_degree_mismatch"
    assert backend.calls == 2


def test_paired_selection_is_deterministic_and_excludes_prior_ids():
    qa = [{"category": category, "question": f"What happened during year {i}?",
           "answer": "held out", "evidence": []}
          for category in range(1, 6) for i in range(10)]
    dataset = [{"sample_id": "conv-test", "qa": qa}]
    selected = select_paired_questions(dataset, per_category=2,
        start_per_category=2, categories=(1, 2, 3, 4, 5), challenge_count=3)
    prior = select_questions(dataset, per_category=2, start_per_category=0,
                             categories=(1, 2, 3, 4, 5))
    assert len(selected) == len({item["id"] for item in selected}) == 13
    assert {item["id"] for item in selected}.isdisjoint(item["id"] for item in prior)
    assert [item["id"] for item in selected] == [item["id"] for item in
        select_paired_questions(dataset, per_category=2, start_per_category=2,
                                categories=(1, 2, 3, 4, 5), challenge_count=3)]
