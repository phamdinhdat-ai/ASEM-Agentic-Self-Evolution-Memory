"""CARE-Bridge contracts for cheap retrieval, speaker binding and retry cost."""

import json

from asem.care import CareBridge, SourceTurn
from scripts.run_care_bridge_locomo import run


TURNS = [SourceTurn("S1", "session_1", "1 January 2023", "Sam", "I resumed hiking.", 0),
         SourceTurn("S2", "session_1", "1 January 2023", "Evan", "I am reading The Great Gatsby.", 1)]


class SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_valid_answer_uses_one_call():
    backend = SequenceBackend([json.dumps({"answer": "hiking", "source_ids": ["S1"],
                                         "search_queries": []})])
    result = CareBridge(TURNS).answer("What activity did Sam resume?", backend)
    assert result["answer"] == "hiking"
    assert result["source_ids"] == ["S1"]
    assert result["escalated"] is False
    assert len(backend.prompts) == 1


def test_wrong_speaker_fact_is_rejected_then_abstained():
    backend = SequenceBackend([
        '{"answer":"The Great Gatsby","source_ids":["S2"],"search_queries":[]}',
        '{"answer":"No information available","source_ids":[],"search_queries":[]}'])
    result = CareBridge(TURNS).answer("What novel is Sam reading?", backend)
    assert result["answer"] == "No information available"
    assert result["escalated"] is True
    assert result["first_pass"]["invalid_reason"] == "speaker_mismatch"
    assert len(backend.prompts) == 2


def test_missing_evidence_search_has_two_call_cap():
    backend = SequenceBackend([
        '{"answer":"No information available","source_ids":[],"search_queries":["Sam hiking"]}',
        '{"answer":"hiking","source_ids":["S1"],"search_queries":[]}'])
    result = CareBridge(TURNS).answer("What activity did Sam resume?", backend)
    assert result["answer"] == "hiking"
    assert result["escalated"] is True
    assert len(backend.prompts) == 2


def test_proof_bm25_never_escalates():
    backend = SequenceBackend([
        '{"answer":"No information available","source_ids":[],"search_queries":["Sam hiking"]}'])
    result = CareBridge(TURNS).answer("What activity did Sam resume?", backend,
                                      mode="proof_bm25")
    assert result["escalated"] is False
    assert len(backend.prompts) == 1


def test_bridge_runner_scores_three_arms_and_resumes(tmp_path):
    dataset = [{"sample_id": "small", "conversation": {
        "session_1_date_time": "1 January 2023",
        "session_1": [{"dia_id": "S1", "speaker": "Sam", "text": "I resumed hiking."}]},
        "qa": [{"question": "What activity did Sam resume?", "answer": "hiking",
                "evidence": ["S1"], "category": 4}]}]
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(dataset), encoding="utf-8")

    class Backend:
        def __init__(self):
            self.calls = []
        def generate(self, prompt, **kwargs):
            self.calls.append(prompt)
            return '{"answer":"hiking","source_ids":["S1"],"search_queries":[]}'

    backend = Backend()
    report = run(data_path, tmp_path / "results", backend,
                 per_category=1, start_per_category=0, categories=(4,))
    assert all(report["aggregate"][arm]["answerable_mean_f1"] == 1.0
               for arm in ("proof_bm25", "bridge", "care_v1"))
    before = len(backend.calls)
    run(data_path, tmp_path / "results", backend,
        per_category=1, start_per_category=0, categories=(4,))
    assert len(backend.calls) == before


def test_holiday_and_change_polarity_require_direct_witness():
    turns = [SourceTurn("H1", "s1", "", "Tim",
                        "During Thanksgiving we watch Home Alone.", 0),
             SourceTurn("H2", "s1", "", "Evan",
                        "I started lifting weights one year ago.", 1)]
    holiday = SequenceBackend([
        '{"answer":"Home Alone","source_ids":["H1"],"search_queries":[]}'])
    result = CareBridge(turns).answer("What movie does Tim watch at Halloween?",
                                      holiday, mode="proof_bm25")
    assert result["answer"] == "No information available"
    assert result["invalid_reason"] == "event_anchor_mismatch"
    polarity = SequenceBackend([
        '{"answer":"lifting weights","source_ids":["H2"],"search_queries":[]}'])
    result = CareBridge(turns).answer("What activity did Evan quit one year ago?",
                                      polarity, mode="proof_bm25")
    assert result["answer"] == "No information available"
    assert result["invalid_reason"] == "change_polarity_mismatch"


def test_vocative_does_not_override_first_person_speaker():
    turns = [SourceTurn("V1", "s1", "", "Melanie",
                        "Thanks, Caroline. Recently I had a setback.", 0),
             SourceTurn("V2", "s1", "", "Caroline", "I hope you recover.", 1)]
    backend = SequenceBackend([
        '{"answer":"a setback","source_ids":["V1"],"search_queries":[]}'])
    result = CareBridge(turns).answer("What setback did Caroline face recently?",
                                      backend, mode="proof_bm25")
    assert result["answer"] == "No information available"
    assert result["invalid_reason"] == "speaker_mismatch"


