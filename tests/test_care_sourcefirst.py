"""Checks the CARE pilot's evidence and benchmark contracts."""

import json

from asem.care import CareSourceFirst, SourceTurn
from scripts.run_care_locomo import run, score_locomo, select_questions, turns_for


class FakeBackend:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if "selected_ids" in prompt:
            return json.dumps({"selected_ids": ["D1:3", "made-up"], "facts": [
                {"source_id": "D1:3", "quote": "moved to Hanoi", "subject": "Alice",
                 "relation": "lives in", "value": "Hanoi", "status": "current"},
                {"source_id": "D1:3", "quote": "moved to Paris", "subject": "Alice",
                 "relation": "lives in", "value": "Paris", "status": "current"}],
                "unresolved_conflict": False})
        return '{"answer":"Hanoi","source_ids":["D1:3","made-up"]}'


def test_care_accepts_only_original_quotes_and_source_ids():
    turns = [SourceTurn("D1:1", "s1", "2024-01-01", "Alice", "I live in Paris.", 0),
             SourceTurn("D1:2", "s1", "2024-01-01", "Bob", "That is nice.", 1),
             SourceTurn("D1:3", "s1", "2024-02-01", "Alice", "I moved to Hanoi.", 2)]
    backend = FakeBackend()
    answer = CareSourceFirst(turns).answer("Where does Alice live now?", backend)
    assert answer["answer"] == "Hanoi"
    assert answer["source_ids"] == ["D1:3"]
    assert answer["compiled"] is True
    assert len(answer["verified_facts"]) == 1
    assert answer["verified_facts"][0]["quote"] == "moved to Hanoi"
    assert "made-up" not in backend.prompts[-1]
    assert "moved to Paris" not in backend.prompts[-1]


def test_care_compiler_is_conditional_and_bm25_uses_same_reader():
    turns = [SourceTurn("D1:1", "s1", "2024-01-01", "Alice", "My cat is Luna.", 0),
             SourceTurn("D1:2", "s1", "2024-01-01", "Bob", "Thanks Alice.", 1)]
    memory = CareSourceFirst(turns)
    first, second = FakeBackend(), FakeBackend()
    baseline = memory.answer("What is Alice's cat's name?", first, compile_mode="never")
    care = memory.answer("What is Alice's cat's name?", second)
    assert not care["compiled"]
    assert len(first.prompts) == len(second.prompts) == 1
    assert first.prompts[0] == second.prompts[0]
    assert baseline["bundle_ids"] == care["bundle_ids"]


def test_locomo_loader_keeps_id_speaker_date_and_selection_is_stable():
    dataset = [{"sample_id": "sample-1", "conversation": {
        "speaker_a": "Alice", "speaker_b": "Bob", "session_1_date_time": "1 Jan 2024",
        "session_1": [{"dia_id": "D1:1", "speaker": "Alice", "text": "My cat is Luna."}]},
        "qa": [{"question": "Name of cat?", "answer": "Luna", "evidence": ["D1:1"], "category": 4},
               {"question": "Unknown detail?", "adversarial_answer": "fake", "category": 5}]}]
    turns = turns_for(dataset[0]["conversation"])
    assert (turns[0].source_id, turns[0].speaker, turns[0].date) == ("D1:1", "Alice", "1 Jan 2024")
    selected = select_questions(dataset, per_category=1, categories=(4,))
    assert selected == select_questions(dataset, per_category=1, categories=(4,))
    assert selected[0]["id"] == "sample-1:0"
    assert select_questions(dataset, per_category=1, categories=(5,))[0]["answer"] == ""
    assert score_locomo("Luna", "Luna", 4) == 1.0
    assert score_locomo("No information available", "", 5) == 1.0


def test_reader_citations_only_reference_visible_turns():
    class CitationBackend:
        def generate(self, prompt, **kwargs):
            return '{"answer":"Luna","source_ids":["D1:1","D1:2"]}'

    turns = [SourceTurn("D1:1", "s1", "", "Alice", "My cat is Luna.", 0),
             SourceTurn("D1:2", "s1", "", "Bob", "x" * 500, 1)]
    result = CareSourceFirst(turns, evidence_chars=200).answer(
        "What is Alice's cat's name?", CitationBackend(), compile_mode="never")
    assert result["bundle_ids"] == ["D1:1"]
    assert result["source_ids"] == ["D1:1"]


def test_locomo_runner_scores_and_resumes_without_gold_in_prompts(tmp_path):
    dataset = [{"sample_id": "sample-1", "conversation": {
        "speaker_a": "Alice", "speaker_b": "Bob", "session_1_date_time": "1 Jan 2024",
        "session_1": [{"dia_id": "D1:1", "speaker": "Alice", "text": "My cat is Luna."}]},
        "qa": [{"question": "Name of cat?", "answer": "Luna", "evidence": ["D1:1"], "category": 4}]}]
    dataset_path = tmp_path / "locomo.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    class AnswerBackend:
        def __init__(self):
            self.calls = []
        def generate(self, prompt, **kwargs):
            self.calls.append(prompt)
            return '{"answer":"Luna","source_ids":["D1:1"]}'

    backend = AnswerBackend()
    first = run(dataset_path, tmp_path / "results", backend,
                per_category=1, categories=(4,))
    assert first["aggregate"]["bm25"]["mean_f1"] == 1.0
    assert first["aggregate"]["care"]["mean_f1"] == 1.0
    assert len(backend.calls) == 2
    assert all('"answer": "Luna"' not in prompt and '"evidence"' not in prompt
               for prompt in backend.calls)
    run(dataset_path, tmp_path / "results", backend,
        per_category=1, categories=(4,))
    assert len(backend.calls) == 2


def test_temporal_hints_are_anchored_to_source_dates_and_compiler_falls_back():
    class TruncatedCompiler:
        def __init__(self):
            self.prompts = []
        def generate(self, prompt, **kwargs):
            self.prompts.append(prompt)
            if "selected_ids" in prompt:
                return '{"selected_ids": ['  # Simulates a length-truncated response.
            return '{"answer":"September 2023","source_ids":["D1:1"]}'

    turn = SourceTurn("D1:1", "s1", "7:55 pm on 9 August, 2023", "Alice",
                      "The talent show is next month.", 0)
    backend = TruncatedCompiler()
    result = CareSourceFirst([turn]).answer("When is the talent show?", backend)
    assert result["answer"] == "September 2023"
    assert result["compiler_error"] == "invalid_compiler_json"
    assert "'next month' relative to 2023-08-09 = September 2023" in backend.prompts[-1]
