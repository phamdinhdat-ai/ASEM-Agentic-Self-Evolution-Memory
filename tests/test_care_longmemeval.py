"""Check the external CARE adapter excludes benchmark labels from prompts."""

from pathlib import Path

from scripts.run_care_longmemeval import run, turns_for_record


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.calls.append({"prompt_tokens": 10, "completion_tokens": 5})
        return '{"answer":"No information available","source_ids":[],"search_queries":[]}'


def test_longmemeval_adapter_and_trace_replay(tmp_path):
    import json
    dataset = json.loads(Path("datasets/longmemeval/longmemeval_oracle.json").read_text())
    turns = turns_for_record(dataset[0])
    assert len(turns) == sum(map(len, dataset[0]["haystack_sessions"]))
    assert all("has_answer" not in turn.text for turn in turns)
    backend = FakeBackend()
    report = run(Path("datasets/longmemeval/longmemeval_oracle.json"),
                 tmp_path / "out", backend, shard_index=0, question_limit=1,
                 max_new_calls=2)
    assert len(report["rows"]) == 1
    assert report["rows"][0]["systems"]["first"]["calls"] == 1
    assert report["rows"][0]["systems"]["conditional"]["calls"] == len(backend.calls)
    assert "has_answer" not in backend.prompts[0]
