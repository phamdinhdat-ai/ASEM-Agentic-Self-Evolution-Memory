"""Full-dataset sharding and append-only resume contracts."""

import json
import pytest

from scripts.run_care_full_locomo import ModelBackendError, SHARDS, run, selected_questions
from scripts.merge_care_full_locomo import merge


def dataset():
    return [{"sample_id": sample_id,
             "conversation": {"session_1_date_time": "1 January 2024",
                              "session_1": [{"dia_id": "D0", "speaker": "Sam",
                                             "text": "I like hiking."}]},
             "qa": [{"category": 4, "question": "What does Sam like?",
                     "answer": "hiking", "evidence": ["D0"]}]}
            for group in SHARDS for sample_id in group]


class Backend:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return '{"answer":"hiking","source_ids":["D0"],"search_queries":[]}'


def test_shards_partition_every_question_once():
    data = dataset()
    selected = [item["id"] for shard in range(5)
                for item in selected_questions(data, shard)]
    assert len(selected) == len(set(selected)) == 10


def test_shard_resume_uses_saved_events_without_new_calls(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(dataset()), encoding="utf-8")
    output = tmp_path / "result"
    backend = Backend()
    report = run(path, output, backend, shard_index=0, max_new_calls=10)
    assert report["aggregate"]["bridge_v2"]["n"] == 2
    assert report["aggregate"]["relation_temporal"]["n"] == 2
    assert len(backend.calls) == 4
    assert len((output / "events.jsonl").read_text(encoding="utf-8").splitlines()) == 4
    resumed = Backend()
    run(path, output, resumed, shard_index=0, max_new_calls=10)
    assert resumed.calls == []


def test_backend_failure_does_not_checkpoint_an_empty_prediction(tmp_path):
    class Unavailable:
        calls = []

        def generate(self, prompt, **kwargs):
            raise ModelBackendError("API unavailable")

    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(dataset()), encoding="utf-8")
    output = tmp_path / "result"
    with pytest.raises(ModelBackendError):
        run(path, output, Unavailable(), shard_index=0, max_new_calls=10,
            model_info={"repo": "configured-model", "provider": "openai-compatible"})
    assert not (output / "events.jsonl").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"]["repo"] == "configured-model"


def test_merge_verifies_complete_partition(tmp_path):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(dataset()), encoding="utf-8")
    root = tmp_path / "shards"
    for index in range(5):
        output = root / f"shard-{index}" / f"care_relation_full_s{index}"
        run(path, output / "results", Backend(), shard_index=index, max_new_calls=10)
        (output / "status.json").write_text('{"state":"complete"}', encoding="utf-8")
    report = merge(path, root)
    assert report["n_questions"] == 10
    assert report["n_predictions"] == 20
    assert report["paired"]["all"]["mean_delta"] == 0
