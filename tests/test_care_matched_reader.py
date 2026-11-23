"""Exercise the paired reader path without calling an answer model."""

import json
import re
from pathlib import Path

from scripts.prepare_care_matched_reader_kaggle import locked_rankings
from scripts.run_care_matched_reader import run


class FakeBackend:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt_tokens": 10, "completion_tokens": 5})
        if len(self.calls) == 1:
            return json.dumps({"answer": "No information available",
                               "source_ids": [], "search_queries": ["Nate friends"]})
        source = re.search(r"\[(D\d+:\d+)\]", prompt).group(1)
        return json.dumps({"answer": "Nate", "source_ids": [source],
                           "search_queries": []})


def test_matched_reader_shares_first_call_and_scores_after_generation(tmp_path):
    rankings = tmp_path / "rankings.json"
    rankings.write_bytes(locked_rankings())
    output = tmp_path / "out"
    backend = FakeBackend()
    result = run(Path("datasets/locomo/locomo10.json"), rankings, output, backend,
                 shard_index=0, question_limit=1, max_new_calls=10)
    row = result["rows"][0]
    assert len(backend.calls) == 2
    assert row["systems"]["dense_8_first"]["model_calls"] == 1
    assert row["systems"]["dense_8_conditional"]["model_calls"] == 2
    assert row["systems"]["dense_8_first"]["answer"] == "No information available"
    assert row["systems"]["dense_8_conditional"]["answer"] == "Nate"
    assert result["aggregate"]["dense_8_conditional"]["prompt_tokens"] == 20
