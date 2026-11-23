"""Check the frozen Qwen3-4B transfer adapter without model inference."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

from scripts.run_care_second_backbone import run


def test_second_backbone_cohort_and_checkpoint(tmp_path: Path) -> None:
    archive = Path("build/kaggle/care-second-backbone-4b-pilot/care_second_backbone_bundle.zip")
    with zipfile.ZipFile(archive) as bundle:
        cohort = bundle.read("cohort_ids.json")
    cohort_path = tmp_path / "cohort_ids.json"
    cohort_path.write_bytes(cohort)

    class FakeBackend:
        def __init__(self):
            self.calls = []

        def generate(self, prompt: str, **kwargs) -> str:
            self.calls.append({"prompt_tokens": 1, "completion_tokens": 1})
            return json.dumps({"answer": "No information available",
                               "source_ids": [], "search_queries": []})

    backend = FakeBackend()
    report = run(Path("datasets/locomo/locomo10.json"), cohort_path,
                 tmp_path / "results", backend, question_limit=2)
    assert len(report["rows"]) == 2
    assert report["manifest"]["model"]["repo"] == "Qwen/Qwen3-4B"
    assert report["aggregate"]["first"]["calls"] == 2
    assert report["aggregate"]["conditional"]["calls"] >= 2
    assert len(backend.calls) == report["aggregate"]["conditional"]["calls"]
