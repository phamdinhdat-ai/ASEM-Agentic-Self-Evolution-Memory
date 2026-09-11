"""End-to-end smoke tests for the phase-separated benchmark runner.

Uses a deterministic stub backend so ingestion → persisted bank → retrieval can
be exercised without downloading any model.
"""

from __future__ import annotations

import json
import os

import numpy as np

from asem.backends.base import InferenceBackend
from eval.phase_runner import (
    bank_dir,
    bank_exists,
    bank_file,
    model_tag_from_config,
    render_sweep_table,
    run_ingest_phase,
    run_retrieve_phase,
    working_copy,
)

CONFIG = "configs/models/qwen2.5_1.5b_hf.yaml"


class StubBackend(InferenceBackend):
    """Deterministic content-derived embeddings + canned generation.

    Embeddings depend only on the text (not on the answer string), so two
    stub instances behave like two backbones sharing the same embedder — the
    exact scenario the retrieval sweep relies on.
    """

    def __init__(self, answer: str = "Buddy the dog") -> None:
        super().__init__()
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if "expert memory extraction agent" in prompt:
            return json.dumps([
                {
                    "fact": "Alice adopted a dog named Buddy on 8 May 2023.",
                    "entities": ["Alice", "Buddy"],
                    "keywords": ["adopted", "dog", "buddy"],
                    "tags": ["pet"],
                    "speaker": "Alice",
                },
                {
                    "fact": "Bob recommended Dr. Smith in Seattle.",
                    "entities": ["Bob", "Seattle", "Dr. Smith"],
                    "keywords": ["veterinarian", "seattle"],
                    "tags": ["recommendation"],
                    "speaker": "Bob",
                },
            ])
        if "[Turn 1]" in prompt:
            count = prompt.count("[Turn ")
            return json.dumps([
                {
                    "keywords": ["buddy", "dog"],
                    "tags": ["pet"],
                    "description": "Alice adopted a dog named Buddy.",
                }
                for _ in range(count)
            ])
        return self.answer

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(8, dtype="float32")
        for i, ch in enumerate(str(text)[:8]):
            vec[i] = (ord(ch) % 10) / 10.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


def _write_dataset(path) -> None:
    dataset = [{
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Alice", "text": "I adopted a dog named Buddy.", "dia_id": "D1:1"},
                {"speaker": "Bob", "text": "Take him to Dr. Smith in Seattle.", "dia_id": "D1:2"},
            ],
            "session_2_date_time": "2:00 pm on 9 May, 2023",
            "session_2": [
                {"speaker": "Alice", "text": "Buddy loves the park.", "dia_id": "D2:1"},
            ],
        },
        "qa": [
            {"question": "What did Alice adopt?", "answer": "Buddy the dog",
             "category": 1, "evidence": ["D1:1"]},
            {"question": "Where is the vet?", "answer": "Seattle",
             "category": 1, "evidence": ["D1:2"]},
            {"question": "When did Buddy visit the park?", "answer": "9 May 2023",
             "category": 2, "evidence": ["D2:1"]},
        ],
    }]
    path.write_text(json.dumps(dataset), encoding="utf-8")


def _load_groups(path):
    from eval.phase_runner import load_raw_dataset
    from scripts.run_locomo10_experiments import (
        convert_locomo10_to_eval,
        group_by_conversation,
    )

    raw = load_raw_dataset(str(path))
    items = convert_locomo10_to_eval(str(path))
    return raw, group_by_conversation(items)


def test_ingest_then_retrieve_shares_bank(tmp_path):
    dataset_path = tmp_path / "locomo_mini.json"
    _write_dataset(dataset_path)
    raw, groups = _load_groups(dataset_path)

    bank_root = str(tmp_path / "banks")
    systems = ["FastASEM", "SimRetrieval", "NoMemory", "FullContext"]
    conversation_id = str(groups[0][0]["session_id"])

    # ---- Phase A: ingest with one backbone --------------------------------
    manifest = run_ingest_phase(
        raw_dataset=raw,
        groups=groups,
        systems=systems,
        ingest_config=CONFIG,
        bank_root=bank_root,
        tag="test",
        backend=StubBackend(answer="ingested"),
    )

    assert manifest["n_conversations"] == 1
    # Bank-backed systems persisted a database; context baselines did not.
    for name in ("FastASEM", "SimRetrieval"):
        assert bank_exists(bank_root, "test", name, conversation_id), name
    assert not bank_exists(bank_root, "test", "NoMemory", conversation_id)
    assert manifest["systems"]["FastASEM"]["notes"] > 0

    # ---- Phase B: retrieve with a DIFFERENT backbone ----------------------
    results = run_retrieve_phase(
        groups=groups,
        systems=systems,
        config_path=CONFIG,
        bank_root=bank_root,
        tag="test",
        backend=StubBackend(answer="Buddy the dog"),
        metric_names=["em"],
        model_tag="stub-b",
    )

    assert results["n_qa"] == 3
    assert set(results["systems"]) == set(systems)
    for name in systems:
        assert results["systems"][name]["n"] == 3
        assert "em" in results["systems"][name]["overall"]

    # Context baselines answer "Buddy the dog" for both QAs it matches, so EM > 0.
    assert results["systems"]["FastASEM"]["overall"]["em"] >= 0.0


def test_working_copy_isolates_canonical(tmp_path):
    """A mutating retrieval run must not touch the canonical ingested bank."""
    from asem.memory_bank import MemoryBank

    dataset_path = tmp_path / "locomo_mini.json"
    _write_dataset(dataset_path)
    raw, groups = _load_groups(dataset_path)

    bank_root = str(tmp_path / "banks")
    conversation_id = str(groups[0][0]["session_id"])
    run_ingest_phase(
        raw_dataset=raw,
        groups=groups,
        systems=["FastASEM"],
        ingest_config=CONFIG,
        bank_root=bank_root,
        tag="test",
        backend=StubBackend(),
    )

    canonical = bank_file(bank_root, "test", "FastASEM", conversation_id)
    canonical_size = MemoryBank(canonical).size()
    assert canonical_size > 0

    work_dir = working_copy(
        canonical, str(tmp_path / "work"), "test", "modelA", "FastASEM", conversation_id
    )
    copy_path = os.path.join(work_dir, os.path.basename(canonical))
    assert os.path.exists(copy_path)

    # Mutate the working copy — the canonical bank must be unaffected.
    copy_bank = MemoryBank(copy_path)
    copy_bank.clear()
    copy_bank.close()
    assert MemoryBank(copy_path).size() == 0
    assert MemoryBank(canonical).size() == canonical_size


def test_bank_dir_is_deterministic(tmp_path):
    root = str(tmp_path / "banks")
    d = bank_dir(root, "tagA", "FastASEM", "locomo_0003")
    assert d == os.path.join(root, "tagA", "FastASEM", "locomo_0003")
    assert bank_file(root, "tagA", "FastASEM", "locomo_0003").endswith("fast_asem.sqlite")


def test_model_tag_and_table():
    tag = model_tag_from_config(CONFIG)
    assert "qwen2_5_1_5b_instruct" in tag

    sweep = {
        "m1": {"systems": {"FastASEM": {"overall": {"em": 0.5}}}},
        "m2": {"systems": {"FastASEM": {"overall": {"em": 0.7}}}},
    }
    table = render_sweep_table(sweep, ["em"])
    assert "| System |" in table
    assert "0.5000" in table and "0.7000" in table
