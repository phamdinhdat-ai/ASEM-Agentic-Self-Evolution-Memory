"""G7: the evidence-recall instrument and the post-fit context telemetry.

Two things are tested here:

1. ``AnswerAgent.last_rendered_note_ids`` records the notes that actually
   reached the prompt after ``_select`` + ``_fit`` — NOT the retriever's
   candidate list. The distinction is the whole point of the metric: a note the
   retriever found but the context builder dropped is unavailable to the model
   exactly as if it had never been retrieved.

2. ``analysis/evidence_recall.py`` joins the frozen bank's ``session_id``
   vocabulary (``s1``) to LoCoMo gold evidence (``D1:3``) and reports session
   recall per category.

Run:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python -m pytest tests/test_evidence_recall.py -q
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from asem.answer_agent import AnswerAgent
from asem.note import Note

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import evidence_recall as ev  # noqa: E402


class _Backend:
    """Minimal backend: records prompts, returns a fixed answer."""

    def __init__(self) -> None:
        self.prompts = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return "an answer"

    def embed(self, text: str) -> np.ndarray:
        return np.zeros(4, dtype=float)


def _note(nid: str, text: str) -> Note:
    v = np.ones(4, dtype=float)
    return Note(
        id=nid,
        c=text,
        t=datetime(2022, 1, 1),
        K=[],
        G=[],
        X=text,
        e=v,
        L=[],
        z=v,
        q=0.5,
    )


def _agent(direct: bool = True, **kw) -> AnswerAgent:
    return AnswerAgent(
        backend=_Backend(),
        prompt_template="{query}\n{candidates}",
        baseline_prompt_template="{query}\n{context}",
        direct_mode=direct,
        **kw,
    )


def test_rendered_ids_reflect_the_context_cap_not_the_candidates() -> None:
    """`max_context_notes` drops candidates; the record must show what survived."""
    agent = _agent(max_context_notes=2)
    candidates = [_note(f"n{i}", f"fact {i}") for i in range(5)]

    agent.direct_answer("q", candidates)

    assert agent.last_rendered_note_ids == ["n0", "n1"], (
        "rendered ids must be the post-cap set, not all five candidates"
    )


def test_rendered_ids_are_unset_before_any_call() -> None:
    assert _agent().last_rendered_note_ids is None


def test_rendered_ids_cover_the_distil_path_too() -> None:
    agent = _agent(direct=False, max_context_notes=1)
    # The backend returns a valid distil payload, so the parse path succeeds and
    # the returned candidate list is the post-fit set.
    agent.backend.generate = lambda prompt, **kw: '{"answer": "x", "used_notes": []}'
    candidates = [_note(f"n{i}", f"fact {i}") for i in range(3)]
    agent.distil_and_answer("q", candidates)
    assert agent.last_rendered_note_ids == ["n0"]


# ---------------------------------------------------------------------------
# The instrument's join
# ---------------------------------------------------------------------------

def test_gold_sessions_maps_dia_ids_to_bank_vocabulary() -> None:
    assert ev.gold_sessions(["D1:3"]) == {"s1"}
    assert ev.gold_sessions(["D2:8", "D1:1"]) == {"s1", "s2"}
    assert ev.gold_sessions(["D10:2"]) == {"s10"}
    # Rows with no parseable evidence contribute nothing (adversarial rows).
    assert ev.gold_sessions([]) == set()
    assert ev.gold_sessions(["garbage"]) == set()


def test_evaluate_arm_reports_session_recall(tmp_path: Path) -> None:
    # A one-conversation dataset with two gold sessions on q0 and one on q1.
    dataset = [
        {
            "sample_id": "conv-0",
            "qa": [
                {"question": "q0", "evidence": ["D1:3", "D2:4"]},
                {"question": "q1", "evidence": ["D3:1"]},
            ],
        }
    ]
    ds_path = tmp_path / "ds.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    # A bank: notes n1..n3 carry sessions s1..s3.
    bank_root = tmp_path / "banks"
    conv_dir = bank_root / "locomo_0000"
    conv_dir.mkdir(parents=True)
    db = conv_dir / "asem_thg.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, session_id TEXT)")
    conn.executemany(
        "INSERT INTO notes VALUES (?, ?)",
        [("n1", "s1"), ("n2", "s2"), ("n3", "s3"), ("n4", "s9")],
    )
    conn.commit()
    conn.close()

    preds = [
        # q0 retrieves both gold sessions -> full hit.
        {"conversation_id": "locomo_0000", "idx": 0, "category_name": "multi_hop",
         "rendered_notes": ["n1", "n2", "n4"]},
        # q1 retrieves only a non-gold session -> miss.
        {"conversation_id": "locomo_0000", "idx": 1, "category_name": "temporal",
         "rendered_notes": ["n4"]},
    ]
    preds_path = tmp_path / "preds.jsonl"
    preds_path.write_text(
        "\n".join(json.dumps(r) for r in preds), encoding="utf-8"
    )

    res = ev.evaluate_arm(str(preds_path), str(ds_path), str(bank_root))
    mh = res["per_category"]["multi_hop"]
    assert mh["n"] == 1 and mh["session_recall"] == 1.0 and mh["all_sessions"] == 1.0
    tmp = res["per_category"]["temporal"]
    assert tmp["n"] == 1 and tmp["session_recall"] == 0.0
    assert res["used_legacy_fallback"] == 0


def test_evaluate_arm_falls_back_to_used_notes_when_rendered_absent(tmp_path: Path) -> None:
    dataset = [{"sample_id": "c", "qa": [{"question": "q", "evidence": ["D1:1"]}]}]
    ds_path = tmp_path / "ds.json"
    ds_path.write_text(json.dumps(dataset), encoding="utf-8")

    bank_root = tmp_path / "banks"
    conv_dir = bank_root / "locomo_0000"
    conv_dir.mkdir(parents=True)
    conn = sqlite3.connect(conv_dir / "asem_thg.sqlite")
    conn.execute("CREATE TABLE notes (id TEXT PRIMARY KEY, session_id TEXT)")
    conn.execute("INSERT INTO notes VALUES ('n1', 's1')")
    conn.commit()
    conn.close()

    preds_path = tmp_path / "p.jsonl"
    preds_path.write_text(
        json.dumps(
            {"conversation_id": "locomo_0000", "idx": 0,
             "category_name": "cat", "used_notes": ["n1"]}
        ),
        encoding="utf-8",
    )

    res = ev.evaluate_arm(str(preds_path), str(ds_path), str(bank_root))
    assert res["used_legacy_fallback"] == 1
    assert res["per_category"]["cat"]["session_recall"] == 1.0
