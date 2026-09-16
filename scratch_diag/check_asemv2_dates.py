"""Diagnostic: do ASEMv2 / FastASEM notes carry session date + time context?

Drives the real ingestion path used by `scripts/build_static_banks.py`
(`eval.phase_runner.ingest_system`) with a deterministic stub LLM, then dumps
every note's temporal/session metadata and the prompts the LLM saw.

Run:  python scratch_diag/check_asemv2_dates.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml  # noqa: E402

# Silence loguru: on PowerShell its stderr output is wrapped as a
# NativeCommandError and truncates the report.
try:
    from loguru import logger as _loguru

    _loguru.remove()
except Exception:  # noqa: BLE001
    pass

from asem.backends.base import InferenceBackend  # noqa: E402
from eval.phase_runner import (  # noqa: E402
    build_eval_system,
    close_system,
    extract_sessions,
    finalize_system,
    ingest_system,
    system_bank_size,
)

BASE_CONFIG = "configs/models/qwen2.5_1.5b_hf.yaml"

DATASET = [{
    "conversation": {
        "speaker_a": "Alice",
        "speaker_b": "Bob",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [
            {"speaker": "Alice", "text": "I adopted a dog named Buddy yesterday.", "dia_id": "D1:1"},
            {"speaker": "Bob", "text": "Take him to Dr. Smith in Seattle.", "dia_id": "D1:2"},
        ],
        "session_2_date_time": "2:00 pm on 9 May, 2023",
        "session_2": [
            {"speaker": "Alice", "text": "Buddy loves the park.", "dia_id": "D2:1"},
        ],
    },
    "qa": [{"question": "What did Alice adopt?", "answer": "Buddy the dog",
            "category": 1, "evidence": ["D1:1"]}],
}]


class StubBackend(InferenceBackend):
    """Answers each ASEMv2 batch prompt with the shape it expects."""

    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []
        self.kinds: list[str] = []

    # A fixed fact keeps note counts meaningful: the point of this probe is the
    # metadata stamped on each note, not what the (stub) LLM extracted.
    FIXED_ASEMV2 = [{
        "content": "[Alice] Alice adopted a dog named Buddy.",
        "keywords": ["alice", "adopted", "dog", "buddy"],
        "tags": ["pet"],
        "description": "Alice adopted a dog named Buddy.",
        "entities": ["Alice", "Buddy"],
        "speaker": "Alice",
    }]
    FIXED_FAST = [{
        "fact": "Alice adopted a dog named Buddy.",
        "entities": ["Alice", "Buddy"],
        "keywords": ["alice", "adopted", "dog", "buddy"],
        "tags": ["pet"],
        "speaker": "Alice",
    }]

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if "analyzing a multi-turn conversation" in prompt:      # ASEMv2 (P4)
            self.kinds.append("extract")
            return json.dumps(self.FIXED_ASEMV2)
        if "expert memory extraction agent" in prompt:            # FastASEM
            self.kinds.append("extract-fast")
            return json.dumps(self.FIXED_FAST)
        if "decide what operation to perform" in prompt:         # ASEMv2 (P5)
            self.kinds.append("memory_ops")
            return "[]"   # -> the ingestor falls back to "ADD every note"
        if "identify ALL meaningful relationships" in prompt:     # ASEMv2 (P6)
            self.kinds.append("link")
            return "[]"
        self.kinds.append("other")
        return "ANSWER"

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(8, dtype="float32")
        for i, ch in enumerate(str(text)[:8]):
            vec[i] = (ord(ch) % 10) / 10.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


def _temp_config() -> str:
    """Copy the base config with max_retries=0 so the stub fully drives the flow."""
    with open(BASE_CONFIG, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["llm_retry"] = {"max_retries": 0}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_no_retry.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)
    return path


def _dump(name: str, notes, sessions) -> None:
    print(f"\n{'=' * 92}")
    print(f"{name}: {len(notes)} notes")
    print("=" * 92)
    print(f"{'#':<3}{'session_id':<12}{'session_date':<26}{'timestamp_iso':<22}"
          f"{'t':<27}{'tz':<6}{'speaker':<9}content")
    for i, n in enumerate(notes):
        t = n.t
        tz = "None" if (t is None or t.tzinfo is None) else "UTC"
        print(f"{i:<3}{str(n.session_id):<12}{str(n.session_date):<26}"
              f"{str(n.timestamp_iso):<22}{str(t):<27}{tz:<6}{str(n.speaker):<9}{n.c[:38]}")

    print()
    checks = []

    # 1. Every note carries a session id and date.
    checks.append(("every note has session_id + session_date",
                   all(n.session_id and n.session_date for n in notes)))

    # 2. session_date matches the session the turn came from (fixed stub -> one
    #    note per ingested session, in order).
    if len(notes) == len(sessions):
        checks.append(("session_date matches each session, in order",
                       all(str(n.session_date) == s.date
                           for n, s in zip(notes, sessions))))

    # 3. timestamp_iso is the session's date rendered as ISO-8601 UTC.
    from asem.temporal import parse_session_datetime
    checks.append(("timestamp_iso == parse(session_date)",
                   all(n.timestamp_iso == parse_session_datetime(n.session_date)[1]
                       for n in notes)))

    # 4. note.t is timezone-aware (naive/aware mixing breaks date arithmetic).
    checks.append(("t is timezone-aware",
                   all(n.t is not None and n.t.tzinfo is not None for n in notes)))

    # 5. the parsed datetime and the ISO string agree.
    checks.append(("t and timestamp_iso agree",
                   all(n.t is not None and
                       n.t.strftime("%Y-%m-%dT%H:%M:%SZ") == n.timestamp_iso
                       for n in notes)))

    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    print(f"\n  distinct session_id   : {sorted({str(n.session_id) for n in notes})}")
    print(f"  distinct session_date : {sorted({str(n.session_date) for n in notes})}")
    print(f"  distinct timestamp_iso: {sorted({str(n.timestamp_iso) for n in notes})}")


def main() -> int:
    config = _temp_config()
    record = DATASET[0]
    sessions = extract_sessions(record["conversation"])
    print(f"Parsed {len(sessions)} sessions from the mini dataset:")
    for s in sessions:
        print(f"  session {s.num}: date={s.date!r} turns={len(s.turns)}")

    for name in ("ASEMv2", "FastASEM"):
        backend = StubBackend()
        db_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), f"_diag_bank_{name.lower()}"
        )
        # _make_bank never deletes the main DB, so clear it for a truthful count.
        shutil.rmtree(db_dir, ignore_errors=True)
        os.makedirs(db_dir, exist_ok=True)
        system = build_eval_system(name, config, db_dir, backend=backend)
        try:
            ingest_system(system, name, sessions)
            edges = finalize_system(system)
            size = system_bank_size(system)
            bank = getattr(getattr(system, "pipeline", None), "memory_bank", None)
            # Read the notes BEFORE closing: close_system shuts the SQLite handle.
            notes = bank.list_notes() if bank is not None else []
        finally:
            close_system(system)

        print(f"\n>>> {name}: bank size={size}, link edges={edges}")
        print(f"    prompt kinds seen: {backend.kinds}")

        # Does the session context actually reach the extraction LLM?
        extract_prompts = [p for p, k in zip(backend.prompts, backend.kinds)
                           if k.startswith("extract")]
        if extract_prompts:
            first = extract_prompts[0]
            s1 = sessions[0]
            print(f"    [{'PASS' if '## Conversation' in first else 'n/a '}] "
                  f"dialogue section present ('## Conversation')")
            print(f"    [{'PASS' if s1.date in first else 'FAIL'}] "
                  f"session date text reaches the prompt ({s1.date!r})")
            print(f"    [{'PASS' if f'Session {s1.num}' in first else 'FAIL'}] "
                  f"session header number reaches the prompt")
            turns_seen = sum(1 for t in s1.turns if t in first)
            print(f"    [{'PASS' if turns_seen == len(s1.turns) else 'FAIL'}] "
                  f"all {len(s1.turns)} session-1 turns reach the prompt "
                  f"({turns_seen}/{len(s1.turns)})")
            print(f"    extraction calls: {len(extract_prompts)} "
                  f"(one per session: {len(sessions)})")

        if notes:
            _dump(name, notes, sessions)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
