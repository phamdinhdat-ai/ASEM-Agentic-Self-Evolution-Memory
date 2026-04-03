"""Utility update and experience consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .backends.base import InferenceBackend
from .memory_bank import MemoryBank
from .note import NoteConstructor


@dataclass
class UtilityUpdater:
    """Update Q-values and consolidate experience into memory."""

    backend: InferenceBackend
    alpha: float
    q0: float
    summary_prompt_template: str
    note_constructor: Optional[NoteConstructor] = None

    def update(
        self,
        reward: float,
        used_notes: List[object],
        memory_bank: MemoryBank,
        query: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> None:
        for note in used_notes:
            new_q = note.q + self.alpha * (reward - note.q)
            memory_bank.update(note.id, {"q": new_q})

        if query is None or answer is None or self.note_constructor is None:
            return

        prompt = self.summary_prompt_template.format(
            query=query,
            answer=answer,
            reward=reward,
        )
        summary = self.backend.generate(prompt)
        new_note = self.note_constructor.build(summary, datetime.utcnow())
        memory_bank.add(new_note)
