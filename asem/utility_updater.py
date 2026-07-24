"""Utility update and experience consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import NoteConstructor

_log = get_logger("S5.utility")


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
            old_q = note.q
            new_q = note.q + self.alpha * (reward - note.q)
            memory_bank.update(note.id, {"q": new_q})
            _log.debug("Q-update | id={}  q: {:.3f} -> {:.3f}  (reward={:.3f}, alpha={:.2f})",
                       note.id[:8], old_q, new_q, reward, self.alpha)

        if query is None or answer is None or self.note_constructor is None:
            return

        prompt = self.summary_prompt_template.format(
            query=query,
            answer=answer,
            reward=reward,
        )
        summary = self.backend.generate(prompt)
        new_note = self.note_constructor.build(summary, datetime.now(timezone.utc))
        memory_bank.add(new_note)
        _log.info("Experience consolidated | new_note_id={}  summary={!r}",
                  new_note.id[:8], summary[:60])
