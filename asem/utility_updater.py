"""Utility update and experience consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import NoteConstructor

_logger = get_logger(__name__)


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
        _logger.debug("update | reward={:.3f} | used_notes={} | has_query={} | has_answer={}",
                      reward, [n.id for n in used_notes], query is not None, answer is not None)

        for note in used_notes:
            new_q = note.q + self.alpha * (reward - note.q)
            memory_bank.update(note.id, {"q": new_q})
            _logger.debug("update | note {} q: {:.3f} → {:.3f} (α={:.2f})",
                          note.id, note.q, new_q, self.alpha)

        if query is None or answer is None or self.note_constructor is None:
            if query is None:
                _logger.debug("update | skipping consolidation (no query)")
            elif self.note_constructor is None:
                _logger.debug("update | skipping consolidation (no note_constructor)")
            return

        _logger.info("update | consolidating experience → new note")
        prompt = self.summary_prompt_template.format(
            query=query,
            answer=answer,
            reward=reward,
        )
        summary = self.backend.generate(prompt)
        new_note = self.note_constructor.build(summary, datetime.utcnow())
        memory_bank.add(new_note)
        _logger.success("update | consolidated note {} (summary={!r})", new_note.id, summary[:100])
