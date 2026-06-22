"""Memory manager for RL write operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .note import Note

_logger = get_logger(__name__)


class Op(str, Enum):
    """Write operations for memory bank updates."""

    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"


@dataclass
class MemoryManager:
    """Select memory write operations for new information."""

    backend: InferenceBackend
    prompt_template: str

    def select_op(self, x: str, M_old: List[Note]) -> Tuple[Op, Optional[Note]]:
        """Select a write operation and optional target note."""
        _logger.debug("select_op | content={!r} | existing_notes={}",
                      x[:120], [n.id for n in M_old])

        prompt = self._build_prompt(x, M_old)
        raw = self.backend.generate(prompt)
        op, target_id = self._parse_decision(raw)

        if op is None:
            _logger.warning("select_op | LLM parse failed, using heuristic fallback | raw={!r}", raw[:100])
            return self._heuristic_fallback(x, M_old)

        target = self._find_target(target_id, M_old)
        _logger.debug("select_op → op={} target_id={}", op.value, target_id or "-")
        return op, target

    def _build_prompt(self, x: str, M_old: List[Note]) -> str:
        # B4 — cap at top-k2=5 notes to prevent linear prompt growth
        context = [
            {
                "id": note.id,
                "keywords": note.K,
                "tags": note.G,
                "description": note.X,
            }
            for note in M_old
        ]
        return self.prompt_template.format(content=x, memory=json.dumps(context))

    def _parse_decision(self, raw: str) -> Tuple[Optional[Op], Optional[str]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None, None

        op_value = str(data.get("op", "")).upper()
        target_id = data.get("target_id")
        if op_value not in Op.__members__:
            return None, None
        return Op[op_value], str(target_id) if target_id else None

    def _heuristic_fallback(
        self, x: str, M_old: List[Note]
    ) -> Tuple[Op, Optional[Note]]:
        if not M_old:
            return Op.ADD, None
        return Op.UPDATE, M_old[0]

    def _find_target(self, target_id: Optional[str], M_old: List[Note]) -> Optional[Note]:
        if not target_id:
            return None
        for note in M_old:
            if note.id == target_id:
                return note
        return None
