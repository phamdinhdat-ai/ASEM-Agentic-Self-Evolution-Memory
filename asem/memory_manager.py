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

    # Batch prompt: decide operations for multiple turns in one LLM call
    _BATCH_PROMPT = (
        "Decide memory write operations for MULTIPLE new pieces of information. "
        "Output a JSON array with one decision per item, in order.\n\n"
        "For each item, choose: ADD (new info), UPDATE (similar note exists, "
        "provide target_id of the existing note), DELETE (contradicted), "
        'or NOOP (irrelevant).\n\n'
        "Existing notes (for reference):\n{memory}\n\n"
        "New items:\n{items_text}\n\n"
        "Output ONLY a JSON array like:\n"
        '[{{"op": "ADD|UPDATE|DELETE|NOOP", "target_id": "<id or null>"}}, ...]\n'
        "No markdown fences, no extra text."
    )

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

    def select_ops_batch(
        self, contents: List[str], M_old: List[Note]
    ) -> List[Tuple[Op, Optional[str]]]:
        """Decide write operations for multiple contents in ONE LLM call.

        Args:
            contents: List of new content strings.
            M_old: Existing notes for context.

        Returns:
            List of (Op, target_id_or_None), one per content item.
        """
        n = len(contents)
        _logger.info("select_ops_batch | items={} | existing_notes={}", n, len(M_old))

        # Build items text with numbered entries
        items_lines = []
        for i, c in enumerate(contents):
            items_lines.append(f"[{i + 1}] {c}")
        items_text = "\n\n".join(items_lines)

        # Build memory context (capped at top-20)
        memory_notes = M_old[:20]
        context = [
            {"id": note.id, "keywords": note.K, "tags": note.G, "description": note.X}
            for note in memory_notes
        ]
        memory_json = json.dumps(context)

        prompt = self._BATCH_PROMPT.format(memory=memory_json, items_text=items_text)
        raw = self.backend.generate(prompt)

        decisions = self._parse_batch_decision(raw, n)
        _logger.info("select_ops_batch → {} decisions", len(decisions))
        return decisions

    def _parse_batch_decision(
        self, raw: str, expected_count: int
    ) -> List[Tuple[Op, Optional[str]]]:
        """Parse JSON array of decisions from batch LLM output."""
        cleaned = raw.strip()
        if "```json" in cleaned:
            start = cleaned.find("```json") + 7
            end = cleaned.find("```", start)
            if end > start:
                cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.find("```") + 3
            end = cleaned.find("```", start)
            if end > start:
                cleaned = cleaned[start:end].strip()

        bracket_start = cleaned.find("[")
        bracket_end = cleaned.rfind("]")
        if bracket_start >= 0 and bracket_end > bracket_start:
            cleaned = cleaned[bracket_start:bracket_end + 1]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            _logger.warning("select_ops_batch | JSON parse failed, raw={!r}", raw[:200])
            return [(Op.ADD, None)] * expected_count

        if not isinstance(data, list):
            return [(Op.ADD, None)] * expected_count

        results: List[Tuple[Op, Optional[str]]] = []
        for item in data:
            if not isinstance(item, dict):
                results.append((Op.ADD, None))
                continue
            op_value = str(item.get("op", "ADD")).upper()
            target_id = item.get("target_id")
            if op_value in Op.__members__:
                results.append((Op[op_value], str(target_id) if target_id else None))
            else:
                results.append((Op.ADD, None))

        while len(results) < expected_count:
            results.append((Op.ADD, None))
        return results[:expected_count]

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
