"""NGMC Tier-0 write gate: deterministic novelty/redundancy short-circuit.

The Memory Manager (S2) LLM is the most expensive stage of ingestion (~1 call
per turn). This gate decides the unambiguous cases from embeddings alone and
only lets the *ambiguous band* fall through to the LLM:

    novelty = 1 - max_sim(z_new, z_existing)
      novelty >= tau_high   -> ADD       (clearly new topic, LLM skipped)
      max_sim  >= tau_redund-> NOOP      (near-duplicate, LLM skipped)
      otherwise             -> ambiguous (Memory Manager LLM decides)

The gate is conservative on purpose: UPDATE/DELETE (contradictions, evolution)
always stay with the LLM, and tau_redund is high enough that only near-verbatim
duplicates are gated. Audit stats record how often the gate fires and what the
LLM chooses inside the ambiguous band, so the thresholds can be calibrated from
data instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .logging_utils import get_logger
from .memory_manager import Op
from .note import Note

_log = get_logger("S2.gate")


@dataclass
class WriteGate:
    """Deterministic novelty/redundancy gate for the S2 write decision."""

    enabled: bool = True
    tau_high: float = 0.45      # novelty = 1 - max_sim; ADD when novelty >= tau_high
    tau_redund: float = 0.92    # NOOP when max_sim >= tau_redund (near-duplicate)

    # Audit: counts of gate decisions and of LLM verdicts in the ambiguous band
    stats: Dict[str, int] = field(default_factory=lambda: {
        "gate_add": 0,
        "gate_noop": 0,
        "ambiguous": 0,
        "amb_add": 0,
        "amb_update": 0,
        "amb_delete": 0,
        "amb_noop": 0,
    })

    def propose(
        self, note: Note, candidates: List[Note]
    ) -> Tuple[Optional[Op], float]:
        """Return ``(op, max_sim)`` for a new note against existing notes.

        ``op`` is ``None`` when the gate is not confident (ambiguous band) and
        the Memory Manager LLM should be consulted. Uses the raw-content
        embeddings ``z`` on both sides so the comparison is like-for-like.
        """
        if not self.enabled:
            return None, 0.0
        if not candidates:
            self.stats["gate_add"] += 1
            return Op.ADD, 0.0

        max_sim = max(self._cosine(note.z, c.z) for c in candidates)
        novelty = 1.0 - max_sim
        if novelty >= self.tau_high:
            self.stats["gate_add"] += 1
            _log.debug("gate -> ADD (novelty={:.3f} >= {:.3f})", novelty, self.tau_high)
            return Op.ADD, max_sim
        if max_sim >= self.tau_redund:
            self.stats["gate_noop"] += 1
            _log.debug("gate -> NOOP (max_sim={:.3f} >= {:.3f})", max_sim, self.tau_redund)
            return Op.NOOP, max_sim

        self.stats["ambiguous"] += 1
        _log.debug("gate -> ambiguous (max_sim={:.3f})", max_sim)
        return None, max_sim

    def record_ambiguous_llm(self, op: Optional[Op]) -> None:
        """Audit what the LLM chose on a gate-ambiguous turn."""
        if op is None:
            return
        key = "amb_" + str(op.value).lower()
        self.stats[key] = self.stats.get(key, 0) + 1
        _log.debug("gate audit | ambiguous -> LLM op={}", op.value)

    def summary(self) -> Dict[str, int]:
        """Snapshot of audit stats (copy, for logging)."""
        return dict(self.stats)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
