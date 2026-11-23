"""Opt-in CARE-Bridge variant with source-grounded temporal proof checks."""

from __future__ import annotations

from typing import Any

from .bridge import CareBridge
from .sourcefirst import SourceTurn, TextBackend
from .temporal_relations import TemporalProofGuard


class CareTemporalBridge(CareBridge):
    """Run the cheap relation guard after each reader answer, before retry logic."""

    def __init__(self, turns: list[SourceTurn], *, evidence_chars: int = 10000):
        super().__init__(turns, evidence_chars=evidence_chars)
        self.temporal_guard = TemporalProofGuard()

    @staticmethod
    def _event_consistent(question: str, turn: SourceTurn) -> str | None:
        # The inherited check tests each citation in isolation. A relation proof
        # can combine cited turns while retaining the answer-to-source binding.
        return None

    def _reader(self, question: str, turns: list[SourceTurn], backend: TextBackend,
                *, correction: str = "") -> dict[str, Any]:
        result = super()._reader(question, turns, backend, correction=correction)
        if result["invalid_reason"] is not None or not result["source_ids"]:
            return result
        cited = [self.by_id[sid] for sid in result["source_ids"]]
        proof = self.temporal_guard.check(question, result["answer"], cited)
        result["temporal_proof"] = {"status": proof.status, "reason": proof.reason,
                                     "source_ids": list(proof.source_ids)}
        fallback = (next((reason for turn in cited
                          if (reason := CareBridge._event_consistent(question, turn))), None)
                    if proof.status == "unknown" else None)
        if proof.status == "incompatible" or fallback:
            result["answer"] = "No information available"
            result["source_ids"] = []
            result["invalid_reason"] = ("temporal_" + str(proof.reason)
                                        if proof.status == "incompatible" else fallback)
        return result
