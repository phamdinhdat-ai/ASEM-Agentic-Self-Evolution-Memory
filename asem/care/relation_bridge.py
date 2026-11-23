"""Opt-in query-relation proof extension to CARE Temporal Bridge."""

from __future__ import annotations

from typing import Any

from .relation_proof import RelationProofGuard
from .sourcefirst import SourceTurn, TextBackend
from .temporal_bridge import CareTemporalBridge


class CareRelationBridge(CareTemporalBridge):
    def __init__(self, turns: list[SourceTurn], *, evidence_chars: int = 10000):
        super().__init__(turns, evidence_chars=evidence_chars)
        self.relation_guard = RelationProofGuard(self.turns)
        self._proof_trace: list[dict[str, Any]] = []

    def answer(self, question: str, backend: TextBackend, *, mode: str = "bridge") -> dict[str, Any]:
        self._proof_trace = []
        result = super().answer(question, backend, mode=mode)
        return {**result, "relation_proof_trace": list(self._proof_trace)}

    def _reader(self, question: str, turns: list[SourceTurn], backend: TextBackend,
                *, correction: str = "") -> dict[str, Any]:
        result = super()._reader(question, turns, backend, correction=correction)
        if result["invalid_reason"] is not None or not result["source_ids"]:
            return result
        proof = self.relation_guard.check(question, result["answer"],
                                          [self.by_id[sid] for sid in result["source_ids"]])
        result["relation_proof"] = {"status": proof.status,
                                    "missing": list(proof.missing),
                                    "counterevidence_ids": list(proof.counterevidence_ids)}
        self._proof_trace.append({**result["relation_proof"],
                                  "candidate_answer": result["answer"],
                                  "candidate_source_ids": list(result["source_ids"])})
        if proof.status in {"incomplete", "contradicted"}:
            result["answer"] = "No information available"
            result["source_ids"] = []
            result["invalid_reason"] = "relation_" + proof.status
        return result
