"""Two-phase hybrid retrieval with value-aware re-ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .backends.base import InferenceBackend
from .memory_bank import MemoryBank
from .note import Note


@dataclass
class HybridRetriever:
    """Hybrid retrieval: similarity filter + value-aware re-rank."""

    backend: InferenceBackend
    k1: int
    k2: int
    delta: float
    lambda_weight: float
    use_zscore: bool = True

    def retrieve(self, query: str, M: MemoryBank) -> List[Note]:
        e_q = self.backend.embed(query)
        candidates = M.ann_search(e_q, k=self.k1)
        if not candidates:
            return []

        sims = [self._cosine(e_q, note.e) for note in candidates]
        filtered = [
            (note, sim)
            for note, sim in zip(candidates, sims)
            if sim > self.delta
        ]
        if not filtered:
            return []

        notes, sim_scores = zip(*filtered)
        q_scores = [note.q for note in notes]
        if self.use_zscore:
            sim_norm = self._zscore(sim_scores)
            q_norm = self._zscore(q_scores)
        else:
            sim_norm = list(sim_scores)
            q_norm = list(q_scores)

        scored = []
        for note, s_norm, q_norm_val in zip(notes, sim_norm, q_norm):
            score = (1.0 - self.lambda_weight) * s_norm + self.lambda_weight * q_norm_val
            scored.append((score, note))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [note for _, note in scored[: self.k2]]

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _zscore(values: List[float]) -> List[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        if std == 0:
            return [0.0 for _ in values]
        return [float((val - mean) / std) for val in values]
