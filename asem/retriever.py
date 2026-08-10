"""Two-phase hybrid retrieval with value-aware re-ranking, multi-hop link traversal, and adaptive lambda."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Set, Tuple

import numpy as np

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import Note

_log = get_logger("S4.retriever")

# A4 — query-type indicators for adaptive lambda
_FACTUAL_PATTERNS = re.compile(
    r"\b(what|who|when|where|which|how many|how much|how often|how long)\b",
    re.IGNORECASE,
)
_REASONING_PATTERNS = re.compile(
    r"\b(why|how|would|could|should|might|likely|probably|explain|reason)\b",
    re.IGNORECASE,
)

# A4 — lambda values per query type
_LAMBDA_FACTUAL = 0.25     # favor semantic similarity for fact lookup
_LAMBDA_REASONING = 0.60    # favor learned utility for multi-hop / inference
_LAMBDA_DEFAULT = 0.40      # balanced (original paper default)

# A1 — max link-traversal hops
_MAX_LINK_HOPS = 3


@dataclass
class HybridRetriever:
    """Hybrid retrieval: similarity filter + value-aware re-rank + link traversal.

    Phase A: similarity-based candidate recall (unchanged)
    Phase B: value-aware composite re-rank with adaptive lambda (A4)
    Phase C: multi-hop link traversal from top candidates (A1)
    """

    backend: InferenceBackend
    k1: int
    k2: int
    delta: float
    lambda_weight: float
    use_zscore: bool = True

    # A1+A4 — new knobs
    enable_link_traversal: bool = True
    max_link_hops: int = 1           # 1 = only direct neighbors
    link_traversal_topn: int = 3     # how many linked neighbors to add
    enable_adaptive_lambda: bool = True

    # Stats (populated during retrieval for introspection / token accounting)
    stats: Dict[str, object] = field(default_factory=dict)

    def retrieve(self, query: str, M: MemoryBank) -> List[Note]:
        self.stats = {}

        # A4 — adaptive lambda based on query type
        lam = self._adaptive_lambda(query) if self.enable_adaptive_lambda else self.lambda_weight

        e_q = self.backend.embed(query)
        candidates = M.ann_search(e_q, k=self.k1)
        if not candidates:
            self.stats["phase_a_hits"] = 0
            _log.debug("Phase A: no candidates from ANN search")
            return []

        sims = [self._cosine(e_q, note.e) for note in candidates]
        filtered = [
            (note, sim)
            for note, sim in zip(candidates, sims)
            if sim > self.delta
        ]
        if not filtered:
            self.stats["phase_a_hits"] = 0
            _log.debug("Phase A: all {} candidates below delta={}", len(candidates), self.delta)
            return []

        self.stats["phase_a_hits"] = len(filtered)
        _log.debug("Phase A: {} / {} candidates pass delta={}", len(filtered), len(candidates), self.delta)

        notes, sim_scores = zip(*filtered)
        q_scores = [note.q for note in notes]
        if self.use_zscore:
            sim_norm = self._zscore(sim_scores)
            q_norm = self._zscore(q_scores)
        else:
            sim_norm = list(sim_scores)
            q_norm = list(q_scores)

        scored: List[Tuple[float, Note]] = []
        for note, s_norm, q_norm_val in zip(notes, sim_norm, q_norm):
            score = (1.0 - lam) * s_norm + lam * q_norm_val
            scored.append((score, note))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = scored[: self.k2]
        result = [note for _, note in top_k]

        self.stats["lambda_used"] = lam
        self.stats["query_type"] = self._classify_query_type(query)

        _log.info("Phase B: k2={}  lambda={:.2f}  query_type={}  top_scores={}",
                  self.k2, lam, self.stats["query_type"],
                  [f"{s:.3f}" for s, _ in top_k[:5]])

        # A1 — traverse link graph from top candidates
        if self.enable_link_traversal and result:
            linked = self._traverse_links(result, e_q, M)
            # Deduplicate by note ID (Note is unhashable due to ndarray fields)
            seen_ids = {n.id for n in result}
            for n in linked:
                if n.id not in seen_ids:
                    result.append(n)
                    seen_ids.add(n.id)
            self.stats["link_traversal_added"] = len(linked)
            self.stats["total_retrieved"] = len(result)
            _log.debug("Phase C: link traversal added {} notes, total={}", len(linked), len(result))

        return result

    # ------------------------------------------------------------------
    # A4: Adaptive lambda
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_query_type(query: str) -> str:
        """Classify query as factual, reasoning, or mixed."""
        has_factual = bool(_FACTUAL_PATTERNS.search(query))
        has_reasoning = bool(_REASONING_PATTERNS.search(query))
        if has_factual and not has_reasoning:
            return "factual"
        if has_reasoning and not has_factual:
            return "reasoning"
        if has_factual and has_reasoning:
            return "mixed"
        return "unknown"

    def _adaptive_lambda(self, query: str) -> float:
        """Return the appropriate lambda for this query type."""
        qtype = self._classify_query_type(query)
        if qtype == "factual":
            return _LAMBDA_FACTUAL
        if qtype == "reasoning":
            return _LAMBDA_REASONING
        return _LAMBDA_DEFAULT

    # ------------------------------------------------------------------
    # A1: Multi-hop link traversal
    # ------------------------------------------------------------------

    def _traverse_links(
        self,
        seed_notes: List[Note],
        query_embedding: np.ndarray,
        M: MemoryBank,
    ) -> List[Note]:
        """Follow the link graph from seed notes to discover linked neighbors.

        Only follows direct (1-hop) links by default.  Each linked neighbor
        is scored by similarity to the query and its utility.  The top
        `link_traversal_topn` are added to the retrieved set.
        """
        seen_ids: Set[str] = {n.id for n in seed_notes}
        candidate_notes: List[Tuple[float, Note]] = []

        for seed in seed_notes:
            if not seed.L:
                continue
            # Batch-lookup linked neighbors by ID
            linked_notes = M.get_notes_by_ids([l.target_id for l in seed.L])
            for neighbor in linked_notes:
                if neighbor.id in seen_ids:
                    continue
                seen_ids.add(neighbor.id)
                sim = self._cosine(query_embedding, neighbor.e)
                # Weight by utility — high-q linked neighbors are preferred
                score = sim * (0.5 + 0.5 * neighbor.q)
                candidate_notes.append((score, neighbor))

        candidate_notes.sort(key=lambda item: item[0], reverse=True)
        added = [note for _, note in candidate_notes[: self.link_traversal_topn]]
        return added

    # ------------------------------------------------------------------
    # Helpers (unchanged)
    # ------------------------------------------------------------------

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
