"""CARE with a shared first pass and pluggable repair expansion.

``CareTwoStage`` splits ``CareBridge.answer_with_initial`` into ``first_pass``
and ``repair`` so several repair variants can reuse one first model call.

* ``repair(mode="lexical")`` reproduces CareBridge exactly: same gate, same
  expansion (unstemmed + stemmed BM25, session neighbours, 6 + 6 hits per
  search), same correction text, same reader. Tested for equality.
* ``repair(mode="hybrid")`` makes one change: it also adds the top
  ``dense_k`` dense hits for the question and for every search. Everything else,
  including conversation-order serialization under the same character cap, is
  unchanged.
* ``repair(mode="graph")`` is ``hybrid`` with the dense ranker replaced by a
  graph ranker (layered memory graph, PPR influence mixed with dense score;
  ``asem.care.graph_memory``). Same counts, same serialization.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .bridge import CareBridge, _NO_INFO, _PLURAL_REQUEST
from .sourcefirst import SourceTurn, TextBackend

DenseRank = Callable[[str, int], list[SourceTurn]]


class CareTwoStage(CareBridge):
    def __init__(self, turns: list[SourceTurn], *, evidence_chars: int = 10000,
                 dense_rank: Optional[DenseRank] = None, dense_k: int = 6,
                 graph_rank: Optional[DenseRank] = None):
        super().__init__(turns, evidence_chars=evidence_chars)
        self.dense_rank = dense_rank
        self.graph_rank = graph_rank
        self.dense_k = dense_k

    def first_pass(self, question: str, backend: TextBackend,
                   initial: list[SourceTurn]) -> dict[str, Any]:
        pool = self.candidate_pool(question)
        risk = self.risk_reasons(question, pool)
        if _PLURAL_REQUEST.search(question):
            risk.append("plural_aggregation")
        first = self._reader(question, initial, backend)
        needs_repair = first["invalid_reason"] is not None or (
            _NO_INFO.casefold() in first["answer"].casefold()
            and (bool(first["search_queries"]) or bool(risk)))
        return {"first": first, "risk": risk, "needs_repair": needs_repair,
                "initial_ids": [turn.source_id for turn in initial]}

    def expansion(self, question: str, first: dict[str, Any], mode: str) -> list[SourceTurn]:
        if mode not in {"lexical", "hybrid", "graph"}:
            raise ValueError("mode must be lexical, hybrid or graph")
        extra = {"lexical": None, "hybrid": self.dense_rank, "graph": self.graph_rank}[mode]
        if mode == "hybrid" and extra is None:
            raise ValueError("hybrid repair needs a dense ranker")
        if mode == "graph" and extra is None:
            raise ValueError("graph repair needs a graph ranker")
        expanded = {turn.source_id for turn in self._hybrid_pool(question)}
        names = self._subject_names(question)
        queries = first["search_queries"] or [question]
        for query in queries:
            search = query
            if len(names) == 1 and not re.search(r"\b" + re.escape(names[0]) + r"\b", query, re.I):
                search = names[0] + " " + query
            expanded.update(turn.source_id for turn in self.stem_rank(search, limit=6))
            expanded.update(turn.source_id for turn in self.rank(search, limit=6))
            if extra is not None:
                expanded.update(turn.source_id for turn in extra(search, self.dense_k))
        if extra is not None:
            expanded.update(turn.source_id for turn in extra(question, self.dense_k))
        return [turn for turn in self.turns if turn.source_id in expanded]

    def repair(self, question: str, backend: TextBackend, fp: dict[str, Any],
               mode: str) -> dict[str, Any]:
        first = fp["first"]
        if not fp["needs_repair"]:
            return {**first, "candidate_ids": fp["initial_ids"], "risk_reasons": fp["risk"],
                    "escalated": False, "first_pass": None}
        second_turns = self.expansion(question, first, mode)
        correction = ("Previous evidence did not bind the named person to the cited first-person claim. "
                      "Check speaker labels strictly. " if first["invalid_reason"] == "speaker_mismatch"
                      else "Find direct evidence for the requested person, event and time. ")
        second = self._reader(question, second_turns, backend, correction=correction)
        return {**second, "candidate_ids": [turn.source_id for turn in second_turns],
                "risk_reasons": fp["risk"], "escalated": True,
                "first_pass": {key: first[key] for key in ("answer", "source_ids", "invalid_reason",
                                                         "bundle_ids", "search_queries")}}
