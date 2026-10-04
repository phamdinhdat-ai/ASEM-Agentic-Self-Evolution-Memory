"""Multi-hop query decomposition (G3).

The G3 probes located the failure precisely. On LoCoMo category-1 questions the
gold evidence for one of the two hops is, for 13 of 16 incomplete questions,
**absent from the candidate pool entirely** — not merely outranked. The graph
walk cannot help: it reorders a pool the evidence is not in. Session
diversification recovers only 3 of 16. The evidence IS reachable by text (the
oracle recovered 16/16 with the gold turn's own words), so what is missing is a
QUERY that surfaces it — which is what this module supplies.

``QueryPlanner`` turns one multi-hop question into sub-questions, each retrieved
independently; the retriever then fuses the candidate pools ("recall then
precision", HippoRAG 2's split). Three modes:

* ``off``       — no planning; the retriever behaves exactly as before.
* ``heuristic`` — no LLM. Fires only on questions the existing
                  ``_classify_query_type`` reads as reasoning/mixed, and splits
                  them on coordinating conjunctions. The zero-cost fallback the
                  plan asks for.
* ``llm``       — one call, asked for the minimal set of sub-questions that
                  together answer the question. Falls back to the heuristic on
                  any parse failure, so a bad generation degrades rather than
                  drops the query.

The planner never raises: a planning failure must not lose a retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from .backends.base import InferenceBackend
from .logging_utils import get_logger

_log = get_logger("S4.query_planner")

# Coordinating conjunctions that separate the two hops of a multi-hop question.
# "and"/"or" and the semicolon are the ones LoCoMo actually uses ("What subject
# have Caroline and Melanie both painted?").
_SPLIT_RE = re.compile(r"\s*(?:;|\band\b|\bor\b)\s+", re.IGNORECASE)

DEFAULT_PLANNER_PROMPT = """\
You decompose a multi-hop question into the minimal set of independent
sub-questions whose answers together answer it. Each sub-question must be
self-contained: it must make sense on its own, name its subject, and not refer
to "it", "that", or "the other one".

Rules:
- Emit at most {max_subquestions} sub-questions.
- If the question is already single-hop, emit exactly one sub-question equal to
  the original question.
- Do not answer the question. Do not explain.

Respond with ONLY a JSON object:
{{"sub_questions": ["...", "..."]}}

Question: {query}
JSON:"""


@dataclass
class QueryPlanner:
    """Decompose a multi-hop query into independently retrievable sub-questions.

    Args:
        backend: the inference backend used in ``llm`` mode. May be ``None``,
            which forces the heuristic (and makes ``off``/``heuristic`` modes
            usable without any model).
        mode: ``"off"``, ``"heuristic"`` or ``"llm"``.
        max_subquestions: hard cap on emitted sub-questions.
        prompt_template: LLM prompt; must contain ``{query}`` and
            ``{max_subquestions}``.
        temperature: sampling temperature for the planning call.
    """

    backend: Optional[InferenceBackend] = None
    mode: str = "heuristic"
    max_subquestions: int = 3
    prompt_template: str = DEFAULT_PLANNER_PROMPT
    temperature: float = 0.0
    # Telemetry from the most recent ``plan`` call, for the per-item record.
    stats: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _classify(self, query: str) -> str:
        """The existing zero-cost query classifier, imported lazily.

        Reused rather than reimplemented so the planner and the retriever agree
        on vocabulary — but note it does NOT detect multi-hop: LoCoMo's
        "What subject have Caroline and Melanie both painted?" reads as
        ``factual`` (it contains "what") despite naming two sessions. It is
        recorded for telemetry, not used as the multi-hop gate.
        """
        from .retriever import HybridRetriever

        try:
            return HybridRetriever._classify_query_type(query)
        except Exception:  # noqa: BLE001 — classifier must never break planning
            return "unknown"

    @staticmethod
    def _has_conjunction(query: str) -> bool:
        """Whether a coordinating conjunction actually splits the question.

        The split must leave >=2 parts of >=2 words each, so "What did Caroline
        research?" (no conjunction) and a question ending in "and?" do not fire.
        """
        parts = [p for p in _SPLIT_RE.split(query or "") if len(p.split()) >= 2]
        return len(parts) >= 2

    def is_multi_hop(self, query: str) -> bool:
        """Whether the HEURISTIC should fire.

        Surface conjunctions only. The LLM mode deliberately does not use this —
        it asks the model, because the classifier above cannot tell a two-session
        question from a one-session one.
        """
        return self._has_conjunction(query)

    # ------------------------------------------------------------------
    def _heuristic(self, query: str) -> List[str]:
        """Split on coordinating conjunctions, keeping the original first.

        The original is always a sub-question: decomposition should ADD queries,
        never replace the one the base retrieval already ran.
        """
        parts = [p.strip(" ?.,") for p in _SPLIT_RE.split(query or "")]
        parts = [p for p in parts if len(p.split()) >= 2]
        subs = [query.strip()]
        for p in parts:
            if p.lower() != query.strip().lower() and p not in subs:
                subs.append(p)
        return subs[: self.max_subquestions]

    def _parse(self, raw: str) -> Optional[List[str]]:
        """Pull ``sub_questions`` out of a model response; None if unusable."""
        if not raw:
            return None
        # Tolerate a fenced block or surrounding prose.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except (ValueError, TypeError):
            return None
        subs = data.get("sub_questions") if isinstance(data, dict) else None
        if not isinstance(subs, list):
            return None
        out = [str(s).strip() for s in subs if str(s).strip()]
        return out or None

    def _llm(self, query: str) -> List[str]:
        """One planning call; falls back to the heuristic on any failure."""
        if self.backend is None:
            return self._heuristic(query)
        prompt = self.prompt_template.format(
            query=query, max_subquestions=self.max_subquestions
        )
        try:
            raw = self.backend.generate(prompt, temperature=self.temperature)
        except Exception as exc:  # noqa: BLE001 — never lose a retrieval
            _log.debug("Query planning call failed, using heuristic: {}", exc)
            return self._heuristic(query)
        subs = self._parse(raw)
        if not subs:
            _log.debug("Query planning parse failed, using heuristic")
            return self._heuristic(query)
        # Guarantee the original is present so the base retrieval is preserved.
        if not any(s.strip().lower() == query.strip().lower() for s in subs):
            subs.insert(0, query.strip())
        return subs[: self.max_subquestions]

    # ------------------------------------------------------------------
    def plan(self, query: str) -> List[str]:
        """Sub-questions to retrieve for ``query``.

        Returns ``[]`` when the planner is off or the query is single-hop, in
        which case the caller must leave retrieval untouched. Otherwise the list
        always contains at least the original query.
        """
        self.stats = {"planner_mode": self.mode, "planner_fired": False}
        if self.mode == "off" or not query:
            return []
        # `llm` asks the model for every query — the classifier cannot detect
        # multi-hop, and the model returning the original question unchanged is
        # the correct answer for a single-hop query. `heuristic` fires only on a
        # surface conjunction, since a split it cannot find is a split it cannot
        # use.
        if self.mode != "llm" and not self.is_multi_hop(query):
            self.stats["planner_query_type"] = self._classify(query)
            return []

        subs = self._llm(query) if self.mode == "llm" else self._heuristic(query)
        if not subs:
            return []
        self.stats["planner_fired"] = True
        self.stats["planner_query_type"] = self._classify(query)
        self.stats["planner_subquestions"] = len(subs)
        self.stats["planner_subquestion_list"] = subs
        return subs
