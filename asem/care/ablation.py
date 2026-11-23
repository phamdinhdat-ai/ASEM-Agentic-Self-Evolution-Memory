"""CARE with switchable mechanisms, for the component ablation (docs/CARE_ablation_protocol_*.md).

``CareAblation`` subclasses ``CareTwoStage`` and changes nothing unless a flag is switched off. With
the default flags it is call-for-call identical to ``CareTwoStage.repair(mode="lexical")``, which is
itself tested equal to ``CareBridge`` (tests/test_care_ablation.py).

The reader prompt is never re-implemented: ``_reader`` runs the parent reader through a capture
wrapper and, only when a validation flag is off, re-applies the validation to the captured raw output.
So prompts are byte-identical to CARE's and arms that send the same prompt can share one model call.

Flags (arm in parentheses):
  require_citation False  accept an answer that cites no visible turn           (no_cite_req)
  check_binding    False  skip the speaker-binding check                         (no_binding)
  check_event      False  skip the holiday and cessation checks                  (no_event)
  gate_always      True   repair every question, not only refusals/invalid ones  (gate_always)
  correction_text  False  second read gets no CORRECTION line                    (repair_expand_only)
  expand           False  second read uses the first bundle (+ correction)       (repair_retry_only)
  use_search       False  expansion ignores the reader's proposed searches       (repair_no_search)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .bridge import _NO_INFO
from .hybrid import CareTwoStage
from .sourcefirst import SourceTurn, TextBackend, _json_object


@dataclass(frozen=True)
class AblationFlags:
    require_citation: bool = True
    check_binding: bool = True
    check_event: bool = True
    gate_always: bool = False
    correction_text: bool = True
    expand: bool = True
    use_search: bool = True

    @property
    def changes_validation(self) -> bool:
        return not (self.require_citation and self.check_binding and self.check_event)


ARMS: dict[str, AblationFlags] = {
    "care": AblationFlags(),
    "no_cite_req": AblationFlags(require_citation=False),
    "no_binding": AblationFlags(check_binding=False),
    "no_event": AblationFlags(check_event=False),
    "no_checks": AblationFlags(check_binding=False, check_event=False),
    "gate_always": AblationFlags(gate_always=True),
    "repair_expand_only": AblationFlags(correction_text=False),
    "repair_retry_only": AblationFlags(expand=False),
    "repair_no_search": AblationFlags(use_search=False),
}
ABLATION_ARMS = tuple(a for a in ARMS if a != "care")


def flags_as_dict() -> dict[str, dict[str, bool]]:
    return {name: asdict(flags) for name, flags in ARMS.items()}


class _Capture:
    """Forwards to the real backend and remembers the last raw reply."""

    def __init__(self, backend: TextBackend):
        self.backend, self.raw = backend, ""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.raw = self.backend.generate(prompt, **kwargs)
        return self.raw


class CareAblation(CareTwoStage):
    _force_revalidate = False        # tests set this to prove the re-validation path equals the parent's

    def __init__(self, turns: list[SourceTurn], *, flags: AblationFlags | None = None, **kwargs: Any):
        super().__init__(turns, **kwargs)
        self.flags = flags or AblationFlags()

    # -- reader ---------------------------------------------------------------------------
    def _reader(self, question: str, turns: list[SourceTurn], backend: TextBackend,
                *, correction: str = "") -> dict[str, Any]:
        capture = _Capture(backend)
        base = super()._reader(question, turns, capture, correction=correction)
        if not (self.flags.changes_validation or self._force_revalidate):
            return base
        return self._revalidate(question, base, capture.raw)

    def _revalidate(self, question: str, base: dict[str, Any], raw: str) -> dict[str, Any]:
        """Same parsing and checks as ``CareBridge._reader``, with the validation flags applied."""
        parsed = _json_object(raw)
        if parsed is None:
            return base
        answer = str(parsed.get("answer") or "").strip()
        visible_ids = set(base["bundle_ids"])
        proposed = parsed.get("source_ids")
        citations = [sid for sid in (proposed if isinstance(proposed, list) else [])
                     if isinstance(sid, str) and sid in visible_ids]
        searches = parsed.get("search_queries")
        queries = [q.strip() for q in (searches if isinstance(searches, list) else [])
                   if isinstance(q, str) and 3 <= len(q.strip()) <= 120][:2]
        names = self._subject_names(question)
        invalid = None
        if not answer:
            invalid = "empty_answer"
        elif _NO_INFO.casefold() not in answer.casefold():
            if not citations:
                if self.flags.require_citation:
                    invalid = "missing_citation"
            elif (self.flags.check_binding and len(names) == 1
                  and any(not self._directly_bound(self.by_id[sid], names[0]) for sid in citations)):
                invalid = "speaker_mismatch"
            elif self.flags.check_event:
                invalid = next((reason for sid in citations
                                if (reason := self._event_consistent(question, self.by_id[sid]))), None)
        if invalid:
            answer, citations = _NO_INFO, []
        return {**base, "answer": answer, "source_ids": list(dict.fromkeys(citations)),
                "search_queries": queries, "invalid_reason": invalid}

    # -- gate -----------------------------------------------------------------------------
    def first_pass(self, question: str, backend: TextBackend, initial: list[SourceTurn]) -> dict[str, Any]:
        fp = super().first_pass(question, backend, initial)
        if self.flags.gate_always:
            fp["needs_repair"] = True
        return fp

    # -- repair ---------------------------------------------------------------------------
    def expansion(self, question: str, first: dict[str, Any], mode: str = "lexical") -> list[SourceTurn]:
        if not self.flags.use_search:
            first = {**first, "search_queries": []}      # parent falls back to the question alone
        return super().expansion(question, first, mode)

    def repair(self, question: str, backend: TextBackend, fp: dict[str, Any],
               mode: str = "lexical") -> dict[str, Any]:
        first = fp["first"]
        if not fp["needs_repair"]:
            return super().repair(question, backend, fp, mode)
        second_turns = (self.expansion(question, first, mode) if self.flags.expand
                        else [self.by_id[i] for i in fp["initial_ids"]])
        correction = ""
        if self.flags.correction_text:
            correction = ("Previous evidence did not bind the named person to the cited first-person claim. "
                          "Check speaker labels strictly. " if first["invalid_reason"] == "speaker_mismatch"
                          else "Find direct evidence for the requested person, event and time. ")
        second = self._reader(question, second_turns, backend, correction=correction)
        return {**second, "candidate_ids": [turn.source_id for turn in second_turns],
                "risk_reasons": fp["risk"], "escalated": True,
                "first_pass": {key: first[key] for key in ("answer", "source_ids", "invalid_reason",
                                                         "bundle_ids", "search_queries")}}
