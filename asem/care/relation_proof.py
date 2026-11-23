"""Conservative query-slot checks over cited turns and nearby counterevidence.

The guard verifies coverage of a few high-confidence query constraints. Nearby
turns can expose a contradiction, but cannot silently supply a missing citation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .sourcefirst import SourceTurn


_PET_TYPES = {
    "cat": ("cat", "cats", "kitten", "kittens", "feline"),
    "dog": ("dog", "dogs", "puppy", "puppies", "pup", "pups", "canine"),
    "bird": ("bird", "birds", "parrot", "parrots"),
    "rabbit": ("rabbit", "rabbits", "bunny", "bunnies"),
}
_RELATIONS = {
    "mother": ("mom", "mother", "mum", "mama"),
    "father": ("dad", "father", "papa"),
    "husband": ("husband",),
    "wife": ("wife",),
    "sister": ("sister",),
    "brother": ("brother",),
    "friend": ("friend",),
}
_SEE = re.compile(r"\b(?:see|saw|seen|seeing|notice|noticed|spot|spotted)\b")
_ADOPT = re.compile(r"\b(?:adopt|adopted|adopting)\b")
_PLAY = re.compile(r"\b(?:play|played|playing|game|games)\b")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^\w]+", " ", text).strip()


def _has(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?:^| )" + re.escape(_norm(phrase)) + r"(?: |$)", _norm(text)))


def _group(text: str, groups: dict[str, tuple[str, ...]]) -> str | None:
    return next((key for key, words in groups.items()
                 if any(_has(text, word) for word in words)), None)


def _unique_group(text: str, groups: dict[str, tuple[str, ...]]) -> str | None:
    found = [key for key, words in groups.items()
             if any(_has(text, word) for word in words)]
    return found[0] if len(found) == 1 else None


def _clauses(text: str) -> list[str]:
    return [part for part in re.split(r"[.!?;]", text) if part.strip()]


@dataclass(frozen=True)
class QuerySlot:
    kind: str
    value: str
    bind_answer: bool = False


@dataclass(frozen=True)
class QueryFrame:
    subject: str
    slots: tuple[QuerySlot, ...]


@dataclass(frozen=True)
class ProofResult:
    status: str  # covered, incomplete, contradicted, unknown
    missing: tuple[str, ...] = ()
    counterevidence_ids: tuple[str, ...] = ()


def compile_query(question: str, known_speakers: set[str]) -> QueryFrame | None:
    q = _norm(question)
    names = [name for name in known_speakers if _has(q, name)]
    if len(names) != 1:
        return None
    subject = names[0]
    slots: list[QuerySlot] = []
    if _ADOPT.search(q):
        pet = _group(q, _PET_TYPES)
        if pet:
            slots.append(QuerySlot("adopted_pet_type", pet,
                                   bind_answer=bool(re.search(r"\bname\b", q))))
    if _SEE.search(q):
        place = re.search(r"\bat (?:the |a )?([\w-]+)\b", q)
        if place:
            slots.append(QuerySlot("observed_at", place.group(1)))
    if _PLAY.search(q):
        companion = re.search(r"\bwith (?:her|his|their|my) ([\w-]+)\b", q)
        if companion:
            relation = _group(companion.group(1), _RELATIONS)
            if relation:
                slots.append(QuerySlot("played_with", relation,
                                       bind_answer=bool(re.search(r"\bgames?\b", q))))
        platform = re.search(r"\bon (?:the |a )?([\w-]+) (?:game system|console|platform)\b", q)
        if platform:
            slots.append(QuerySlot("played_on", platform.group(1),
                                   bind_answer=bool(re.search(r"\bgames?\b", q))))
    return QueryFrame(subject, tuple(slots)) if slots else None


class RelationProofGuard:
    """Check cited query slots, using same-session neighbors for counterevidence."""

    def __init__(self, turns: list[SourceTurn]):
        self.speakers = {turn.speaker.split()[0].casefold() for turn in turns
                         if turn.speaker.strip()}
        self.by_position = {(turn.session_id, turn.ordinal): turn for turn in turns}

    def _neighbors(self, cited: list[SourceTurn]) -> list[SourceTurn]:
        nearby = []
        for source in cited:
            for distance in (1, 2):
                turn = self.by_position.get((source.session_id, source.ordinal - distance))
                if turn is not None and turn.speaker.casefold() == source.speaker.casefold():
                    nearby.append(turn)
        return nearby

    def check(self, question: str, answer: str,
              cited: list[SourceTurn]) -> ProofResult:
        frame = compile_query(question, self.speakers)
        if frame is None or not cited or not answer.strip():
            return ProofResult("unknown")
        nearby = self._neighbors(cited)
        missing: list[str] = []
        counter: list[str] = []
        for slot in frame.slots:
            if slot.kind == "adopted_pet_type":
                accepted = _PET_TYPES[slot.value]
                if not any(_ADOPT.search(_norm(clause))
                           and any(_has(clause, word) for word in accepted)
                           and (not slot.bind_answer or _has(clause, answer))
                           for turn in cited for clause in _clauses(turn.text)):
                    missing.append(slot.kind)
                    # A nearby pet type is counterevidence only when the cited
                    # naming/adoption turn has a deictic link to that pet.
                    deictic = any(_ADOPT.search(_norm(turn.text))
                                  and re.search(r"\b(?:he|him|his|she|her|it|its)\b",
                                                _norm(turn.text)) for turn in cited)
                    if deictic:
                        counter.extend(turn.source_id for turn in nearby
                                       if _unique_group(turn.text, _PET_TYPES)
                                       not in (None, slot.value))
            elif slot.kind == "observed_at":
                target = frame.subject
                def explicit_observation(turn: SourceTurn) -> bool:
                    for clause in _clauses(turn.text):
                        text = _norm(clause)
                        if not _has(text, slot.value) or not _SEE.search(text):
                            continue
                        first_person = (turn.speaker.split()[0].casefold() == target
                                        and re.search(r"\bi (?:saw|see|noticed|spotted)\b", text))
                        named = re.search(r"\b" + re.escape(target)
                                          + r"\b.{0,50}\b(?:saw|see|noticed|spotted)\b", text)
                        if first_person or named:
                            return True
                    return False
                if not any(explicit_observation(turn) for turn in cited):
                    missing.append(slot.kind)
            elif slot.kind == "played_with":
                accepted = _RELATIONS[slot.value]
                if not any(_PLAY.search(_norm(clause))
                           and any(_has(clause, word) for word in accepted)
                           and (not slot.bind_answer or _has(clause, answer))
                           for turn in cited for clause in _clauses(turn.text)):
                    missing.append(slot.kind)
                    if any(re.search(r"\bwe\b", _norm(turn.text)) and _PLAY.search(
                            _norm(turn.text)) for turn in cited):
                        counter.extend(turn.source_id for turn in nearby
                                       if _PLAY.search(_norm(turn.text))
                                       and _unique_group(turn.text, _RELATIONS)
                                       not in (None, slot.value))
            elif slot.kind == "played_on":
                if not any(_PLAY.search(_norm(clause)) and _has(clause, slot.value)
                           and (not slot.bind_answer or _has(clause, answer))
                           for turn in cited for clause in _clauses(turn.text)):
                    missing.append(slot.kind)
        if counter:
            return ProofResult("contradicted", tuple(missing), tuple(dict.fromkeys(counter)))
        if missing:
            return ProofResult("incomplete", tuple(missing))
        return ProofResult("covered")
