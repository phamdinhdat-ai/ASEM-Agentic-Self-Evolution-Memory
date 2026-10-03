"""Single-Pass Session Ingestor (ASEM-THG).

Performs single-pass LLM extraction per session, transforming dialogue turns
into structured atomic facts with triplets, then deterministically populates
the MemoryBank and TemporalHyperGraph in <1ms without extra LLM linking calls.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backends.base import InferenceBackend
from .hyper_graph import TemporalHyperGraph, Triplet
from .llm_validator import LLMRetryHandler, validate_fact_array
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import (
    LinkRecord,
    Note,
    TripletRecord,
    _try_extract_json,
    cap_description,
    cap_keywords,
)
from .temporal import extract_session_header, parse_session_datetime

_log = get_logger("THG.ingest")

# Applied when the extractor returns no tags, so the tag channel is never flat
# across the bank (ASEM v1 / FastASEM both populate G).
_DEFAULT_TAG = "fact"

_SINGLE_PASS_PROMPT_TEMPLATE = """You are an expert memory extraction agent.
Convert ONE dialogue session into standalone, atomic factual notes that become
nodes in a temporal knowledge graph.

SESSION DATE: {session_date}

CRITICAL RULES
1. RESOLVE PRONOUNS: replace every pronoun with the person's name ("I went there" -> "Caroline went to Hawaii").
2. RESOLVE EVERY DATE TO AN ABSOLUTE CALENDAR DATE. This is the single most
   important rule: the memory is queried later with questions that SUBTRACT two
   dates ("how many days between X and Y"), so a fact that keeps a relative or
   shorthand date is unusable.
   a) NORMALISE dates the speaker already stated. Write the full month name and
      the year, never a bare numeric shorthand:
        - "on 3/1"        -> "on 1 March 2022"
        - "on 2/15"       -> "on 15 February 2022"
        - "January 10th"  -> "on 10 January 2023"
        - "on Feb 26th"   -> "on 26 February 2023"
      Infer the YEAR from the session date when the speaker omits it.
   b) RESOLVE relative wording against the session date:
        - "yesterday" -> the day before the session date
        - "last week" / "a few days ago" -> the matching earlier date in the same month
        - "last Saturday" -> the most recent Saturday before the session date
        - "next week" / "next month" -> the following week / month
        - "last year" -> the previous calendar year
   c) ONE DATE PER FACT. If a single turn states or implies TWO or more dated
      events, emit a SEPARATE note for each with its own absolute date. Never
      merge two differently-dated events into one sentence, and never put two
      competing dates in one fact — the reader must be able to pair each event
      with exactly one date.
      Example — one turn saying "I started working with her on 2/15 ... I saw a
      house I love on 3/1" becomes TWO notes:
        {{"fact": "The user started working with realtor Rachel on 15 February 2022.", ...}}
        {{"fact": "The user saw a house they love on 1 March 2022.", ...}}
      NOT one note containing both dates.
   d) State the date INSIDE the fact sentence (e.g. "On 6 May 2023, ...") so it
      survives without external context.
   e) If a fact has no determinable date, write it without a date. NEVER invent
      a date, and never copy the session date onto an event that did not happen
      on it.
3. ONE FACT PER NOTE: emit a separate object for EVERY distinct event, activity,
   plan, intention, preference, relationship and status.
   - Do NOT merge several events into one thematic summary.
   - Do NOT drop a fact because it sounds minor.
   - A past event and a future plan are TWO different notes.
4. ATOMIC TRIPLET: give the (subject, predicate, object) the fact is about, so the
   graph can version it deterministically. subject = main person/entity,
   predicate = verb/status/relationship, object = target entity or value.
   For a DATED EVENT, put the date in the fact sentence (rule 2d) and keep the
   object the entity/value the predicate points at (not the date).
5. REACTIVE / ATTRIBUTION TURNS: when a speaker REACTS to or MENTIONS a fact
   about the OTHER person, emit a note that captures the attribution. Example:
   - Turn: "That's gorgeous, Caroline! It's awesome what items can mean so much to us, right? Got
     that from my grandma in Sweden."
   - Emit: {{"fact": "Melanie reacted to Caroline's necklace, saying it symbolizes love and
     strength; Melanie's own grandmother gave her a necklace from Sweden.",
     "subject": "Melanie", "predicate": "reacted_to", "object": "Caroline's necklace",
     "speaker": "Melanie"}}
   - This is CRITICAL for misattribution questions: the listener's reaction is where
     the "who said what" signal lives. Do NOT skip reactive turns.

DIALOGUE:
{dialogue}

Return a JSON array of objects with this schema (one object per fact):
[
  {{
    "fact": "One standalone factual sentence, max 45 words, with the absolute date where applicable",
    "subject": "Main person/entity",
    "predicate": "verb / status / relationship",
    "object": "Target entity or value",
    "entities": ["Person", "Place", "Organisation"],
    "keywords": ["keyword1", "keyword2"],
    "tags": ["personal|professional|event|preference|plan|relationship|location|health|travel|pet|temporal|fact"],
    "speaker": "Name of the speaker of the turn the fact came from"
  }}
]
Output ONLY the JSON array (no markdown fences, no commentary)."""


def build_fact_array_schema() -> Dict[str, Any]:
    """JSON Schema for the single-pass extraction response.

    Hand this to the server as guided decoding (``response_format``), and the
    backend physically cannot emit a malformed array: no stray prose around it,
    no trailing comma, no single quotes. Worth enabling because the fallback
    path (one note per dialogue line) throws away every triplet, so the graph
    loses its ``same-entity`` / ``superseded_by`` edges.
    """
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "fact": {"type": "string"},
                "subject": {"type": "string"},
                "predicate": {"type": "string"},
                "object": {"type": "string"},
                "entities": {"type": "array", "items": {"type": "string"}},
                "keywords": {"type": "array", "items": {"type": "string"}},
                "tags": {"type": "array", "items": {"type": "string"}},
                "speaker": {"type": "string"},
            },
            "required": ["fact"],
        },
    }


# Config value for `ingestion.response_format`: the OpenAI-compatible
# structured-output form, using this stage's own schema.
def json_schema_response_format() -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "atomic_facts",
            "schema": build_fact_array_schema(),
        },
    }



class SinglePassSessionIngestor:
    """Ingest multi-turn dialogue sessions in 1 single LLM call."""

    def __init__(
        self,
        backend: InferenceBackend,
        hyper_graph: Optional[TemporalHyperGraph] = None,
        q0: float = 0.5,
        max_retries: int = 0,
        response_format: Optional[Any] = None,
        temperature: Optional[float] = None,
    ) -> None:
        self.backend = backend
        # ``hyper_graph or TemporalHyperGraph()`` was WRONG and silently discarded
        # the caller's graph whenever it was EMPTY. ``TemporalHyperGraph`` defines
        # ``__len__`` (returns the fact-node count), so a freshly-constructed graph
        # is FALSY, and the ``or`` fell through to a brand-new default graph. The
        # effect was that ``enable_supersession`` (and any other constructor option)
        # set by the caller never reached the ingestor: the retriever held the
        # configured graph while the ingestor wrote into an unconfigured one, so
        # GB2e's disable-by-default suppressed the relation on the READ side only.
        #   Measured on configs/models/thg_v2_postfix_off.yaml (supersession OFF):
        #     retriever.hyper_graph.enable_supersession  = False
        #     ingestor.hyper_graph.enable_supersession   = True    <- default, ignored
        #     retriever.hyper_graph is ingestor.hyper_graph -> False
        # and the resulting bank carried 242 ``superseded_by`` records instead of 0.
        # An explicit ``is None`` test keeps "no graph supplied" working while never
        # second-guessing a graph the caller actually passed.
        self.hyper_graph = hyper_graph if hyper_graph is not None else TemporalHyperGraph()
        self.q0 = q0
        self.max_retries = max_retries
        # Structured output: `"json_schema"` (this stage's own schema) or any
        # raw `response_format` dict the provider understands. `None` keeps the
        # plain prompt-only contract.
        self.response_format = response_format
        # Extraction temperature. None inherits the backend's config value. Set
        # this for reproducible graph builds: fact/entity extraction at a nonzero
        # temperature produces a different graph on every ingest, so a bank built
        # twice is not comparable.
        self.temperature = temperature
        # Observability: how the last ingest session actually went.
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_model(self, prompt: str) -> str:
        """One extraction call, with the configured structured-output mode."""
        kwargs: Dict[str, Any] = {}
        if self.response_format:
            kwargs["response_format"] = self.response_format
        if self.temperature is not None:
            kwargs["temperature"] = float(self.temperature)
        if not kwargs:
            return self.backend.generate(prompt)
        try:
            return self.backend.generate(prompt, **kwargs)
        except TypeError as exc:
            # A backend that does not accept these kwargs falls back to defaults.
            if "unexpected keyword" in str(exc) or "positional argument" in str(exc):
                _log.debug("Ingest backend ignores per-call kwargs: {}", exc)
                if self.response_format:
                    return self.backend.generate(
                        prompt, response_format=self.response_format
                    )
                return self.backend.generate(prompt)
            raise

    def _extract_facts(
        self, prompt: str, dialogue_turns: List[str]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Generate + parse the fact list, with retry and truncation awareness.

        Returns ``(facts, source)`` where ``source`` is ``"llm"``,
        ``"llm_retry"`` (recovered by the format-correction retry) or
        ``"fallback"``.

        The retry is wired here because ``max_retries`` used to be stored and
        never used: a malformed response went straight to the per-turn
        fallback, even with ``llm_retry.max_retries`` set in the config.
        """
        stats: Dict[str, Any] = {
            "attempts": 0,
            "raw_len": 0,
            "finish_reason": None,
            "truncated": False,
        }

        def _generate(attempt_prompt: str) -> str:
            raw = self._call_model(attempt_prompt)
            stats["attempts"] += 1
            stats["raw_len"] = len(raw or "")
            stats["finish_reason"] = getattr(
                self.backend, "last_finish_reason", None
            )
            return raw

        handler = LLMRetryHandler(_generate, max_retries=max(0, self.max_retries))
        extracted, attempt = handler.invoke(
            prompt,
            parse_fn=lambda raw: _try_extract_json(raw, expect_array=True),
            validate_fn=validate_fact_array,
        )

        stats["truncated"] = stats["finish_reason"] == "length"
        self.last_stats = stats
        source = "llm" if attempt == 0 else "llm_retry"

        if not isinstance(extracted, list) or not extracted:
            truncated = stats["truncated"]
            _log.warning(
                "SinglePassSessionIngestor | JSON extraction fallback triggered "
                "(raw_len={} parsed={} finish_reason={} attempts={}){}",
                stats["raw_len"],
                type(extracted).__name__,
                stats["finish_reason"] or "?",
                stats["attempts"],
                (
                    " — the completion hit the output/context budget "
                    "(finish_reason='length'), so the JSON was never closed. "
                    "Raise the served --max-model-len and/or the ingest "
                    "max_tokens; prompt wording cannot fix this."
                    if truncated
                    else ""
                ),
            )
            extracted = self._fallback_extract(dialogue_turns)
            self.last_stats = {**stats, "source": "fallback"}
            return extracted, "fallback"

        self.last_stats = {**stats, "source": source, "facts": len(extracted)}
        return extracted, source

    def ingest_session(
        self,
        dialogue_turns: List[str],
        memory_bank: MemoryBank,
        session_date: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> List[Note]:
        """Ingest conversation session into MemoryBank and HyperGraph in 1 LLM call."""
        if not dialogue_turns:
            return []

        # 1. Resolve session date & ID
        if not session_date:
            for turn in dialogue_turns[:2]:
                h_num, h_date = extract_session_header(turn)
                if h_date:
                    session_date = h_date
                    if h_num and not session_id:
                        session_id = f"session_{h_num}"
                    break

        dt_obj, iso_str = parse_session_datetime(session_date)

        # 2. Build single-pass prompt & call LLM (1 call)
        dialogue_text = "\n".join(dialogue_turns)

        # The extractor is asked to "resolve relative time to absolute dates
        # USING THE SESSION DATE", so it needs a date it can actually reason
        # about. The raw corpus stamp is machine-formatted and sometimes carries
        # a weekday — "2022/03/02 (Wed) 22:25" — and the instruction used to
        # receive exactly that. When the central parser can resolve it, hand over
        # the human form ("2 March 2022") and keep the raw string only as a
        # parenthetical, so "yesterday"/"last Saturday" arithmetics have a clear
        # anchor. When parsing failed the date is UNKNOWN, not "whatever string
        # was passed" — saying so stops the model from inventing one.
        resolved_date: str
        if iso_str is None:
            resolved_date = "UNKNOWN (no reliable session date was supplied)"
        else:
            resolved_date = dt_obj.strftime("%d %B %Y").lstrip("0")
            if session_date and str(session_date).strip() not in resolved_date:
                resolved_date = f"{resolved_date} (raw stamp: {session_date})"

        prompt = _SINGLE_PASS_PROMPT_TEMPLATE.format(
            session_date=resolved_date,
            dialogue=dialogue_text,
        )

        extracted, extract_source = self._extract_facts(prompt, dialogue_turns)

        # 3. Create Notes & Index into HyperGraph
        created_notes: List[Note] = []
        note_ids: List[str] = []

        for item in extracted:
            if not isinstance(item, dict):
                continue

            # Length caps keep a verbose extraction from bloating the bank (and
            # the answer prompt that renders it); ASEM v1 / FastASEM apply them too.
            fact_text = cap_description(
                str(item.get("fact") or item.get("content") or item.get("description") or "").strip()
            )
            if not fact_text:
                continue

            subj = str(item.get("subject") or "").strip()
            pred = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            # `or []` (rather than the `.get(k, [])` default) is load-bearing: the
            # extractor emits an explicit `"entities": null` often enough that the
            # default would let `None` through and raise on iteration.
            entities = cap_keywords(
                [str(e).strip() for e in (item.get("entities") or []) if str(e).strip()]
            )
            keywords = cap_keywords(
                [str(k).strip() for k in (item.get("keywords") or []) if str(k).strip()]
            )
            tags = [str(t).strip() for t in (item.get("tags") or []) if str(t).strip()]
            if not tags:
                tags = [_DEFAULT_TAG]
            speaker = str(item.get("speaker") or "").strip() or None

            # Joint embedding, same shape as ASEM v1 / FastASEM:
            # z = the atomic fact, e = fact + keywords + tags + entities.
            e_vec = self.backend.embed(
                " ".join([fact_text, " ".join(keywords), " ".join(tags), " ".join(entities)])
            )
            z_vec = self.backend.embed(fact_text)

            note = Note(
                id=str(uuid.uuid4()),
                c=fact_text,
                t=dt_obj,
                K=keywords,
                G=tags,
                X=fact_text,
                e=e_vec,
                L=[],
                z=z_vec,
                q=self.q0,
                session_id=session_id,
                session_date=session_date,
                timestamp_iso=iso_str,
                entities=entities,
                speaker=speaker,
            )

            # Deterministic hyper-graph links (zero LLM): same-entity peers,
            # semantic neighbours and -- through the triplet -- temporal versioning.
            triplet = Triplet(subject=subj, predicate=pred, object=obj, timestamp=iso_str or "")
            # Persist the triplet ON the note as well as passing it to the graph.
            # The in-memory versioning indices do not survive the process
            # boundary between ingest and evaluation, so without this a note that
            # already exists in the bank can never be superseded by a freshly
            # ingested one (measured: 0 superseded_by edges on the LongMemEval
            # bank). Only a usable triplet is recorded, mirroring add_note's own
            # subject/predicate guard, so the note gains no empty record.
            if triplet.subject and triplet.predicate:
                note.triplets.append(
                    TripletRecord(
                        subject=triplet.subject,
                        predicate=triplet.predicate,
                        object=triplet.object,
                        recorded_at=triplet.timestamp or None,
                    )
                )
            note.L.extend(self.hyper_graph.add_note(note, triplet))

            created_notes.append(note)
            note_ids.append(note.id)

        # 4. Chronological edges between consecutive notes of the session
        self.hyper_graph.add_temporal_sequence(note_ids)

        # 5. One transaction: the new notes, plus every pre-existing peer whose
        #    link set the graph just extended.
        self._persist(created_notes, memory_bank)

        _log.info(
            "Single-pass ingest done | turns={} facts={} bank_size={} source={}",
            len(dialogue_turns), len(created_notes), memory_bank.size(),
            extract_source,
        )
        return created_notes

    def _persist(self, created_notes: List[Note], memory_bank: MemoryBank) -> None:
        """Persist the new notes plus every pre-existing peer the graph touched.

        The hyper-graph writes mirror edges onto peer notes in place. Peers from
        earlier sessions exist only in the bank, so they have to be written back
        for the new edge to survive into the evaluation phase (which runs in a
        separate process and only ever sees the stored ``Note.L``).
        """
        if not created_notes:
            return
        by_id = {note.id: note for note in created_notes}
        extra = [n for n in self.hyper_graph.pop_touched_peers() if n.id not in by_id]
        memory_bank.add_many(created_notes + extra)
    @staticmethod
    def _fallback_extract(turns: List[str]) -> List[Dict[str, Any]]:
        """Last-resort extraction: one note per dialogue line.

        Strips the ``[Speaker]`` prefix off the fact text (the bank stores the
        speaker in its own field) so a fallback note is still readable evidence.
        """
        results: List[Dict[str, Any]] = []
        for line in turns:
            line = line.strip()
            if len(line) < 10 or line.startswith("[Session"):
                continue
            match = re.match(r"^\[(.*?)\]\s*(.*)", line)
            speaker, content = (match.group(1), match.group(2)) if match else ("", line)
            content = content.strip()
            if not content:
                continue
            results.append({
                "fact": content,
                "subject": speaker,
                "predicate": "",
                "object": "",
                "entities": [speaker] if speaker else [],
                "keywords": re.findall(r"\w{4,}", content.lower())[:4],
                "tags": [_DEFAULT_TAG],
                "speaker": speaker,
            })
        return results

