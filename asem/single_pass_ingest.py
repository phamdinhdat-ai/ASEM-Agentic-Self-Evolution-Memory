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
from .llm_validator import LLMRetryHandler
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .note import (
    LinkRecord,
    Note,
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
2. RESOLVE RELATIVE TIME TO ABSOLUTE DATES using the session date:
   - "yesterday" -> the day before the session date
   - "last week" / "a few days ago" -> the matching earlier date in the same month
   - "last Saturday" -> the most recent Saturday before the session date
   - "next week" / "next month" -> the following week / month
   - "last year" -> the previous calendar year
   State the resolved absolute date inside the fact (e.g. "On 6 May 2023, ...").
3. ONE FACT PER NOTE: emit a separate object for EVERY distinct event, activity,
   plan, intention, preference, relationship and status.
   - Do NOT merge several events into one thematic summary.
   - Do NOT drop a fact because it sounds minor.
   - A past event and a future plan are TWO different notes.
4. ATOMIC TRIPLET: give the (subject, predicate, object) the fact is about, so the
   graph can version it deterministically. subject = main person/entity,
   predicate = verb/status/relationship, object = target entity or value.

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


class SinglePassSessionIngestor:
    """Ingest multi-turn dialogue sessions in 1 single LLM call."""

    def __init__(
        self,
        backend: InferenceBackend,
        hyper_graph: Optional[TemporalHyperGraph] = None,
        q0: float = 0.5,
        max_retries: int = 0,
    ) -> None:
        self.backend = backend
        self.hyper_graph = hyper_graph or TemporalHyperGraph()
        self.q0 = q0
        self.max_retries = max_retries

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
        prompt = _SINGLE_PASS_PROMPT_TEMPLATE.format(
            session_date=session_date or (dt_obj.strftime("%d %B %Y") if dt_obj else "Unknown"),
            dialogue=dialogue_text,
        )

        raw = self.backend.generate(prompt)
        extracted = _try_extract_json(raw, expect_array=True)
        if not isinstance(extracted, list) or not extracted:
            _log.warning(
                "SinglePassSessionIngestor | JSON extraction fallback triggered "
                "(raw_len={} parsed={})",
                len(raw or ""),
                type(extracted).__name__,
            )
            extracted = self._fallback_extract(dialogue_turns)

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
            note.L.extend(self.hyper_graph.add_note(note, triplet))

            created_notes.append(note)
            note_ids.append(note.id)

        # 4. Chronological edges between consecutive notes of the session
        self.hyper_graph.add_temporal_sequence(note_ids)

        # 5. One transaction: the new notes, plus every pre-existing peer whose
        #    link set the graph just extended.
        self._persist(created_notes, memory_bank)

        _log.info(
            "Single-pass ingest done | turns={} facts={} bank_size={}",
            len(dialogue_turns), len(created_notes), memory_bank.size()
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

