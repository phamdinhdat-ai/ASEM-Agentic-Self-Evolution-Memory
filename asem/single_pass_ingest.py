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
from .note import LinkRecord, Note, _try_extract_json
from .temporal import extract_session_header, parse_session_datetime

_log = get_logger("THG.ingest")

_SINGLE_PASS_PROMPT_TEMPLATE = """You are an expert memory extraction agent.
Your task is to extract clear, standalone, atomic factual notes from a dialogue session.

SESSION DATE: {session_date}

CRITICAL RULES:
1. RESOLVE PRONOUNS: Replace every pronoun with the speaker's name or full person name.
2. RESOLVE RELATIVE TIME TO ABSOLUTE DATES: Convert relative time ("yesterday", "last week") into absolute dates using the session date.
3. EXTRACT ATOMIC TRIPLETS: Provide subject, predicate, and object for precise graph versioning.
4. EXTRACT NAMED ENTITIES AND KEYWORDS.

Return a JSON array of objects with the following schema:
[
  {{
    "fact": "Standalone factual sentence with absolute date where applicable",
    "subject": "Main Person/Entity",
    "predicate": "Action/Status/Relationship",
    "object": "Target Entity/Value",
    "entities": ["Entity1", "Entity2"],
    "keywords": ["keyword1", "keyword2"],
    "speaker": "SpeakerName"
  }}
]
Output ONLY valid JSON array."""


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
        if not isinstance(extracted, list):
            _log.warning("SinglePassSessionIngestor | JSON extraction fallback triggered")
            extracted = self._fallback_extract(dialogue_turns)

        # 3. Create Notes & Index into HyperGraph
        created_notes: List[Note] = []
        note_ids: List[str] = []

        for item in extracted:
            if not isinstance(item, dict):
                continue
            fact_text = str(item.get("fact", item.get("content", ""))).strip()
            if not fact_text:
                continue

            subj = str(item.get("subject", "")).strip()
            pred = str(item.get("predicate", "")).strip()
            obj = str(item.get("object", "")).strip()
            entities = [str(e).strip() for e in item.get("entities", []) if str(e).strip()]
            keywords = [str(k).strip() for k in item.get("keywords", []) if str(k).strip()]
            speaker = str(item.get("speaker", "")).strip() or None

            # Embeddings (z = raw fact, e = fact + keywords)
            e_text = f"{fact_text} {' '.join(keywords)} {' '.join(entities)}"
            e_vec = self.backend.embed(e_text)
            z_vec = self.backend.embed(fact_text)

            note = Note(
                id=str(uuid.uuid4()),
                c=fact_text,
                t=dt_obj,
                K=keywords,
                G=["atomic_fact"],
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

            # Deterministic HyperGraph links
            triplet = Triplet(subject=subj, predicate=pred, object=obj, timestamp=iso_str or "")
            auto_links = self.hyper_graph.add_note(note, triplet)
            note.L.extend(auto_links)

            created_notes.append(note)
            note_ids.append(note.id)

        # 4. Add Temporal Sequence Edges
        self.hyper_graph.add_temporal_sequence(note_ids)

        # 5. Save to MemoryBank in single transaction
        memory_bank.add_many(created_notes)

        _log.info(
            "Single-pass ingest done | turns={} facts={} bank_size={}",
            len(dialogue_turns), len(created_notes), memory_bank.size()
        )
        return created_notes

    @staticmethod
    def _fallback_extract(turns: List[str]) -> List[Dict[str, Any]]:
        results = []
        for line in turns:
            line = line.strip()
            if line and len(line) >= 10:
                results.append({"fact": line, "keywords": [], "entities": []})
        return results

