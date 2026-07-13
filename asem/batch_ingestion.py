"""Batch session ingestion for ASEM memory bank.

Instead of processing dialogue turns one-by-one (3+ LLM calls per turn),
this module ingests an entire multi-turn conversation session in ~3 LLM calls:

1. **Batch Note Extraction** (1 LLM call): Extract ALL atomic facts from the
   full dialogue as structured notes.
2. **Batch Memory Operations** (1 LLM call): Decide ADD/UPDATE/DELETE/NOOP
   for ALL extracted notes at once.
3. **Batch Link Generation** (1 LLM call): Identify ALL pairwise relationships
   between new notes and existing neighbors.

This reduces ingestion from O(turns) to O(1) LLM calls per session.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backends.base import InferenceBackend
from .logging_utils import get_logger
from .memory_bank import MemoryBank
from .memory_manager import Op
from .note import Note

_log = get_logger("batch_ingest")


class BatchIngestor:
    """Ingest an entire multi-turn session dialogue in batch.

    Reuses the existing inference backend.  Prompt templates control the
    extraction, operation selection, and link generation behaviour.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        extraction_prompt: str,
        memory_ops_prompt: str,
        link_prompt: str,
        q0: float = 0.5,
        top_k_neighbors: int = 10,
    ) -> None:
        self._backend = backend
        self._extraction_prompt = extraction_prompt
        self._memory_ops_prompt = memory_ops_prompt
        self._link_prompt = link_prompt
        self._q0 = q0
        self._top_k_neighbors = top_k_neighbors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_conversation(
        self,
        dialogue_turns: List[str],
        memory_bank: MemoryBank,
    ) -> List[Note]:
        """Ingest all dialogue turns from a conversation into the memory bank.

        Args:
            dialogue_turns: Ordered list of formatted dialogue turn strings
                (e.g. ``"[Caroline] I went to the store."``).
            memory_bank: The memory bank to write into.

        Returns:
            List of newly created (or updated) Note objects.
        """
        if not dialogue_turns:
            _log.info("No dialogue turns to ingest")
            return []

        dialogue_text = "\n".join(dialogue_turns)
        _log.info(
            "Batch ingestion started | turns={}  chars={}",
            len(dialogue_turns),
            len(dialogue_text),
        )

        # Step 1 — Extract all notes from the dialogue
        extracted = self._extract_notes(dialogue_text)
        if not extracted:
            _log.warning("No notes extracted from dialogue")
            return []

        # Step 2 — Embed all extracted notes
        raw_notes = self._embed_notes(extracted)
        _log.info("Extracted {} raw notes", len(raw_notes))

        # Step 3 — Batch memory operations
        ops = self._batch_memory_ops(raw_notes, memory_bank)
        _log.info("Memory ops decided | adds={}  updates={}  deletes={}  noops={}",
                  sum(1 for o in ops if o["op"] == "ADD"),
                  sum(1 for o in ops if o["op"] == "UPDATE"),
                  sum(1 for o in ops if o["op"] == "DELETE"),
                  sum(1 for o in ops if o["op"] == "NOOP"))

        # Step 4 — Execute operations (track which notes actually get added)
        added_notes = self._execute_ops(raw_notes, ops, memory_bank)

        # Step 5 — Batch link generation
        if added_notes:
            self._batch_link(added_notes, memory_bank)

        # Step 6 — Rebuild FAISS once
        if added_notes or any(o["op"] in ("UPDATE", "DELETE") for o in ops):
            memory_bank._rebuild_index()

        _log.success(
            "Batch ingestion complete | extracted={}  added={}  bank_size={}",
            len(raw_notes),
            len(added_notes),
            memory_bank.size(),
        )
        return added_notes

    # ------------------------------------------------------------------
    # Step 1 — Batch note extraction
    # ------------------------------------------------------------------

    def _extract_notes(self, dialogue_text: str) -> List[Dict[str, Any]]:
        """Extract all atomic facts from the full dialogue in one LLM call."""
        prompt = self._extraction_prompt.format(dialogue=dialogue_text)
        raw = self._backend.generate(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            _log.warning("Failed to parse batch extraction JSON — trying fallback")
            data = self._fallback_extract(dialogue_text)

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    def _fallback_extract(self, dialogue_text: str) -> List[Dict[str, Any]]:
        """Fallback: extract per-line when batch JSON parsing fails."""
        results: List[Dict[str, Any]] = []
        for line in dialogue_text.split("\n"):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            results.append({
                "content": line,
                "keywords": [],
                "tags": ["dialogue"],
                "description": line[:120],
            })
        return results

    # ------------------------------------------------------------------
    # Step 2 — Embed extracted notes
    # ------------------------------------------------------------------

    def _embed_notes(self, extracted: List[Dict[str, Any]]) -> List[Note]:
        """Create Note objects with embeddings for each extracted fact."""
        notes: List[Note] = []
        for item in extracted:
            c = str(item.get("content", ""))
            K = list(item.get("keywords", []))
            G = list(item.get("tags", []))
            X = str(item.get("description", ""))

            if not c.strip():
                continue

            # Joint embedding (matches NoteConstructor.build)
            e_vec = self._backend.embed(" ".join([c, " ".join(K), " ".join(G), X]))
            z_vec = self._backend.embed(c)

            note = Note(
                id=str(uuid.uuid4()),
                c=c,
                t=datetime.utcnow(),
                K=K,
                G=G,
                X=X,
                e=e_vec,
                L=[],
                z=z_vec,
                q=self._q0,
            )
            notes.append(note)
        return notes

    # ------------------------------------------------------------------
    # Step 3 — Batch memory operations
    # ------------------------------------------------------------------

    def _batch_memory_ops(
        self,
        new_notes: List[Note],
        memory_bank: MemoryBank,
    ) -> List[Dict[str, Any]]:
        """Decide ADD/UPDATE/DELETE/NOOP for all new notes in one LLM call."""
        existing = memory_bank.list_notes()

        # Build compact payloads
        new_payloads = [
            {"index": i, "id": n.id, "keywords": n.K, "tags": n.G, "description": n.X}
            for i, n in enumerate(new_notes)
        ]
        existing_payloads = [
            {"id": n.id, "keywords": n.K, "tags": n.G, "description": n.X}
            for n in existing[:20]  # cap at 20 for prompt size
        ]

        prompt = self._memory_ops_prompt.format(
            new_notes=json.dumps(new_payloads),
            existing_memory=json.dumps(existing_payloads) if existing_payloads else "[]",
        )
        raw = self._backend.generate(prompt)
        try:
            decisions = json.loads(raw)
        except json.JSONDecodeError:
            _log.warning("Failed to parse batch memory ops — defaulting to ADD all")
            return [{"index": i, "op": "ADD", "target_id": None}
                    for i in range(len(new_notes))]

        if not isinstance(decisions, list):
            return [{"index": i, "op": "ADD", "target_id": None}
                    for i in range(len(new_notes))]

        # Build complete decision list, filling gaps
        decided = {d.get("index", -1): d for d in decisions if isinstance(d, dict)}
        result = []
        for i in range(len(new_notes)):
            if i in decided:
                d = decided[i]
                result.append({
                    "index": i,
                    "op": str(d.get("op", "ADD")).upper(),
                    "target_id": d.get("target_id"),
                })
            else:
                result.append({"index": i, "op": "ADD", "target_id": None})
        return result

    # ------------------------------------------------------------------
    # Step 4 — Execute operations
    # ------------------------------------------------------------------

    def _execute_ops(
        self,
        new_notes: List[Note],
        ops: List[Dict[str, Any]],
        memory_bank: MemoryBank,
    ) -> List[Note]:
        """Execute batch memory operations and return notes that were added."""
        added: List[Note] = []
        for decision in ops:
            idx = decision["index"]
            if idx >= len(new_notes):
                continue
            note = new_notes[idx]
            op = decision["op"]
            target_id = decision.get("target_id")

            if op == "ADD":
                memory_bank.add(note)
                added.append(note)
            elif op == "UPDATE" and target_id:
                target = memory_bank.get_note(str(target_id))
                if target is not None:
                    merged = Note(
                        id=target.id,
                        c=note.c,
                        t=note.t,
                        K=note.K,
                        G=note.G,
                        X=note.X,
                        e=note.e,
                        L=target.L,
                        z=note.z,
                        q=target.q,
                    )
                    memory_bank.add(merged)
                    added.append(merged)
                else:
                    memory_bank.add(note)
                    added.append(note)
            elif op == "DELETE" and target_id:
                memory_bank.delete(str(target_id))
            elif op == "NOOP":
                pass
            else:
                # Unknown or invalid op — default to ADD
                memory_bank.add(note)
                added.append(note)
        return added

    # ------------------------------------------------------------------
    # Step 5 — Batch link generation
    # ------------------------------------------------------------------

    def _batch_link(
        self,
        added_notes: List[Note],
        memory_bank: MemoryBank,
    ) -> None:
        """Generate all pairwise links in one LLM call."""
        if not added_notes:
            return

        # Find the most similar existing notes for context
        all_existing = memory_bank.list_notes()
        all_existing_ids = {n.id for n in all_existing}

        # Get neighbors for each added note via ANN
        neighbor_set: set[str] = set()
        for note in added_notes:
            neighbors = memory_bank.ann_search(note.e, k=self._top_k_neighbors)
            for n in neighbors:
                if n.id not in {a.id for a in added_notes}:
                    neighbor_set.add(n.id)

        neighbor_notes = [n for n in all_existing if n.id in neighbor_set]

        # Build payloads
        new_payloads = [
            {"id": n.id, "keywords": n.K, "tags": n.G, "description": n.X}
            for n in added_notes
        ]
        neighbor_payloads = [
            {"id": n.id, "keywords": n.K, "tags": n.G, "description": n.X}
            for n in neighbor_notes[:20]  # cap for prompt
        ]

        prompt = self._link_prompt.format(
            new_notes=json.dumps(new_payloads),
            neighbors=json.dumps(neighbor_payloads) if neighbor_payloads else "[]",
        )
        raw = self._backend.generate(prompt)
        try:
            relations = json.loads(raw)
        except json.JSONDecodeError:
            _log.warning("Failed to parse batch link generation — no links created")
            return

        if not isinstance(relations, list):
            return

        # Apply bidirectional links
        all_note_map = {n.id: n for n in all_existing}
        for n in added_notes:
            all_note_map[n.id] = n

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            source = str(rel.get("source", ""))
            target = str(rel.get("target", ""))
            if source not in all_note_map or target not in all_note_map:
                continue
            if source == target:
                continue

            # Add bidirectional links (match LinkEvolver._apply_links behaviour)
            src_note = all_note_map[source]
            tgt_note = all_note_map[target]
            if target not in src_note.L:
                src_note.L.append(target)
            if source not in tgt_note.L:
                tgt_note.L.append(source)

            # Persist
            memory_bank.update(source, {"L": src_note.L})
            memory_bank.update(target, {"L": tgt_note.L})

        _log.info("Batch links created | relations={}  edges={}",
                  len(relations),
                  sum(1 for r in relations if isinstance(r, dict)))
