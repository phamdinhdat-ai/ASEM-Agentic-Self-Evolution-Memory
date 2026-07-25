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

import ast
import json
import re
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


def _extract_json(raw: str, expect_array: bool = True) -> Any:
    """Robust JSON extraction from LLM output.

    Handles common LLM artifacts: markdown fences, leading/trailing text,
    nested JSON in objects, and malformed quotes.

    Args:
        raw: Raw LLM output string.
        expect_array: If True, searches for ``[...]``; else expects ``{...}``.

    Returns:
        Parsed Python object, or None if parsing fails.
    """
    cleaned = raw.strip()

    # 1. Strip markdown fences
    #    Handles: ```json ... ```, ``` ... ```, and leading/trailing backticks
    fence_patterns = [
        (r"```json\s*", r"\s*```"),
        (r"```\s*", r"\s*```"),
    ]
    for open_pat, close_pat in fence_patterns:
        cleaned = re.sub(rf"^{open_pat}", "", cleaned)
        cleaned = re.sub(rf"{close_pat}$", "", cleaned)

    # 2. Find the outermost bracket pair
    open_br = "[" if expect_array else "{"
    close_br = "]" if expect_array else "}"

    start = cleaned.find(open_br)
    end = cleaned.rfind(close_br)
    if start >= 0 and end > start:
        cleaned = cleaned[start:end + 1]

    # 3. Try direct JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 4. Try ast.literal_eval (handles single-quoted JSON-like structures)
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        pass

    # 5. Try naive fix: replace single quotes with double quotes
    #    (only if the content looks like it uses single quotes)
    if expect_array and cleaned.count("'") > cleaned.count('"'):
        try:
            # Replace single quotes only outside of strings (heuristic)
            fixed = cleaned.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

    return None


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

        # --- Detailed note listing ---
        for i, n in enumerate(raw_notes):
            _log.debug(
                "  note[{}] | K=[{}]  G=[{}]  X={!r}",
                i,
                ", ".join(n.K[:5]) if n.K else "—",
                ", ".join(n.G[:3]) if n.G else "—",
                n.X[:100],
            )

        # Step 3 — Batch memory operations
        bank_size_before = memory_bank.size()
        ops = self._batch_memory_ops(raw_notes, memory_bank)
        n_add = sum(1 for o in ops if o["op"] == "ADD")
        n_update = sum(1 for o in ops if o["op"] == "UPDATE")
        n_delete = sum(1 for o in ops if o["op"] == "DELETE")
        n_noop = sum(1 for o in ops if o["op"] == "NOOP")
        _log.info("Memory ops | adds={}  updates={}  deletes={}  noops={}  "
                  "bank_before={}",
                  n_add, n_update, n_delete, n_noop, bank_size_before)

        # Step 4 — Execute operations (track which notes actually get added)
        added_notes = self._execute_ops(raw_notes, ops, memory_bank)

        # Step 5 — Batch link generation (cross-session aware)
        link_count = 0
        cross_session_links = 0
        if added_notes:
            link_count, cross_session_links = self._batch_link(added_notes, memory_bank)

        # Step 6 — Rebuild FAISS once (fast for small banks)
        if added_notes or n_update > 0 or n_delete > 0:
            memory_bank._rebuild_index()

        # --- Per-note summary ---
        for i, n in enumerate(raw_notes):
            op_label = ops[i]["op"] if i < len(ops) else "?"
            marker = ""
            if op_label == "ADD":
                marker = "+"
            elif op_label == "UPDATE":
                marker = "~"
            elif op_label == "DELETE":
                marker = "-"
            elif op_label == "NOOP":
                marker = "."
            kw_str = ", ".join(n.K[:4]) if n.K else "—"
            _log.info(
                "  {} [{}] {} | {} links | K=[{}]",
                marker, op_label, n.X[:80],
                len(n.L), kw_str,
            )

        _log.success(
            "Session ingest done | extracted={}  added={}  links={}  "
            "cross_session_links={}  bank: {} -> {}",
            len(raw_notes), len(added_notes),
            link_count, cross_session_links,
            bank_size_before, memory_bank.size(),
        )
        return added_notes

    # ------------------------------------------------------------------
    # Step 1 — Batch note extraction
    # ------------------------------------------------------------------

    def _extract_notes(self, dialogue_text: str) -> List[Dict[str, Any]]:
        """Extract all atomic facts from the full dialogue in one LLM call."""
        prompt = self._extraction_prompt.format(dialogue=dialogue_text)
        raw = self._backend.generate(prompt)
        data = _extract_json(raw, expect_array=True)

        if isinstance(data, list):
            filtered = [item for item in data if isinstance(item, dict)]
            if filtered:
                return filtered

        # If the LLM returned a dict with a key containing the array, try that
        if isinstance(data, dict):
            for key in ("notes", "facts", "results", "data", "items"):
                if isinstance(data.get(key), list):
                    return [item for item in data[key] if isinstance(item, dict)]

        _log.warning(
            "Failed to parse batch extraction JSON — trying fallback\n"
            "  raw[:500] = {!r}",
            raw[:500],
        )
        return self._fallback_extract(dialogue_text)

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
        decisions = _extract_json(raw, expect_array=True)

        if isinstance(decisions, list) and decisions:
            # Normalize: ensure each decision has "index", "op", "target_id"
            normalized = []
            for d in decisions:
                if isinstance(d, dict):
                    normalized.append({
                        "index": d.get("index", len(normalized)),
                        "op": str(d.get("op", "ADD")).upper(),
                        "target_id": d.get("target_id"),
                    })
            if normalized:
                # Build complete decision list, filling gaps
                decided = {d["index"]: d for d in normalized}
                result = []
                for i in range(len(new_notes)):
                    if i in decided:
                        result.append(decided[i])
                    else:
                        result.append({"index": i, "op": "ADD", "target_id": None})
                return result

        _log.warning(
            "Failed to parse batch memory ops — defaulting to ADD all\n"
            "  raw[:500] = {!r}",
            raw[:500],
        )
        return [{"index": i, "op": "ADD", "target_id": None}
                for i in range(len(new_notes))]

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
    ) -> Tuple[int, int]:
        """Generate all pairwise links in one LLM call.

        Returns:
            ``(total_links, cross_session_links)`` where *cross_session_links*
            are links between a new note and a pre-existing note (from an
            earlier session).
        """
        if not added_notes:
            return 0, 0

        # Track IDs: which notes are new vs pre-existing
        new_ids = {n.id for n in added_notes}

        # Find the most similar existing notes for context
        all_existing = memory_bank.list_notes()

        # Get neighbors for each added note via ANN
        neighbor_set: set[str] = set()
        for note in added_notes:
            neighbors = memory_bank.ann_search(note.e, k=self._top_k_neighbors)
            for n in neighbors:
                if n.id not in new_ids:
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
        relations = _extract_json(raw, expect_array=True)

        if isinstance(relations, dict) and not isinstance(relations, list):
            # Some models wrap the array in an object: {"relations": [...]}
            for key in ("relations", "links", "edges", "results", "data"):
                if isinstance(relations.get(key), list):
                    relations = relations[key]
                    break

        if not isinstance(relations, list) or not relations:
            _log.warning(
                "Failed to parse batch link generation — no links created\n"
                "  raw[:500] = {!r}",
                raw[:500],
            )
            return 0, 0

        # Apply bidirectional links, tracking cross-session
        all_note_map = {n.id: n for n in all_existing}
        for n in added_notes:
            all_note_map[n.id] = n

        link_count = 0
        cross_session = 0

        for rel in relations:
            if not isinstance(rel, dict):
                continue
            source = str(rel.get("source", ""))
            target = str(rel.get("target", ""))
            relation_type = str(rel.get("relation", "linked"))
            if source not in all_note_map or target not in all_note_map:
                continue
            if source == target:
                continue

            # Determine if this is a cross-session link
            src_is_new = source in new_ids
            tgt_is_new = target in new_ids
            is_cross = src_is_new != tgt_is_new  # XOR: one new, one old

            # Add bidirectional links
            src_note = all_note_map[source]
            tgt_note = all_note_map[target]
            if target not in src_note.L:
                src_note.L.append(target)
            if source not in tgt_note.L:
                tgt_note.L.append(source)

            # Persist
            memory_bank.update(source, {"L": src_note.L})
            memory_bank.update(target, {"L": tgt_note.L})

            link_count += 1
            if is_cross:
                cross_session += 1

            _log.debug(
                "  link | {} --[{}]--> {}  {}",
                source[:8], relation_type, target[:8],
                "(cross-session)" if is_cross else "(intra-session)",
            )

        _log.info(
            "Links created | total={}  cross_session={}  intra_session={}",
            link_count, cross_session, link_count - cross_session,
        )
        return link_count, cross_session
