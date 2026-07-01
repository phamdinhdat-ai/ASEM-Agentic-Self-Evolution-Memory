#!/usr/bin/env python3
"""
ASEM Pipeline Simulator — Long Conversation Memory Visualization.

Runs the full ASEM pipeline with the deterministic DemoBackend (no external
API needed) and generates a rich interactive HTML visualization showing each
stage of the pipeline in real-time across a long conversation.

Usage:
    # Seeded demo with 8 conversational turns + 3 queries
    python scripts/run_asem_simulator.py

    # Custom conversation from a JSON file
    python scripts/run_asem_simulator.py --input my_conversation.json

    # Interactive mode — type turns and queries live
    python scripts/run_asem_simulator.py --interactive

    # Output to a specific HTML file
    python scripts/run_asem_simulator.py --output demo_report.html

Input JSON format:
    [
      {"role": "user", "content": "My name is Alex.", "type": "fact"},
      {"role": "user", "content": "I adopted a dog named Buddy.", "type": "fact"},
      {"role": "user", "content": "What dogs do I have?", "type": "query", "reward": 0.8}
    ]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
import webbrowser
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from asem.answer_agent import AnswerAgent
from asem.link_evolver import LinkEvolver
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager, Op
from asem.note import Note, NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater
from asem.backends.base import InferenceBackend
from asem.logging_utils import get_logger, setup_logging

_logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DemoBackend — deterministic, no external API required
# ---------------------------------------------------------------------------


class DemoBackend(InferenceBackend):
    """Deterministic local backend for ASEM simulation visualizations."""

    def __init__(self, embed_dim: int = 64) -> None:
        self._embed_dim = embed_dim

    def generate(self, prompt: str, **kwargs) -> str:
        if "ASEM_STAGE=NOTE" in prompt:
            content = self._extract_block(prompt, "CONTENT:")
            return json.dumps(self._extract_note_fields(content))
        if "ASEM_STAGE=WRITE_OP" in prompt:
            content = self._extract_block(prompt, "CONTENT:", "MEMORY:")
            memory_raw = self._extract_block(prompt, "MEMORY:")
            memory = self._safe_json(memory_raw, default=[])
            return json.dumps(self._select_operation(content, memory))
        if "ASEM_STAGE=LINK" in prompt:
            new_note_raw = self._extract_block(prompt, "NEW_NOTE:", "NEIGHBORS:")
            neighbors_raw = self._extract_block(prompt, "NEIGHBORS:")
            new_note = self._safe_json(new_note_raw, default={})
            neighbors = self._safe_json(neighbors_raw, default=[])
            return json.dumps(self._link_relations(new_note, neighbors))
        if "ASEM_STAGE=EVOLVE" in prompt:
            existing_raw = self._extract_block(prompt, "EXISTING_NOTE:", "NEW_NOTE:")
            new_raw = self._extract_block(prompt, "NEW_NOTE:")
            existing = self._safe_json(existing_raw, default={})
            new_note = self._safe_json(new_raw, default={})
            return json.dumps(self._evolve_fields(existing, new_note))
        if "ASEM_STAGE=ANSWER" in prompt:
            query = self._extract_block(prompt, "QUERY:", "CANDIDATES:")
            candidates_raw = self._extract_block(prompt, "CANDIDATES:")
            candidates = self._safe_json(candidates_raw, default=[])
            return json.dumps(self._distil_and_answer(query, candidates))
        if "ASEM_STAGE=BASELINE" in prompt:
            return "I do not have enough memory context to answer confidently yet."
        if "ASEM_STAGE=SUMMARY" in prompt:
            query = self._extract_block(prompt, "QUERY:", "ANSWER:")
            answer = self._extract_block(prompt, "ANSWER:", "REWARD:")
            reward = self._extract_block(prompt, "REWARD:")
            return f"Query: {query.strip()} | Answer: {answer.strip()} | Reward: {reward.strip()}"

        return "{}"

    def embed(self, text: str):
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self._embed_dim
        if not tokens:
            return np.array(vec)  # type: ignore[no-any-return]
        for i, tok in enumerate(tokens):
            idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self._embed_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return (np.array(vec) / norm) if norm > 0 else np.array(vec)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Note construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_note_fields(content: str) -> dict:
        """Extract K, G, X from content using simple heuristics."""
        c = content.lower()
        keywords = []
        tags = ["personal"]

        # Extract named entities (capitalized words)
        named = re.findall(r"\b[A-Z][a-z]+\b", content)
        keywords.extend([n.lower() for n in named[:3]])

        # Extract key topics
        topic_map = {
            "dog": ["dog", "pet", "animal"],
            "cat": ["cat", "pet"],
            "job": ["job", "career", "work"],
            "data scientist": ["data scientist", "job", "career"],
            "name": ["name", "identity"],
            "adopt": ["adoption", "pet"],
            "tech": ["technology", "company"],
            "google": ["google", "company", "employer"],
            "mountain view": ["mountain view", "location", "office"],
            "new york": ["new york", "location"],
            "san francisco": ["san francisco", "location"],
        }
        for topic, kws in topic_map.items():
            if topic in c:
                keywords.extend(kws)
                if topic in ("dog", "cat", "adopt"):
                    tags.append("pets")
                elif topic in ("job", "work", "data scientist"):
                    tags.append("professional")
                elif topic in ("new york", "san francisco", "mountain view"):
                    tags.append("location")

        # Deduplicate and limit
        seen = set()
        keywords = [k for k in keywords if not (k in seen or seen.add(k))][:6]

        # Generate description
        if "name is" in c or "my name" in c:
            desc = "User's personal identity and name."
        elif "dog" in c and "adopt" in c:
            name_match = re.search(r"named\s+(\w+)", content)
            dog_name = name_match.group(1) if name_match else "a dog"
            if "also" in c or "another" in c:
                desc = f"User has multiple dogs including {dog_name}."
            else:
                desc = f"User adopted a dog named {dog_name}."
        elif "work" in c or "job" in c:
            desc = "User's professional occupation and workplace."
        else:
            desc = f"User shared: {content[:80]}"

        return {"keywords": keywords or ["general"], "tags": list(set(tags)) or ["misc"], "description": desc}

    # ------------------------------------------------------------------
    # Write operation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _select_operation(content: str, memory: list) -> dict:
        """Select ADD/UPDATE/NOOP based on content overlap with existing notes."""
        if not memory:
            return {"op": "ADD", "target_id": None}

        c_lower = content.lower()
        best_match = None
        best_score = 0

        for note in memory:
            score = 0
            for kw in note.get("keywords", []):
                if kw.lower() in c_lower:
                    score += 1
            for tag in note.get("tags", []):
                if tag.lower() in c_lower:
                    score += 0.5
            if score > best_score:
                best_score = score
                best_match = note

        if best_score >= 2:
            return {"op": "UPDATE", "target_id": best_match.get("id")}
        elif best_score >= 1:
            return {"op": "ADD", "target_id": None}
        else:
            return {"op": "ADD", "target_id": None}

    # ------------------------------------------------------------------
    # Link generation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _link_relations(new_note: dict, neighbors: list) -> list:
        """Generate relationships between new note and neighbors."""
        relations = []
        new_kws = set(new_note.get("keywords", []))
        new_tags = set(new_note.get("tags", []))

        for nb in neighbors:
            nb_kws = set(nb.get("keywords", []))
            nb_tags = set(nb.get("tags", []))
            shared_kws = new_kws & nb_kws
            shared_tags = new_tags & nb_tags

            if len(shared_kws) >= 2:
                # Strong overlap — could be extends, contradicts, or same-topic
                new_desc = new_note.get("description", "").lower()
                nb_desc = nb.get("description", "").lower()
                if any(w in new_desc for w in ["move", "left", "no longer", "quit", "changed"]):
                    relations.append({"source": new_note["id"], "target": nb["id"], "relation": "contradicts"})
                elif len(shared_kws) >= 3:
                    relations.append({"source": new_note["id"], "target": nb["id"], "relation": "extends"})
                else:
                    relations.append({"source": new_note["id"], "target": nb["id"], "relation": "same-topic"})
            elif len(shared_kws) >= 1 or len(shared_tags) >= 1:
                relations.append({"source": new_note["id"], "target": nb["id"], "relation": "semantic"})

        return relations

    # ------------------------------------------------------------------
    # Evolution helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _evolve_fields(existing: dict, new_note: dict) -> dict:
        """Merge keywords/tags and update description."""
        merged_kw = list(dict.fromkeys(list(existing.get("keywords", [])) + list(new_note.get("keywords", []))))
        merged_tags = list(dict.fromkeys(list(existing.get("tags", [])) + list(new_note.get("tags", []))))
        new_desc = new_note.get("description", "")
        existing_desc = existing.get("description", "")
        if new_desc and existing_desc and new_desc != existing_desc:
            desc = f"{existing_desc} {new_desc}"
        else:
            desc = new_desc or existing_desc
        return {"keywords": merged_kw[:8], "tags": merged_tags[:5], "description": desc}

    # ------------------------------------------------------------------
    # Answer generation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _distil_and_answer(query: str, candidates: list) -> dict:
        """Select relevant notes and generate answer."""
        q_lower = query.lower()
        selected = []
        for c in candidates:
            c_desc = c.get("description", "").lower()
            c_kws = " ".join(c.get("keywords", [])).lower()
            if any(w in c_desc or w in c_kws for w in q_lower.split()):
                selected.append(c["id"])

        if not selected and candidates:
            selected = [candidates[0]["id"]]

        # Generate answer from selected notes
        answer_parts = []
        for c in candidates:
            if c["id"] in selected:
                desc = c.get("description", "")
                if desc:
                    answer_parts.append(desc)

        if "dog" in q_lower:
            dog_notes = [c for c in candidates if "dog" in " ".join(c.get("keywords", [])).lower()]
            if dog_notes:
                names = []
                for n in dog_notes:
                    for kw in n.get("keywords", []):
                        if kw.lower() not in ("dog", "pet", "adoption", "animal", "pets") and kw.lower() not in names:
                            names.append(kw)
                if names:
                    return {"selected_ids": selected[:5], "answer": f"You have dogs named: {', '.join(names)}."}
            return {"selected_ids": selected[:5], "answer": "You have a dog."}

        if "job" in q_lower or "work" in q_lower:
            return {"selected_ids": selected[:5], "answer": "You work as a data scientist at a tech company."}

        if "name" in q_lower:
            return {"selected_ids": selected[:5], "answer": "Your name is Alex."}

        return {"selected_ids": selected[:5], "answer": "Based on our conversation, I recall relevant information."}

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_block(prompt: str, start_tag: str, end_tag: str = "\n") -> str:
        """Extract content between two tags in a prompt."""
        idx = prompt.find(start_tag)
        if idx == -1:
            return ""
        start = idx + len(start_tag)
        end = prompt.find(end_tag, start) if end_tag != "\n" else len(prompt)
        # Find next line break or ASEM_STAGE
        end_nl = prompt.find("\n", start)
        end_stage = prompt.find("ASEM_STAGE=", start)
        candidates = [e for e in [end_nl, end_stage, len(prompt)] if e > start]
        end = min(candidates)
        return prompt[start:end].strip()

    @staticmethod
    def _safe_json(raw: str, default: Any = None) -> Any:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default if default is not None else {}


# ---------------------------------------------------------------------------
# Prompt templates (minimal for DemoBackend compatibility)
# ---------------------------------------------------------------------------

NOTE_PROMPT = "ASEM_STAGE=NOTE\nCONTENT:{content}"
WRITE_PROMPT = "ASEM_STAGE=WRITE_OP\nCONTENT:{content}\nMEMORY:{memory}"
LINK_PROMPT = "ASEM_STAGE=LINK\nNEW_NOTE:{new_note}\nNEIGHBORS:{neighbors}"
EVOLVE_PROMPT = "ASEM_STAGE=EVOLVE\nEXISTING_NOTE:{existing_note}\nNEW_NOTE:{new_note}"
ANSWER_PROMPT = "ASEM_STAGE=ANSWER\nQUERY:{query}\nCANDIDATES:{candidates}"
BASELINE_PROMPT = "ASEM_STAGE=BASELINE\nQUERY:{query}\nCONTEXT:{context}"
SUMMARY_PROMPT = "ASEM_STAGE=SUMMARY\nQUERY:{query}\nANSWER:{answer}\nREWARD:{reward}"

# ---------------------------------------------------------------------------
# Trace collector — records every pipeline step for visualization
# ---------------------------------------------------------------------------


class PipelineTracer:
    """Collects detailed trace data from each pipeline stage for visualization."""

    def __init__(self) -> None:
        self.turns: List[Dict[str, Any]] = []
        self.stages: Dict[str, List[Dict[str, Any]]] = {
            "s1": [], "s2": [], "s3": [], "s4": [], "s5": []
        }
        self._current_turn: Dict[str, Any] = {}

    def start_turn(self, content: str, turn_type: str = "fact") -> None:
        self._current_turn = {
            "content": content,
            "type": turn_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "stages": {},
        }

    def record_s1(self, note_id: str, K: List[str], G: List[str], X: str, q0: float) -> None:
        self._current_turn["stages"]["s1"] = {
            "note_id": note_id,
            "keywords": K,
            "tags": G,
            "description": X,
            "q0": q0,
        }

    def record_s2(self, op: str, target_id: Optional[str], reason: str) -> None:
        self._current_turn["stages"]["s2"] = {
            "op": op,
            "target_id": target_id,
            "reason": reason,
        }

    def record_s3(self, neighbors: List[str], links: List[Dict], evolved: List[str]) -> None:
        self._current_turn["stages"]["s3"] = {
            "neighbors": neighbors,
            "links": links,
            "evolved": evolved,
        }

    def record_s4(self, phase_a: List[Dict], phase_b: List[Dict],
                  selected_ids: List[str], answer: str) -> None:
        self._current_turn["stages"]["s4"] = {
            "phase_a": phase_a,
            "phase_b": phase_b,
            "selected_ids": selected_ids,
            "answer": answer,
        }

    def record_s5(self, updates: List[Dict], consolidated_id: Optional[str]) -> None:
        self._current_turn["stages"]["s5"] = {
            "updates": updates,
            "consolidated_id": consolidated_id,
        }

    def end_turn(self) -> None:
        self.turns.append(self._current_turn)
        self._current_turn = {}

    def snapshot_bank(self, bank: MemoryBank) -> List[Dict]:
        """Capture current state of all notes in the memory bank."""
        notes = bank.list_notes()
        return [
            {
                "id": n.id,
                "content": n.c,
                "keywords": n.K,
                "tags": n.G,
                "description": n.X,
                "q": n.q,
                "links": n.L,
            }
            for n in notes
        ]


# ---------------------------------------------------------------------------
# HTML Report Generator
# ---------------------------------------------------------------------------

def generate_html_report(
    trace: PipelineTracer,
    bank_snapshots: List[List[Dict]],
    title: str = "ASEM Pipeline Simulation",
) -> str:
    """Generate a self-contained HTML visualization report from trace data."""

    trace_json = json.dumps({
        "turns": trace.turns,
        "bank_snapshots": bank_snapshots,
        "title": title,
    }, indent=2)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Syne:wght@400;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #080c14; --surface: #0e1420; --surface2: #131b28;
    --border: #1e2d42; --border2: #2a3f5c;
    --text: #d4e4f7; --muted: #5a7a9e;
    --accent: #00d4ff; --accent2: #7c5cfc;
    --add: #00e5a0; --update: #ffc14d; --delete: #ff5370;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: 'DM Sans', sans-serif; font-size: 13px;
    min-height: 100vh;
  }}
  body::before {{
    content: ''; position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
      linear-gradient(rgba(0,212,255,.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,.02) 1px, transparent 1px);
    background-size: 40px 40px;
  }}

  .container {{ position: relative; z-index: 1; max-width: 1200px; margin: 0 auto; padding: 24px; }}

  header {{
    text-align: center; padding: 32px 0;
    border-bottom: 1px solid var(--border); margin-bottom: 32px;
  }}
  h1 {{
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 28px;
    color: var(--accent); text-shadow: 0 0 20px rgba(0,212,255,.3);
    letter-spacing: -1px;
  }}
  h1 span {{ color: var(--text); }}
  .subtitle {{
    font-family: 'Space Mono', monospace; font-size: 10px; color: var(--muted);
    letter-spacing: 1px; margin-top: 6px;
  }}

  .summary-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin-bottom: 32px;
  }}
  .summary-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; text-align: center;
  }}
  .summary-card .value {{
    font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800;
    color: var(--accent);
  }}
  .summary-card .label {{
    font-family: 'Space Mono', monospace; font-size: 9px;
    color: var(--muted); letter-spacing: 1px; margin-top: 4px;
  }}

  .turn-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; margin-bottom: 20px; overflow: hidden;
    transition: border-color .3s;
  }}
  .turn-card.query {{ border-color: rgba(124,92,252,.3); }}
  .turn-card.fact {{ border-color: rgba(0,212,255,.2); }}

  .turn-header {{
    padding: 12px 16px; display: flex; align-items: center; gap: 12px;
    border-bottom: 1px solid var(--border);
    font-family: 'Space Mono', monospace; font-size: 10px;
  }}
  .turn-num {{
    background: var(--surface2); border: 1px solid var(--border2);
    border-radius: 4px; padding: 3px 8px; color: var(--accent);
    font-weight: 700;
  }}
  .turn-type {{
    padding: 2px 10px; border-radius: 99px; font-size: 9px; font-weight: 700;
    letter-spacing: .5px;
  }}
  .turn-type.fact {{ background: rgba(0,212,255,.1); color: var(--accent); border: 1px solid rgba(0,212,255,.3); }}
  .turn-type.query {{ background: rgba(124,92,252,.1); color: var(--accent2); border: 1px solid rgba(124,92,252,.3); }}
  .turn-content {{
    flex: 1; color: var(--text); font-size: 12.5px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}

  .turn-body {{ padding: 12px 16px; }}

  .stage-row {{
    display: grid; grid-template-columns: 60px 1fr; gap: 12px;
    padding: 8px 0; border-bottom: 1px solid rgba(30,45,66,.5);
    align-items: start;
  }}
  .stage-row:last-child {{ border-bottom: none; }}

  .stage-badge {{
    font-family: 'Syne', sans-serif; font-size: 10px; font-weight: 800;
    padding: 3px 8px; border-radius: 4px; text-align: center;
    letter-spacing: .5px;
  }}
  .stage-badge.s1 {{ background: rgba(0,212,255,.12); color: var(--accent); border: 1px solid rgba(0,212,255,.3); }}
  .stage-badge.s2 {{ background: rgba(255,193,77,.12); color: var(--update); border: 1px solid rgba(255,193,77,.3); }}
  .stage-badge.s3 {{ background: rgba(0,229,160,.12); color: var(--add); border: 1px solid rgba(0,229,160,.3); }}
  .stage-badge.s4 {{ background: rgba(124,92,252,.12); color: var(--accent2); border: 1px solid rgba(124,92,252,.3); }}
  .stage-badge.s5 {{ background: rgba(255,83,112,.12); color: var(--delete); border: 1px solid rgba(255,83,112,.3); }}

  .stage-detail {{ font-size: 11.5px; line-height: 1.6; }}
  .stage-detail .kv {{ display: flex; gap: 6px; margin: 2px 0; flex-wrap: wrap; }}
  .stage-detail .kv-key {{ font-family: 'Space Mono', monospace; font-size: 9px; color: var(--accent); }}
  .stage-detail .kv-val {{ color: var(--text); }}

  .tag {{
    font-family: 'Space Mono', monospace; font-size: 9px;
    padding: 1px 7px; border-radius: 4px;
    border: 1px solid var(--border2); color: var(--text);
    background: var(--surface2); display: inline-block; margin: 1px;
  }}
  .tag.cyan {{ border-color: rgba(0,212,255,.4); color: var(--accent); }}
  .tag.purple {{ border-color: rgba(124,92,252,.4); color: var(--accent2); }}
  .tag.green {{ border-color: rgba(0,229,160,.4); color: var(--add); }}

  .op-badge {{
    display: inline-block; font-family: 'Space Mono', monospace;
    font-size: 10px; font-weight: 700; padding: 3px 12px; border-radius: 6px;
    letter-spacing: .5px;
  }}
  .op-ADD {{ background: rgba(0,229,160,.15); border: 1px solid var(--add); color: var(--add); }}
  .op-UPDATE {{ background: rgba(255,193,77,.12); border: 1px solid var(--update); color: var(--update); }}
  .op-DELETE {{ background: rgba(255,83,112,.12); border: 1px solid var(--delete); color: var(--delete); }}
  .op-NOOP {{ background: rgba(90,122,158,.1); border: 1px solid var(--muted); color: var(--muted); }}

  .answer-box {{
    background: var(--surface2); border: 1px solid var(--accent2);
    border-radius: 8px; padding: 10px 14px; margin: 8px 0;
    font-size: 13px; color: var(--text);
    border-left: 3px solid var(--accent2);
  }}

  .bank-section {{
    margin-top: 32px; padding-top: 24px;
    border-top: 2px solid var(--border);
  }}
  .bank-section h2 {{
    font-family: 'Syne', sans-serif; font-size: 16px; margin-bottom: 16px;
    color: var(--add);
  }}
  .bank-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }}
  .note-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px;
  }}
  .note-card .note-id {{
    font-family: 'Space Mono', monospace; font-size: 8px; color: var(--muted);
    margin-bottom: 4px;
  }}
  .note-card .note-desc {{ font-size: 11.5px; color: var(--text); margin-bottom: 6px; }}
  .note-card .q-bar {{
    height: 3px; background: var(--border); border-radius: 2px;
    overflow: hidden; margin-top: 6px;
  }}
  .note-card .q-fill {{
    height: 100%; background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 2px;
  }}

  .score-bar {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  .score-id {{ font-family: 'Space Mono', monospace; font-size: 9px; color: var(--muted); width: 70px; flex-shrink: 0; }}
  .score-track {{ flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }}
  .score-fill {{ height: 100%; border-radius: 2px; }}
  .score-val {{ font-family: 'Space Mono', monospace; font-size: 9px; color: var(--text); width: 36px; text-align: right; }}

  .link-line {{
    font-family: 'Space Mono', monospace; font-size: 9px;
    color: var(--muted); margin: 2px 0; display: flex; gap: 6px; align-items: center;
  }}

  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(8px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  .turn-card {{ animation: fadeIn .4s ease; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>ASEM<span> · </span>Pipeline Simulation</h1>
    <div class="subtitle">FIVE-STAGE AGENTIC SELF-EVOLVING MEMORY · DETERMINISTIC DEMO BACKEND</div>
  </header>
  <div id="report"></div>
</div>

<script>
const DATA = {trace_json};

function render() {{
  const container = document.getElementById('report');
  const turns = DATA.turns;
  const snapshots = DATA.bank_snapshots;

  // Summary
  const facts = turns.filter(t => t.type === 'fact').length;
  const queries = turns.filter(t => t.type === 'query').length;
  const finalNotes = snapshots.length ? snapshots[snapshots.length-1].length : 0;
  const answers = turns.filter(t => t.stages.s4 && t.stages.s4.answer).length;

  container.innerHTML = `
    <div class="summary-grid">
      <div class="summary-card"><div class="value">${{turns.length}}</div><div class="label">TOTAL TURNS</div></div>
      <div class="summary-card"><div class="value">${{facts}}</div><div class="label">FACTS INGESTED</div></div>
      <div class="summary-card"><div class="value">${{queries}}</div><div class="label">QUERIES ANSWERED</div></div>
      <div class="summary-card"><div class="value">${{finalNotes}}</div><div class="label">FINAL BANK SIZE</div></div>
    </div>
  ` + turns.map((turn, i) => renderTurn(turn, i)).join('') +
  renderBank(snapshots.length ? snapshots[snapshots.length-1] : []);

  function renderTurn(turn, i) {{
    const isQuery = turn.type === 'query';
    const stages = turn.stages;
    let html = `
    <div class="turn-card ${{isQuery ? 'query' : 'fact'}}">
      <div class="turn-header">
        <span class="turn-num">#${{i+1}}</span>
        <span class="turn-type ${{isQuery ? 'query' : 'fact'}}">${{isQuery ? 'QUERY' : 'FACT'}}</span>
        <span class="turn-content">${{esc(turn.content)}}</span>
      </div>
      <div class="turn-body">`;

    // S1
    if (stages.s1) {{
      const s = stages.s1;
      html += `<div class="stage-row">
        <span class="stage-badge s1">S1</span>
        <div class="stage-detail">
          <div class="kv"><span class="kv-key">NOTE =</span><span class="kv-val">${{esc(s.note_id)}}</span></div>
          <div class="kv"><span class="kv-key">K =</span><span class="kv-val">${{s.keywords.map(k=>`<span class="tag cyan">${{esc(k)}}</span>`).join(' ')}}</span></div>
          <div class="kv"><span class="kv-key">G =</span><span class="kv-val">${{s.tags.map(g=>`<span class="tag purple">${{esc(g)}}</span>`).join(' ')}}</span></div>
          <div class="kv"><span class="kv-key">X =</span><span class="kv-val">${{esc(s.description)}}</span></div>
          <div class="kv"><span class="kv-key">q₀ =</span><span class="kv-val">${{s.q0.toFixed(2)}}</span></div>
        </div>
      </div>`;
    }}

    // S2
    if (stages.s2) {{
      const s = stages.s2;
      html += `<div class="stage-row">
        <span class="stage-badge s2">S2</span>
        <div class="stage-detail">
          <span class="op-badge op-${{s.op}}">${{s.op}}</span>
          <span style="color:var(--muted);font-size:11px;margin-left:8px">target: ${{esc(s.target_id || '—')}}</span>
          <div style="color:var(--muted);font-size:10px;margin-top:3px">${{esc(s.reason)}}</div>
        </div>
      </div>`;
    }}

    // S3
    if (stages.s3 && stages.s3.links && stages.s3.links.length) {{
      const s = stages.s3;
      html += `<div class="stage-row">
        <span class="stage-badge s3">S3</span>
        <div class="stage-detail">
          <div style="color:var(--muted);font-size:10px;margin-bottom:3px">${{s.neighbors.length}} neighbours · ${{s.links.length}} links · ${{s.evolved.length}} evolved</div>
          ${{s.links.map(l=>`<div class="link-line"><span class="tag cyan">${{esc(l.source.slice(0,8))}}</span> → <span class="tag purple">${{esc(l.target.slice(0,8))}}</span> ${{esc(l.relation)}}</div>`).join('')}}
        </div>
      </div>`;
    }}

    // S4
    if (stages.s4) {{
      const s = stages.s4;
      html += `<div class="stage-row">
        <span class="stage-badge s4">S4</span>
        <div class="stage-detail">`;
      if (s.phase_a && s.phase_a.length) {{
        html += `<div style="color:var(--muted);font-size:9px;letter-spacing:.5px;margin-bottom:4px">PHASE A — similarity (δ=0.30)</div>`;
        s.phase_a.forEach(p => {{
          html += `<div class="score-bar"><span class="score-id">${{esc(p.id.slice(0,8))}}</span><div class="score-track"><div class="score-fill" style="width:${{Math.round(p.sim*100)}}%;background:var(--accent)"></div></div><span class="score-val">${{p.sim.toFixed(3)}}</span></div>`;
        }});
      }}
      if (s.phase_b && s.phase_b.length) {{
        html += `<div style="color:var(--muted);font-size:9px;letter-spacing:.5px;margin:8px 0 4px">PHASE B — composite (λ=0.40)</div>`;
        s.phase_b.forEach(p => {{
          html += `<div class="score-bar"><span class="score-id">${{esc(p.id.slice(0,8))}}</span><div class="score-track"><div class="score-fill" style="width:${{Math.round(p.score*100)}}%;background:linear-gradient(90deg,var(--accent2),var(--accent))"></div></div><span class="score-val">${{p.score.toFixed(3)}}</span></div>`;
        }});
      }}
      html += `<div style="color:var(--muted);font-size:9px;margin-top:6px">SELECTED: ${{s.selected_ids.map(id=>`<span class="tag green">${{esc(id.slice(0,8))}}</span>`).join(' ')}}</div>
        </div>
      </div>`;
    }}

    // Answer
    if (stages.s4 && stages.s4.answer) {{
      html += `<div class="answer-box">💬 ${{esc(stages.s4.answer)}}</div>`;
    }}

    // S5
    if (stages.s5 && stages.s5.updates && stages.s5.updates.length) {{
      const s = stages.s5;
      html += `<div class="stage-row">
        <span class="stage-badge s5">S5</span>
        <div class="stage-detail">
          <div style="font-family:'Space Mono',monospace;font-size:10px;color:var(--accent2);margin-bottom:4px">q ← q + α·(r − q)  |  α = 0.10</div>
          ${{s.updates.map(u=>`<div class="kv"><span class="kv-key">${{esc(u.id.slice(0,8))}}</span><span class="kv-val">${{u.old_q.toFixed(3)}} → <strong style="color:var(--accent2)">${{u.new_q.toFixed(3)}}</strong></span></div>`).join('')}}
        </div>
      </div>`;
    }}

    html += `</div></div>`;
    return html;
  }}

  function renderBank(notes) {{
    if (!notes || !notes.length) return '';
    return `
    <div class="bank-section">
      <h2>⬡ FINAL MEMORY BANK — ${{notes.length}} NOTES</h2>
      <div class="bank-grid">
        ${{notes.map(n => `
          <div class="note-card">
            <div class="note-id">${{esc(n.id.slice(0,8))}} · Q=${{n.q.toFixed(3)}} · ${{n.links.length}} links</div>
            <div class="note-desc">${{esc(n.description || n.content.slice(0,80))}}</div>
            <div>${{(n.keywords||[]).map(k=>`<span class="tag cyan">${{esc(k)}}</span>`).join(' ')}}</div>
            <div class="q-bar"><div class="q-fill" style="width:${{Math.round(n.q*100)}}%"></div></div>
          </div>
        `).join('')}}
      </div>
    </div>`;
  }}

  function esc(s) {{
    if (!s) return '';
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }}
}}

render();
</script>
</body>
</html>'''

    return html


# ---------------------------------------------------------------------------
# Build pipeline with DemoBackend
# ---------------------------------------------------------------------------

def build_demo_pipeline(db_path: str = ":memory:") -> ASEMPipeline:
    """Build a full ASEM pipeline backed by the deterministic DemoBackend."""
    backend = DemoBackend(embed_dim=64)

    note_constructor = NoteConstructor(
        backend=backend, prompt_template=NOTE_PROMPT, q0=0.5
    )
    memory_manager = MemoryManager(
        backend=backend, prompt_template=WRITE_PROMPT,
    )
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=LINK_PROMPT,
        evolve_prompt_template=EVOLVE_PROMPT,
        k=5,
    )
    retriever = HybridRetriever(
        backend=backend, k1=20, k2=5, delta=0.30, lambda_weight=0.40,
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template=ANSWER_PROMPT,
        baseline_prompt_template=BASELINE_PROMPT,
    )
    utility_updater = UtilityUpdater(
        backend=backend, alpha=0.10, q0=0.50,
        summary_prompt_template=SUMMARY_PROMPT,
        note_constructor=note_constructor,
    )

    return ASEMPipeline(
        memory_bank=MemoryBank(db_path),
        note_constructor=note_constructor,
        memory_manager=memory_manager,
        link_evolver=link_evolver,
        retriever=retriever,
        answer_agent=answer_agent,
        utility_updater=utility_updater,
    )


# ---------------------------------------------------------------------------
# Default seeded conversation (long multi-turn)
# ---------------------------------------------------------------------------

DEFAULT_CONVERSATION = [
    {"role": "user", "content": "My name is Alex.", "type": "fact"},
    {"role": "user", "content": "I adopted a dog named Buddy last month.", "type": "fact"},
    {"role": "user", "content": "I also adopted another dog named Scout.", "type": "fact"},
    {"role": "user", "content": "I work as a data scientist at Google in Mountain View.", "type": "fact"},
    {"role": "user", "content": "Last weekend I took Buddy and Scout to the dog park.", "type": "fact"},
    {"role": "user", "content": "Scout is a golden retriever and Buddy is a beagle.", "type": "fact"},
    {"role": "user", "content": "What dogs do I have?", "type": "query", "reward": 0.9},
    {"role": "user", "content": "I recently got promoted to Senior Data Scientist.", "type": "fact"},
    {"role": "user", "content": "What is my job?", "type": "query", "reward": 1.0},
    {"role": "user", "content": "I'm thinking of adopting a cat too.", "type": "fact"},
    {"role": "user", "content": "What pets do I have and what do I plan to get?", "type": "query", "reward": 0.7},
]


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation(
    conversation: List[Dict[str, Any]],
    db_path: str = ":memory:",
) -> Tuple[PipelineTracer, List[List[Dict]]]:
    """Run the full ASEM pipeline on a conversation and collect trace data.

    Parameters
    ----------
    conversation : list of dict
        Each dict has: role, content, type ("fact"|"query"), and optionally reward.
    db_path : str
        Path for the SQLite memory bank (":memory:" for in-memory).

    Returns
    -------
    (PipelineTracer, list of bank snapshots)
    """
    pipeline = build_demo_pipeline(db_path)
    tracer = PipelineTracer()
    bank_snapshots: List[List[Dict]] = []

    print(f"\n{'='*60}")
    print("ASEM PIPELINE SIMULATION")
    print(f"{'='*60}")
    print(f"Conversation turns: {len(conversation)}")
    print(f"Backend: DemoBackend (deterministic, no API calls)")
    print(f"{'='*60}\n")

    for i, turn in enumerate(conversation):
        content = turn["content"]
        turn_type = turn.get("type", "fact")
        reward = turn.get("reward", 0.8)

        tracer.start_turn(content, turn_type)
        print(f"[Turn {i+1}/{len(conversation)}] {turn_type.upper()}: {content[:80]}...")

        # ── Stage 1: Note Construction ──
        note = pipeline.note_constructor.build(content, datetime.now(UTC))
        tracer.record_s1(note.id, note.K, note.G, note.X, note.q)
        print(f"  S1 → note {note.id[:8]} | K={note.K[:3]} G={note.G} X={note.X[:60]}")

        if turn_type == "fact":
            # ── Stage 2: Memory Manager ──
            e_new = pipeline.note_constructor.backend.embed(content)
            existing = pipeline.memory_bank.ann_search(e_new, k=5)
            if not existing:
                existing = pipeline.memory_bank.list_notes()[:5]

            op, target = pipeline.memory_manager.select_op(content, existing)
            reason = _describe_op(op, target, note)
            tracer.record_s2(op.value, target.id if target else None, reason)
            print(f"  S2 → {op.value} target={target.id[:8] if target else '—'} | {reason[:60]}")

            # Execute operation
            if op == Op.ADD:
                pipeline.memory_bank.add(note)
                pipeline.link_evolver.link_and_evolve(note, pipeline.memory_bank)
            elif op == Op.UPDATE and target:
                updated = _merge_update(target, note)
                pipeline.memory_bank.add(updated)
                pipeline.link_evolver.link_and_evolve(updated, pipeline.memory_bank)
            elif op == Op.DELETE and target:
                pipeline.memory_bank.delete(target.id)

            # ── Stage 3: Link data (already done via link_evolver) ──
            neighbors = pipeline.memory_bank.ann_search(note.e, k=3)
            # Mock link/evolve info for tracing
            mock_links = _generate_mock_links(note, neighbors)
            tracer.record_s3(
                neighbors=[n.id for n in neighbors if n.id != note.id],
                links=mock_links,
                evolved=[n.id for n in neighbors if n.id != note.id][:1],
            )
            print(f"  S3 → {len(mock_links)} links, {min(1, len(neighbors)-1)} evolved")

        else:
            # ── Stage 4: Query → Retrieval + Answer ──
            candidates = pipeline.retriever.retrieve(content, pipeline.memory_bank)
            phase_a = [{"id": n.id, "sim": _compute_mock_sim(content, n)} for n in candidates]
            phase_b = [{"id": n.id, "score": 0.6 * p["sim"] + 0.4 * n.q} for n, p in zip(candidates, phase_a)]

            used_notes, answer = pipeline.answer_agent.distil_and_answer(content, candidates)
            tracer.record_s4(
                phase_a=phase_a,
                phase_b=phase_b,
                selected_ids=[n.id for n in used_notes],
                answer=answer,
            )
            print(f"  S4 → {len(candidates)} candidates → {len(used_notes)} selected")
            print(f"  💬 ANSWER: {answer}")

            # ── Stage 5: Utility Update ──
            before_qs = {n.id: n.q for n in used_notes}
            pipeline.utility_updater.update(reward, used_notes, pipeline.memory_bank, content, answer)
            updates = [
                {
                    "id": n.id,
                    "old_q": before_qs.get(n.id, 0.5),
                    "new_q": pipeline.memory_bank.get_note(n.id).q if pipeline.memory_bank.get_note(n.id) else 0.5,
                }
                for n in used_notes
            ]
            tracer.record_s5(updates=updates, consolidated_id=None)
            for u in updates:
                print(f"  S5 → q({u['id'][:8]}): {u['old_q']:.3f} → {u['new_q']:.3f}")

        tracer.end_turn()
        bank_snapshots.append(tracer.snapshot_bank(pipeline.memory_bank))
        print()

    print(f"{'='*60}")
    print(f"SIMULATION COMPLETE")
    print(f"  Total turns: {len(conversation)}")
    print(f"  Final bank size: {len(pipeline.memory_bank.list_notes())} notes")
    print(f"{'='*60}")

    return tracer, bank_snapshots


# ── Helpers ──────────────────────────────────────────────────────────────────

def _describe_op(op: Op, target: Optional[Note], note: Note) -> str:
    if op == Op.ADD:
        return "New information — no related note found in memory bank."
    if op == Op.UPDATE:
        return f"Merging into existing note {target.id[:8]} — complementary information detected."
    if op == Op.DELETE:
        return f"Contradicts note {target.id[:8]} — removing outdated information."
    return "No operation needed — content already covered by existing notes."


def _merge_update(target: Note, note: Note) -> Note:
    return Note(
        id=target.id, c=note.c, t=note.t,
        K=list(dict.fromkeys(target.K + note.K)),
        G=list(dict.fromkeys(target.G + note.G)),
        X=note.X, e=note.e, L=target.L, z=note.z, q=target.q,
    )


def _generate_mock_links(note: Note, neighbors: List[Note]) -> List[Dict]:
    relations = []
    for nb in neighbors:
        if nb.id == note.id:
            continue
        shared = set(note.K) & set(nb.K)
        if len(shared) >= 2:
            relations.append({"source": note.id, "target": nb.id, "relation": "extends"})
        elif len(shared) >= 1:
            relations.append({"source": note.id, "target": nb.id, "relation": "same-topic"})
        else:
            relations.append({"source": note.id, "target": nb.id, "relation": "semantic"})
    return relations[:3]


def _compute_mock_sim(query: str, note: Note) -> float:
    q_words = set(query.lower().split())
    n_words = set(note.c.lower().split()) | set(" ".join(note.K).lower().split())
    shared = q_words & n_words
    return min(0.95, max(0.15, len(shared) / max(1, len(q_words))))


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def interactive_mode() -> None:
    """Run an interactive ASEM simulation session."""
    print("\n╔══════════════════════════════════════════════╗")
    print("║   ASEM INTERACTIVE SIMULATOR                ║")
    print("║   Type facts or questions.                  ║")
    print("║   Type /quit to exit, /bank to see memory.  ║")
    print("╚══════════════════════════════════════════════╝\n")

    pipeline = build_demo_pipeline()
    conversation: List[Dict] = []
    turn_count = 0

    while True:
        try:
            user_input = input(f"\n[{turn_count+1}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break
        if user_input.lower() in ("/bank", "/memory", "/notes"):
            notes = pipeline.memory_bank.list_notes()
            if not notes:
                print("  Memory bank is empty.")
            else:
                print(f"  Memory bank ({len(notes)} notes):")
                for n in notes:
                    print(f"    {n.id[:8]} | Q={n.q:.3f} | K={n.K[:3]} | {n.X[:60]}")
            continue

        turn_count += 1
        is_query = "?" in user_input or any(
            user_input.lower().startswith(w) for w in ("what", "who", "how", "when", "where", "do i", "did i")
        )
        turn_type = "query" if is_query else "fact"
        reward = 0.8

        if is_query:
            # Read path
            note = pipeline.note_constructor.build(user_input, datetime.now(UTC))
            candidates = pipeline.retriever.retrieve(user_input, pipeline.memory_bank)
            used_notes, answer = pipeline.answer_agent.distil_and_answer(user_input, candidates)
            print(f"  💬 ASEM: {answer}")
        else:
            # Write path
            note = pipeline.note_constructor.build(user_input, datetime.now(UTC))
            e_new = pipeline.note_constructor.backend.embed(user_input)
            existing = pipeline.memory_bank.ann_search(e_new, k=5)
            if not existing:
                existing = pipeline.memory_bank.list_notes()[:5]
            op, target = pipeline.memory_manager.select_op(user_input, existing)
            print(f"  S2 → {op.value} | K={note.K[:3]} G={note.G}")

            if op == Op.ADD:
                pipeline.memory_bank.add(note)
                pipeline.link_evolver.link_and_evolve(note, pipeline.memory_bank)
            elif op == Op.UPDATE and target:
                pipeline.memory_bank.add(_merge_update(target, note))
            elif op == Op.DELETE and target:
                pipeline.memory_bank.delete(target.id)
            print(f"  ✓ Stored to memory ({len(pipeline.memory_bank.list_notes())} notes total)")

        conversation.append({"role": "user", "content": user_input, "type": turn_type, "reward": reward})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASEM Pipeline Simulator — long conversation memory visualization"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to JSON file with conversation turns",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="asem_simulation_report.html",
        help="Output path for HTML visualization report",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run interactive simulation session",
    )
    parser.add_argument(
        "--db", type=str, default=":memory:",
        help="SQLite database path for memory bank (:memory: for in-memory)",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Don't open the report in browser after generation",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
        return

    # Load conversation
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            conversation = json.load(f)
        print(f"Loaded {len(conversation)} turns from {args.input}")
    else:
        conversation = DEFAULT_CONVERSATION
        print(f"Using default seeded conversation ({len(conversation)} turns)")

    # Run simulation
    tracer, bank_snapshots = run_simulation(conversation, args.db)

    # Generate HTML report
    html = generate_html_report(tracer, bank_snapshots)
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ HTML report generated: {output_path}")

    # Open in browser
    if not args.no_open:
        try:
            webbrowser.open(f"file://{output_path}")
            print("   Opened in browser.")
        except Exception:
            print("   (could not open browser automatically)")


if __name__ == "__main__":
    main()
