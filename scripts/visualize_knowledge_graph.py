#!/usr/bin/env python3
"""
ASEM Knowledge Graph Visualizer — per-turn graph evolution images.

Runs the ASEM pipeline with the DemoBackend and generates a knowledge graph
image after each conversation turn, showing:
  - Notes as nodes (colored by operation: ADD=green, UPDATE=orange, DELETE=red)
  - Links/relationships as directed edges (colored by relation type)
  - Node size proportional to utility Q-value
  - Keyword labels on each node
  - Turn number, operation, and content preview in the title

Usage:
    # Default seeded conversation → graph_frames/ directory
    python scripts/visualize_knowledge_graph.py

    # Custom conversation
    python scripts/visualize_knowledge_graph.py --input my_conversation.json

    # Generate animated GIF from frames
    python scripts/visualize_knowledge_graph.py --gif

    # Interactive HTML with auto-advancing frames
    python scripts/visualize_knowledge_graph.py --html

Output:
    graph_frames/
        turn_01_fact_ADD.png
        turn_02_fact_ADD.png
        turn_03_fact_UPDATE.png
        ...
        graph_animation.html   (if --html)
        graph_evolution.gif    (if --gif)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

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

# Suppress verbose logging for clean output
setup_logging(level="WARNING")

# ---------------------------------------------------------------------------
# DemoBackend — deterministic, no external API
# ---------------------------------------------------------------------------


class DemoBackend(InferenceBackend):
    """Deterministic local backend for ASEM graph visualizations."""

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
            return "I lack sufficient memory context to answer confidently."
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
            return np.array(vec)
        for i, tok in enumerate(tokens):
            idx = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16) % self._embed_dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return (np.array(vec) / norm) if norm > 0 else np.array(vec)

    @staticmethod
    def _extract_note_fields(content: str) -> dict:
        c = content.lower()
        keywords = []
        tags = ["personal"]
        named = re.findall(r"\b[A-Z][a-z]+\b", content)
        keywords.extend([n.lower() for n in named[:3]])
        topic_map = {
            "dog": ["dog", "pet"], "cat": ["cat", "pet"],
            "job": ["job", "career"], "data scientist": ["data scientist", "job"],
            "name": ["name", "identity"], "adopt": ["adoption", "pet"],
            "google": ["google", "company"], "promot": ["promotion", "career"],
        }
        for topic, kws in topic_map.items():
            if topic in c:
                keywords.extend(kws)
                if topic in ("dog", "cat", "adopt"):
                    tags.append("pets")
                elif topic in ("job", "data scientist", "promot"):
                    tags.append("professional")
        seen = set()
        keywords = [k for k in keywords if not (k in seen or seen.add(k))][:6]
        if "name is" in c or "my name" in c:
            desc = "User's identity and name."
        elif "dog" in c:
            name_match = re.search(r"named\s+(\w+)", content)
            dog_name = name_match.group(1) if name_match else "a dog"
            desc = f"User has a dog named {dog_name}." if "also" not in c else f"User has multiple dogs including {dog_name}."
        elif "work" in c or "job" in c:
            desc = "User's professional occupation."
        else:
            desc = f"User shared: {content[:60]}"
        return {"keywords": keywords or ["general"], "tags": list(set(tags)) or ["misc"], "description": desc}

    @staticmethod
    def _select_operation(content: str, memory: list) -> dict:
        if not memory:
            return {"op": "ADD", "target_id": None}
        c_lower = content.lower()
        best_match, best_score = None, 0
        for note in memory:
            score = sum(1 for kw in note.get("keywords", []) if kw.lower() in c_lower)
            score += sum(0.5 for tag in note.get("tags", []) if tag.lower() in c_lower)
            if score > best_score:
                best_score, best_match = score, note
        if best_score >= 2:
            return {"op": "UPDATE", "target_id": best_match.get("id")}
        return {"op": "ADD", "target_id": None}

    @staticmethod
    def _link_relations(new_note: dict, neighbors: list) -> list:
        relations = []
        new_kws = set(new_note.get("keywords", []))
        for nb in neighbors:
            nb_kws = set(nb.get("keywords", []))
            shared = new_kws & nb_kws
            if len(shared) >= 2:
                rel = "extends" if len(shared) >= 3 else "same-topic"
                relations.append({"source": new_note["id"], "target": nb["id"], "relation": rel})
            elif len(shared) >= 1:
                relations.append({"source": new_note["id"], "target": nb["id"], "relation": "semantic"})
        return relations

    @staticmethod
    def _evolve_fields(existing: dict, new_note: dict) -> dict:
        merged_kw = list(dict.fromkeys(list(existing.get("keywords", [])) + list(new_note.get("keywords", []))))
        merged_tags = list(dict.fromkeys(list(existing.get("tags", [])) + list(new_note.get("tags", []))))
        desc = new_note.get("description", "") or existing.get("description", "")
        return {"keywords": merged_kw[:8], "tags": merged_tags[:5], "description": desc}

    @staticmethod
    def _distil_and_answer(query: str, candidates: list) -> dict:
        q_lower = query.lower()
        selected = [c["id"] for c in candidates if any(
            w in c.get("description", "").lower() or w in " ".join(c.get("keywords", [])).lower()
            for w in q_lower.split()
        )]
        if not selected and candidates:
            selected = [candidates[0]["id"]]
        if "dog" in q_lower:
            names = []
            for c in candidates:
                for kw in c.get("keywords", []):
                    if kw.lower() not in ("dog", "pet", "adoption", "animal", "pets", "general") and kw.lower() not in names:
                        names.append(kw)
            return {"selected_ids": selected[:5], "answer": f"You have dogs: {', '.join(names)}." if names else "You have dogs."}
        if "job" in q_lower or "work" in q_lower:
            return {"selected_ids": selected[:5], "answer": "You work as a data scientist at a tech company."}
        if "name" in q_lower:
            return {"selected_ids": selected[:5], "answer": "Your name is Alex."}
        if "pet" in q_lower:
            return {"selected_ids": selected[:5], "answer": "You have dogs and are considering a cat."}
        return {"selected_ids": selected[:5], "answer": "Based on our conversation, I recall relevant details."}

    @staticmethod
    def _extract_block(prompt: str, start_tag: str, end_tag: str = "\n") -> str:
        idx = prompt.find(start_tag)
        if idx == -1:
            return ""
        start = idx + len(start_tag)
        end_nl = prompt.find("\n", start)
        end_stage = prompt.find("ASEM_STAGE=", start)
        candidates = [e for e in [end_nl, end_stage, len(prompt)] if e > start]
        return prompt[start:min(candidates)].strip()

    @staticmethod
    def _safe_json(raw: str, default: Any = None) -> Any:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return default if default is not None else {}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

NOTE_PROMPT = "ASEM_STAGE=NOTE\nCONTENT:{content}"
WRITE_PROMPT = "ASEM_STAGE=WRITE_OP\nCONTENT:{content}\nMEMORY:{memory}"
LINK_PROMPT = "ASEM_STAGE=LINK\nNEW_NOTE:{new_note}\nNEIGHBORS:{neighbors}"
EVOLVE_PROMPT = "ASEM_STAGE=EVOLVE\nEXISTING_NOTE:{existing_note}\nNEW_NOTE:{new_note}"
ANSWER_PROMPT = "ASEM_STAGE=ANSWER\nQUERY:{query}\nCANDIDATES:{candidates}"
BASELINE_PROMPT = "ASEM_STAGE=BASELINE\nQUERY:{query}\nCONTEXT:{context}"
SUMMARY_PROMPT = "ASEM_STAGE=SUMMARY\nQUERY:{query}\nANSWER:{answer}\nREWARD:{reward}"

# ---------------------------------------------------------------------------
# Graph state tracker
# ---------------------------------------------------------------------------

@dataclass
class GraphState:
    """Tracks the evolving knowledge graph across turns."""
    nodes: Dict[str, Dict[str, Any]]  # node_id → {label, keywords, q, op, ...}
    edges: List[Tuple[str, str, str]]  # (source, target, relation)
    turn_history: List[Dict]  # snapshot of each turn's graph changes


class GraphTracker:
    """Maintains the knowledge graph state and records per-turn snapshots."""

    def __init__(self) -> None:
        self.state = GraphState(nodes={}, edges=[], turn_history=[])

    def record_turn(
        self,
        turn_num: int,
        op: str,
        note_id: str,
        keywords: List[str],
        q_value: float,
        content_preview: str,
        links: List[Tuple[str, str, str]],
        target_id: Optional[str] = None,
        is_query: bool = False,
        answer: str = "",
    ) -> Dict:
        """Record a turn's effect on the graph and return a snapshot."""

        # Add/update the node
        prev_node = self.state.nodes.get(note_id)
        self.state.nodes[note_id] = {
            "label": note_id[:8],
            "keywords": keywords,
            "q": q_value,
            "op": op,
            "content": content_preview,
            "turn": turn_num,
        }

        # If UPDATE on existing node, mark target
        if op == "UPDATE" and target_id and target_id in self.state.nodes:
            self.state.nodes[target_id]["op"] = "UPDATED"
            self.state.nodes[target_id]["turn"] = turn_num

        # Add new edges
        for src, tgt, rel in links:
            if (src, tgt) not in [(e[0], e[1]) for e in self.state.edges]:
                self.state.edges.append((src, tgt, rel))

        # Create snapshot
        snapshot = {
            "turn": turn_num,
            "op": op,
            "note_id": note_id,
            "is_query": is_query,
            "answer": answer,
            "content": content_preview,
            "nodes": dict(self.state.nodes),
            "edges": list(self.state.edges),
        }
        self.state.turn_history.append(snapshot)
        return snapshot


# ---------------------------------------------------------------------------
# Graph Renderer
# ---------------------------------------------------------------------------

# Color scheme
COLORS = {
    "ADD":     "#00e5a0",  # green
    "UPDATE":  "#ffc14d",  # amber
    "UPDATED": "#ff8c00",  # dark orange
    "DELETE":  "#ff5370",  # red
    "EXISTING":"#5a7a9e",  # muted blue-grey
}

EDGE_COLORS = {
    "extends":     "#00d4ff",  # cyan
    "contradicts": "#ff5370",  # red
    "causal":      "#ffc14d",  # amber
    "same-topic":  "#7c5cfc",  # purple
    "semantic":    "#5a7a9e",  # muted
    "temporal":    "#00e5a0",  # green
}


def render_graph(
    snapshot: Dict,
    output_path: str,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = 120,
) -> None:
    """Render a single knowledge graph snapshot as a PNG image.

    Parameters
    ----------
    snapshot : dict
        Graph snapshot from GraphTracker.record_turn().
    output_path : str
        File path for the output PNG.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution.
    """
    G = nx.DiGraph()
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]
    turn = snapshot["turn"]
    op = snapshot["op"]
    is_query = snapshot["is_query"]
    answer = snapshot.get("answer", "")
    content = snapshot.get("content", "")

    # Build graph
    for nid, ndata in nodes.items():
        G.add_node(nid, **ndata)
    for src, tgt, rel in edges:
        if src in nodes and tgt in nodes:
            G.add_edge(src, tgt, relation=rel)

    if len(G) == 0:
        # Empty graph — create a placeholder
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "Empty Memory Bank\n(No notes yet)", ha="center", va="center",
                fontsize=16, color="#5a7a9e", fontfamily="monospace", transform=ax.transAxes)
        ax.set_facecolor("#080c14")
        fig.patch.set_facecolor("#080c14")
        _save_and_close(fig, output_path)
        return

    # Layout
    if len(G) <= 3:
        pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)
    elif len(G) <= 10:
        pos = nx.spring_layout(G, k=1.8, iterations=150, seed=42)
    else:
        pos = nx.kamada_kawai_layout(G)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#080c14")
    fig.patch.set_facecolor("#080c14")

    # Node colors and sizes
    node_colors = []
    node_sizes = []
    for nid in G.nodes():
        nd = nodes.get(nid, {})
        n_op = nd.get("op", "EXISTING")
        node_colors.append(COLORS.get(n_op, COLORS["EXISTING"]))
        q = nd.get("q", 0.5)
        node_sizes.append(800 + q * 1200)  # 800–2000 range

    # Edge colors
    edge_colors = []
    edge_widths = []
    for _, _, edata in G.edges(data=True):
        rel = edata.get("relation", "semantic")
        edge_colors.append(EDGE_COLORS.get(rel, EDGE_COLORS["semantic"]))
        edge_widths.append(2.5 if rel in ("extends", "contradicts", "causal") else 1.2)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.1",
        min_source_margin=18,
        min_target_margin=18,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors="#1e2d42",
        linewidths=1.5,
    )

    # Draw labels
    labels = {nid: nd.get("label", nid[:8]) for nid, nd in nodes.items()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=8,
        font_family="monospace",
        font_color="#d4e4f7",
        font_weight="bold",
    )

    # Draw keyword sub-labels below each node
    for nid, (x, y) in pos.items():
        kws = nodes.get(nid, {}).get("keywords", [])[:3]
        if kws:
            kw_text = ", ".join(kws)
            ax.annotate(
                kw_text,
                (x, y - 0.06),
                fontsize=5.5,
                fontfamily="monospace",
                color="#5a7a9e",
                ha="center",
                va="top",
                alpha=0.8,
            )

    # Edge labels (relation types) — show for strong relations
    edge_labels = {}
    for src, tgt, edata in G.edges(data=True):
        rel = edata.get("relation", "")
        if rel in ("extends", "contradicts", "causal"):
            edge_labels[(src, tgt)] = rel
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels, ax=ax,
            font_size=6,
            font_family="monospace",
            font_color="#7c5cfc",
            alpha=0.8,
            label_pos=0.5,
        )

    # Title
    turn_label = f"Turn {turn}"
    op_badge = f"[{op}]"
    if is_query:
        title = f"{turn_label} [QUERY]  |  Answer: {answer[:80]}"
    else:
        title = f"{turn_label} {op_badge}  |  {content[:80]}"

    ax.set_title(title, fontsize=11, fontfamily="monospace", color="#d4e4f7",
                 fontweight="bold", pad=12, loc="left")

    # Subtitle with stats
    stats = f"Notes: {len(nodes)}  |  Links: {len(edges)}  |  Q-range: {min(nd['q'] for nd in nodes.values()):.2f}–{max(nd['q'] for nd in nodes.values()):.2f}"
    ax.text(0.01, 0.99, stats, transform=ax.transAxes,
            fontsize=8, fontfamily="monospace", color="#5a7a9e",
            va="top", ha="left")

    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS["ADD"], label="ADD (new note)"),
        mpatches.Patch(color=COLORS["UPDATE"], label="UPDATE (merged)"),
        mpatches.Patch(color=COLORS["EXISTING"], label="Existing note"),
        Line2D([0], [0], color=EDGE_COLORS["extends"], lw=2, label="extends"),
        Line2D([0], [0], color=EDGE_COLORS["same-topic"], lw=1.5, label="same-topic"),
        Line2D([0], [0], color=EDGE_COLORS["semantic"], lw=1, label="semantic"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=7,
        framealpha=0.3,
        facecolor="#0e1420",
        edgecolor="#1e2d42",
        labelcolor="#d4e4f7",
    )

    ax.axis("off")
    fig.tight_layout(pad=1.5)
    _save_and_close(fig, output_path)


def _save_and_close(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=fig.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_demo_pipeline(db_path: str) -> ASEMPipeline:
    """Build ASEM pipeline with DemoBackend."""
    import tempfile
    backend = DemoBackend(embed_dim=64)
    if db_path == ":memory:":
        db_path = os.path.join(tempfile.gettempdir(), f"asem_graph_{uuid.uuid4().hex[:8]}.sqlite")

    return ASEMPipeline(
        memory_bank=MemoryBank(db_path),
        note_constructor=NoteConstructor(backend=backend, prompt_template=NOTE_PROMPT, q0=0.5),
        memory_manager=MemoryManager(backend=backend, prompt_template=WRITE_PROMPT),
        link_evolver=LinkEvolver(backend=backend, link_prompt_template=LINK_PROMPT,
                                  evolve_prompt_template=EVOLVE_PROMPT, k=5),
        retriever=HybridRetriever(backend=backend, k1=20, k2=5, delta=0.30, lambda_weight=0.40),
        answer_agent=AnswerAgent(backend=backend, prompt_template=ANSWER_PROMPT,
                                  baseline_prompt_template=BASELINE_PROMPT),
        utility_updater=UtilityUpdater(backend=backend, alpha=0.10, q0=0.50,
                                        summary_prompt_template=SUMMARY_PROMPT,
                                        note_constructor=NoteConstructor(backend=backend,
                                                                          prompt_template=NOTE_PROMPT, q0=0.5)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_update(target: Note, note: Note) -> Note:
    return Note(id=target.id, c=note.c, t=note.t,
                K=list(dict.fromkeys(target.K + note.K)),
                G=list(dict.fromkeys(target.G + note.G)),
                X=note.X, e=note.e, L=target.L, z=note.z, q=target.q)


def _describe_op(op: Op, target: Optional[Note]) -> str:
    if op == Op.ADD: return "New information — no related note found."
    if op == Op.UPDATE: return f"Merging into existing note {target.id[:8]}."
    if op == Op.DELETE: return f"Removing contradicted note {target.id[:8]}."
    return "No operation needed."


# ---------------------------------------------------------------------------
# Default conversation
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

def run_graph_simulation(
    conversation: List[Dict],
    output_dir: str = "graph_frames",
    db_path: str = ":memory:",
) -> GraphTracker:
    """Run the full ASEM pipeline and generate per-turn graph images.

    Returns the GraphTracker with full history.
    """
    pipeline = build_demo_pipeline(db_path)
    tracker = GraphTracker()
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("ASEM KNOWLEDGE GRAPH VISUALIZER")
    print(f"{'='*60}")
    print(f"Turns: {len(conversation)}  |  Output: {output_dir}/")
    print(f"{'='*60}\n")

    reward = 0.8  # default

    for i, turn in enumerate(conversation):
        content = turn["content"]
        turn_type = turn.get("type", "fact")
        reward = turn.get("reward", reward)
        turn_num = i + 1

        # ── Stage 1: Note Construction ──
        note = pipeline.note_constructor.build(content, datetime.now(UTC))

        if turn_type == "fact":
            # ── Stage 2: Memory Manager ──
            e_new = pipeline.note_constructor.backend.embed(content)
            existing = pipeline.memory_bank.ann_search(e_new, k=5)
            if not existing:
                existing = pipeline.memory_bank.list_notes()[:5]

            op, target = pipeline.memory_manager.select_op(content, existing)

            # Execute operation
            if op == Op.ADD:
                pipeline.memory_bank.add(note)
                pipeline.link_evolver.link_and_evolve(note, pipeline.memory_bank)
            elif op == Op.UPDATE and target:
                updated = _merge_update(target, note)
                pipeline.memory_bank.add(updated)
                pipeline.link_evolver.link_and_evolve(updated, pipeline.memory_bank)
                note = updated  # use updated note for graph
            elif op == Op.DELETE and target:
                pipeline.memory_bank.delete(target.id)

            # Collect links from the bank via graph API
            graph = pipeline.memory_bank.get_link_graph()
            links = [(e["source_full"], e["target_full"], e["relation"]) for e in graph["edges"]]
            stats = pipeline.memory_bank.get_link_statistics()

            # Record in graph tracker
            tracker.record_turn(
                turn_num=turn_num,
                op=op.value,
                note_id=note.id,
                keywords=note.K,
                q_value=note.q,
                content_preview=content,
                links=links,
                target_id=target.id if target else None,
            )

            link_str = f"🔗 {len(links)} links" if links else "no links"
            print(f"  [{turn_num:2d}] {op.value:6s} | {note.id[:8]} | K={note.K[:3]} | {link_str}")
            if stats["total_nodes"] > 1:
                print(f"       ↳ graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges, "
                      f"density={stats['density']:.4f}, {stats['num_components']} components")

        else:
            # ── Query path ──
            candidates = pipeline.retriever.retrieve(content, pipeline.memory_bank)
            used_notes, answer = pipeline.answer_agent.distil_and_answer(content, candidates)

            # Update utilities
            pipeline.utility_updater.update(reward, used_notes, pipeline.memory_bank, content, answer)

            # Refresh Q-values from bank
            for n in pipeline.memory_bank.list_notes():
                if n.id in tracker.state.nodes:
                    tracker.state.nodes[n.id]["q"] = n.q

            # Collect current links via graph API
            graph = pipeline.memory_bank.get_link_graph()
            links = [(e["source_full"], e["target_full"], e["relation"]) for e in graph["edges"]]
            stats = pipeline.memory_bank.get_link_statistics()

            tracker.record_turn(
                turn_num=turn_num,
                op="QUERY",
                note_id=note.id,
                keywords=[],
                q_value=0.5,
                content_preview=content,
                links=links,
                is_query=True,
                answer=answer,
            )

            print(f"  [{turn_num:2d}] QUERY  | 💬 {answer[:70]}")

        # ── Render graph image ──
        snapshot = tracker.state.turn_history[-1]
        frame_name = f"turn_{turn_num:02d}_{turn_type}_{snapshot['op']}.png"
        frame_path = os.path.join(output_dir, frame_name)
        render_graph(snapshot, frame_path)
        print(f"       ↳ saved: {frame_name}")

    # Final summary
    final_notes = pipeline.memory_bank.list_notes()
    final_stats = pipeline.memory_bank.get_link_statistics()
    print(f"\n{'='*60}")
    print(f"COMPLETE — {len(conversation)} turns processed")
    print(f"  Final bank: {len(final_notes)} notes")
    print(f"  Graph: {final_stats['total_nodes']} nodes, {final_stats['total_edges']} edges, "
          f"density={final_stats['density']:.4f}, {final_stats['num_components']} components")
    if final_stats['total_edges'] > 0:
        print(f"  Max degree: {final_stats['max_degree']}, Avg degree: {final_stats['avg_degree']:.2f}")
    print(f"  Frames saved: {output_dir}/turn_*.png")
    print(f"{'='*60}")

    return tracker


# ---------------------------------------------------------------------------
# HTML Animation Generator
# ---------------------------------------------------------------------------

def generate_html_animation(frames_dir: str, output_path: str, num_frames: int) -> None:
    """Generate an auto-advancing HTML slideshow of graph frames."""
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))

    slides = "\n".join(
        f'''      <div class="slide {"active" if i == 0 else ""}" id="slide{i}">
        <img src="{os.path.join(frames_dir, f)}" alt="Turn {i+1}">
        <div class="caption">Turn {i+1} — {f.replace(".png", "").replace("_", " ")}</div>
      </div>'''
        for i, f in enumerate(frame_files)
    )

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ASEM Knowledge Graph Evolution</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #080c14; color: #d4e4f7;
    font-family: 'Space Mono', monospace;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 16px;
  }}
  h1 {{
    font-family: 'Syne', sans-serif; font-size: 22px;
    color: #00d4ff; margin-bottom: 4px;
  }}
  .subtitle {{ font-size: 10px; color: #5a7a9e; letter-spacing: 1px; margin-bottom: 16px; }}
  .slideshow {{
    position: relative; max-width: 1000px; width: 100%;
    border: 1px solid #1e2d42; border-radius: 12px; overflow: hidden;
    background: #0e1420;
  }}
  .slide {{ display: none; text-align: center; }}
  .slide.active {{ display: block; }}
  .slide img {{ width: 100%; height: auto; border-radius: 0; }}
  .caption {{
    padding: 10px; font-size: 11px; color: #5a7a9e;
    background: #0e1420; border-top: 1px solid #1e2d42;
  }}
  .controls {{
    display: flex; gap: 12px; align-items: center;
    margin-top: 16px; font-size: 12px;
  }}
  button {{
    font-family: 'Space Mono', monospace; font-size: 10px;
    padding: 6px 14px; border-radius: 6px; cursor: pointer;
    border: 1px solid #00d4ff; color: #00d4ff;
    background: rgba(0,212,255,.08); transition: all .2s;
  }}
  button:hover {{ background: rgba(0,212,255,.18); }}
  button.auto {{ border-color: #00e5a0; color: #00e5a0; background: rgba(0,229,160,.08); }}
  button.auto:hover {{ background: rgba(0,229,160,.18); }}
  button.auto.running {{ border-color: #ff5370; color: #ff5370; background: rgba(255,83,112,.1); }}
  .counter {{ color: #7c5cfc; font-weight: bold; min-width: 60px; text-align: center; }}
</style>
</head>
<body>
  <h1>ASEM · Knowledge Graph Evolution</h1>
  <div class="subtitle">{num_frames} TURNS · FULL PIPELINE TRACE</div>
  <div class="slideshow" id="slideshow">
{slides}
  </div>
  <div class="controls">
    <button onclick="prevSlide()">◀ PREV</button>
    <span class="counter" id="counter">1 / {num_frames}</span>
    <button onclick="nextSlide()">NEXT ▶</button>
    <button class="auto" id="autoBtn" onclick="toggleAuto()">▶ AUTO PLAY</button>
    <span style="color:#5a7a9e;font-size:10px">every 1.5s</span>
  </div>

<script>
let current = 0;
const total = {num_frames};
let autoInterval = null;

function showSlide(n) {{
  document.querySelectorAll('.slide').forEach((s,i) => s.classList.toggle('active', i===n));
  document.getElementById('counter').textContent = (n+1) + ' / ' + total;
  current = n;
}}

function nextSlide() {{ showSlide((current + 1) % total); }}
function prevSlide() {{ showSlide((current - 1 + total) % total); }}

function toggleAuto() {{
  const btn = document.getElementById('autoBtn');
  if (autoInterval) {{
    clearInterval(autoInterval); autoInterval = null;
    btn.textContent = '▶ AUTO PLAY'; btn.classList.remove('running');
  }} else {{
    autoInterval = setInterval(nextSlide, 1500);
    btn.textContent = '⏸ PAUSE'; btn.classList.add('running');
  }}
}}

document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowRight') nextSlide();
  if (e.key === 'ArrowLeft') prevSlide();
  if (e.key === ' ') {{ e.preventDefault(); toggleAuto(); }}
}});
</script>
</body>
</html>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✅ HTML animation saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASEM Knowledge Graph Visualizer — per-turn graph images"
    )
    parser.add_argument("--input", "-i", type=str, default=None,
                       help="JSON conversation file")
    parser.add_argument("--output", "-o", type=str, default="graph_frames",
                       help="Output directory for graph images")
    parser.add_argument("--html", action="store_true",
                       help="Generate auto-advancing HTML slideshow")
    parser.add_argument("--gif", action="store_true",
                       help="Generate animated GIF from frames (requires Pillow)")
    parser.add_argument("--db", type=str, default=":memory:",
                       help="SQLite path for memory bank")
    args = parser.parse_args()

    # Load conversation
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            conversation = json.load(f)
    else:
        conversation = DEFAULT_CONVERSATION

    # Run simulation → generates PNG frames
    tracker = run_graph_simulation(conversation, args.output, args.db)

    # Generate HTML slideshow
    if args.html:
        html_path = os.path.join(args.output, "graph_animation.html")
        generate_html_animation(args.output, html_path, len(conversation))

    # Generate animated GIF
    if args.gif:
        try:
            from PIL import Image
            frame_files = sorted(
                os.path.join(args.output, f)
                for f in os.listdir(args.output)
                if f.endswith(".png")
            )
            if frame_files:
                frames = [Image.open(f) for f in frame_files]
                gif_path = os.path.join(args.output, "graph_evolution.gif")
                frames[0].save(
                    gif_path, save_all=True, append_images=frames[1:],
                    duration=1200, loop=0,
                )
                print(f"✅ Animated GIF saved: {gif_path}")
        except ImportError:
            print("⚠️  Pillow not installed — skipping GIF. Install: pip install Pillow")


if __name__ == "__main__":
    main()
