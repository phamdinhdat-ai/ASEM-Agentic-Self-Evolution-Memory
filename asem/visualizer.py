"""Memory bank graph visualization.

Extracts a knowledge graph from the ASEM memory bank (SQLite) and renders
interactive HTML via pyvis or static PNG via matplotlib.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .logging_utils import get_logger
from .memory_bank import MemoryBank

_log = get_logger("visualizer")

# ── Colour palette for tag groups ──────────────────────────────────────
_TAG_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
]


def _hex_to_rgba(hex_color: str, alpha: float = 0.9) -> str:
    """Convert hex colour to rgba() string for pyvis."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


class MemoryGraphBuilder:
    """Extract a networkx graph from a MemoryBank instance."""

    def __init__(self, memory_bank: MemoryBank) -> None:
        import networkx as nx

        self._bank = memory_bank
        self.graph: nx.Graph = nx.Graph()
        self._tag_color_map: Dict[str, str] = {}
        self._cosine_cache: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_graph(self) -> "nx.Graph":  # type: ignore[name-defined]  # noqa: F821
        """Populate the networkx graph from the memory bank.

        Returns the graph (also available as ``self.graph``).
        """
        import networkx as nx

        notes = self._bank.list_notes()
        if not notes:
            _log.warning("Memory bank is empty — nothing to visualize")
            return self.graph

        for note in notes:
            tag = note.G[0] if note.G else "untagged"
            color = self._tag_color_for(tag)
            label = (note.X[:60] + "…") if len(note.X) > 60 else note.X
            if not label.strip():
                label = (note.c[:60] + "…") if len(note.c) > 60 else note.c

            self.graph.add_node(
                note.id,
                label=label,
                title=self._build_tooltip(note),
                q=note.q,
                group=tag,
                color=color,
                size=max(10, min(40, 15 + note.q * 25)),
                border_width=1 + min(len(note.L), 10) * 0.5,
                content=note.c,
                keywords=note.K,
                tags=note.G,
                description=note.X,
                num_links=len(note.L),
                timestamp=note.t.isoformat(),
            )

        # Edges from L fields (bidirectional, deduplicate)
        edge_set: set = set()
        for note in notes:
            for target_id in note.L:
                edge_key = tuple(sorted([note.id, target_id]))
                if edge_key in edge_set:
                    continue
                edge_set.add(edge_key)
                sim = self._cosine_sim(note.id, target_id)
                self.graph.add_edge(
                    note.id,
                    target_id,
                    weight=float(sim),
                    title=f"cosine similarity: {sim:.3f}",
                )

        _log.info(
            "Graph built | nodes={}  edges={}  tags={}",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            len(self._tag_color_map),
        )
        return self.graph

    # ------------------------------------------------------------------
    # Edge-type inference (heuristic — no LLM required)
    # ------------------------------------------------------------------

    def infer_edge_types(self) -> None:
        """Colour edges by inferred relationship type using heuristics."""
        if self.graph.number_of_edges() == 0:
            return

        type_colors = {
            "same-topic": "#4e79a7",
            "extends": "#59a14f",
            "contradicts": "#e15759",
            "semantic": "#bab0ac",
            "linked": "#d0d0d0",
        }

        for u, v in self.graph.edges():
            inferred = self._infer_relation(u, v)
            self.graph.edges[u, v]["relation"] = inferred
            self.graph.edges[u, v]["color"] = type_colors.get(inferred, "#d0d0d0")

        _log.info("Edge types inferred for {} edges", self.graph.number_of_edges())

    def _infer_relation(self, node_u: str, node_v: str) -> str:
        u_data = self.graph.nodes[node_u]
        v_data = self.graph.nodes[node_v]

        u_tags = set(u_data.get("tags", []))
        v_tags = set(v_data.get("tags", []))
        u_kw = set(u_data.get("keywords", []))
        v_kw = set(v_data.get("keywords", []))

        # Tag overlap ≥ 50 % -> same-topic
        if u_tags and v_tags:
            overlap = len(u_tags & v_tags) / min(len(u_tags), len(v_tags))
            if overlap >= 0.5:
                return "same-topic"

        # Keyword overlap ≥ 50 % -> extends
        if u_kw and v_kw:
            overlap = len(u_kw & v_kw) / min(len(u_kw), len(v_kw))
            if overlap >= 0.5:
                return "extends"

        # Q-value difference > 0.3 -> contradicts
        q_diff = abs(u_data.get("q", 0.5) - v_data.get("q", 0.5))
        if q_diff > 0.3:
            return "contradicts"

        # Cosine similarity > 0.7 -> semantic
        sim = self._cosine_sim(node_u, node_v)
        if sim > 0.7:
            return "semantic"

        return "linked"

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def compute_metrics(self) -> Dict[str, Any]:
        """Compute centrality / clustering metrics for the graph."""
        import networkx as nx

        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}

        try:
            degree_cent = nx.degree_centrality(self.graph)
        except Exception:
            degree_cent = {}

        try:
            clustering = nx.average_clustering(self.graph)
        except Exception:
            clustering = 0.0

        try:
            components = list(nx.connected_components(self.graph))
            num_components = len(components)
            largest_comp = max(len(c) for c in components) if components else 0
        except Exception:
            num_components = 0
            largest_comp = 0

        metrics = {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_clustering": float(clustering),
            "num_connected_components": num_components,
            "largest_component_size": largest_comp,
            "top_degree_nodes": sorted(
                [(n, float(d)) for n, d in degree_cent.items()],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }
        return metrics

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tag_color_for(self, tag: str) -> str:
        if tag not in self._tag_color_map:
            idx = len(self._tag_color_map) % len(_TAG_COLORS)
            self._tag_color_map[tag] = _TAG_COLORS[idx]
        return self._tag_color_map[tag]

    @staticmethod
    def _build_tooltip(note: Any) -> str:
        """Build a rich HTML tooltip for a note node."""
        c_preview = note.c[:120] + "…" if len(note.c) > 120 else note.c
        kw = ", ".join(note.K[:8]) if note.K else "—"
        tags = ", ".join(note.G[:6]) if note.G else "—"

        return (
            f"<div style='max-width:320px;font-family:sans-serif;font-size:12px;'>"
            f"<b>ID:</b> {note.id[:12]}…<br>"
            f"<b>Content:</b> {_escape_html(c_preview)}<br>"
            f"<b>Keywords:</b> {_escape_html(kw)}<br>"
            f"<b>Tags:</b> {_escape_html(tags)}<br>"
            f"<b>Description:</b> {_escape_html(note.X)}<br>"
            f"<b>Q-value:</b> {note.q:.3f}<br>"
            f"<b>Links:</b> {len(note.L)}<br>"
            f"<b>Time:</b> {note.t.isoformat()[:19]}"
            f"</div>"
        )

    def _cosine_sim(self, id_a: str, id_b: str) -> float:
        cache_key = tuple(sorted([id_a, id_b]))
        if cache_key in self._cosine_cache:
            return self._cosine_cache[cache_key]

        note_a = self._bank.get_note(id_a)
        note_b = self._bank.get_note(id_b)
        if note_a is None or note_b is None:
            self._cosine_cache[cache_key] = 0.0
            return 0.0

        a = np.asarray(note_a.e, dtype=float)
        b = np.asarray(note_b.e, dtype=float)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        sim = float(np.dot(a, b) / denom) if denom != 0 else 0.0
        self._cosine_cache[cache_key] = sim
        return sim


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ── Rendering ───────────────────────────────────────────────────────────


def render_interactive(
    graph: "nx.Graph",  # type: ignore[name-defined]  # noqa: F821
    output_path: str,
    title: str = "ASEM Memory Graph",
    physics_enabled: bool = True,
    height: str = "750px",
    width: str = "100%",
) -> None:
    """Render a networkx graph as an interactive HTML file using pyvis."""
    from pyvis.network import Network

    net = Network(height=height, width=width, directed=False, notebook=False)
    net.set_options(_pyvis_options(physics_enabled))

    # Transfer nodes
    for node_id, data in graph.nodes(data=True):
        color = data.get("color", "#97c2fc")
        if color.startswith("#"):
            color = _hex_to_rgba(color, 0.9)
        net.add_node(
            node_id,
            label=data.get("label", node_id[:8]),
            title=data.get("title", ""),
            color=color,
            size=data.get("size", 20),
            borderWidth=data.get("border_width", 1),
        )

    # Transfer edges
    for u, v, data in graph.edges(data=True):
        edge_color = data.get("color", "#d0d0d0")
        if edge_color.startswith("#"):
            edge_color = _hex_to_rgba(edge_color, 0.6)
        net.add_edge(
            u, v,
            title=data.get("title", ""),
            value=data.get("weight", 0.5),
            color=edge_color,
            width=max(1, data.get("weight", 0.5) * 5),
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    net.save_graph(output_path)

    # Inject legend after save
    _inject_legend(output_path, graph)

    _log.success("Interactive graph saved -> {}", output_path)


def render_static(
    graph: "nx.Graph",  # type: ignore[name-defined]  # noqa: F821
    output_path: str,
    title: str = "ASEM Memory Graph",
) -> None:
    """Render a static PNG using matplotlib + networkx layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx

    fig, ax = plt.subplots(figsize=(16, 12))

    # Spring layout
    pos = nx.spring_layout(graph, k=2.0, iterations=50, seed=42)

    # Node attributes
    node_colors = [
        graph.nodes[n].get("color", "#97c2fc") for n in graph.nodes()
    ]
    node_sizes = [
        graph.nodes[n].get("size", 20) * 3 for n in graph.nodes()
    ]

    # Draw nodes
    nx.draw_networkx_nodes(
        graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        ax=ax,
    )

    # Draw edges
    edge_colors = [
        graph.edges[u, v].get("color", "#d0d0d0") for u, v in graph.edges()
    ]
    edge_widths = [
        max(0.5, graph.edges[u, v].get("weight", 0.5) * 3)
        for u, v in graph.edges()
    ]
    nx.draw_networkx_edges(
        graph, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.5,
        ax=ax,
    )

    # Draw labels (shortened)
    labels = {
        n: graph.nodes[n].get("label", n[:8])[:30] for n in graph.nodes()
    }
    nx.draw_networkx_labels(graph, pos, labels, font_size=7, ax=ax)

    ax.set_title(title, fontsize=14)
    ax.axis("off")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _log.success("Static PNG saved -> {}", output_path)


def _inject_legend(html_path: str, graph: "nx.Graph") -> None:  # type: ignore[name-defined]  # noqa: F821
    """Post-process pyvis HTML to inject the legend overlay."""
    try:
        with open(html_path, "r", encoding="utf-8") as fh:
            html = fh.read()
    except Exception:
        return

    # Collect tag->colour mapping from nodes
    tag_colors: Dict[str, str] = {}
    for _, data in graph.nodes(data=True):
        group = data.get("group", "untagged")
        color = data.get("color", "#97c2fc")
        if group not in tag_colors:
            tag_colors[group] = color

    # Collect edge type->colour mapping
    edge_type_colors: Dict[str, str] = {}
    for _, _, data in graph.edges(data=True):
        rel = data.get("relation", "linked")
        color = data.get("color", "#d0d0d0")
        if rel not in edge_type_colors:
            edge_type_colors[rel] = color

    legend = (
        '<div id="asem-legend" style="position:absolute;top:10px;left:10px;'
        "background:rgba(255,255,255,0.92);padding:10px 14px;"
        "border-radius:8px;font-family:sans-serif;font-size:12px;"
        "box-shadow:0 2px 8px rgba(0,0,0,0.15);max-height:80%;"
        'overflow-y:auto;z-index:999;">'
        '<b style="font-size:14px;">Legend</b><br><br>'
        '<b>Tags (nodes):</b><br>'
    )
    for tag, color in sorted(tag_colors.items()):
        legend += (
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:50%;margin-right:6px;"></span>'
            f'{_escape_html(tag)}<br>'
        )

    legend += '<br><b>Relations (edges):</b><br>'
    for rel, color in sorted(edge_type_colors.items()):
        legend += (
            f'<span style="display:inline-block;width:20px;height:3px;'
            f'background:{color};margin-right:6px;vertical-align:middle;">'
            f'</span>{_escape_html(rel)}<br>'
        )
    legend += "</div>"

    html = html.replace("<body>", "<body>\n" + legend, 1)

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def _pyvis_options(physics_enabled: bool = True) -> str:
    """Return pyvis options JSON string with tuned physics."""
    phys = (
        """
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 150,
                "springConstant": 0.08,
                "damping": 0.4
            },
            "maxVelocity": 30,
            "minVelocity": 0.5,
            "solver": "forceAtlas2Based",
            "stabilization": {
                "enabled": true,
                "iterations": 200,
                "updateInterval": 25
            }
        },
        """
        if physics_enabled
        else """
        "physics": {"enabled": false},
        """
    )

    return (
        "{"
        + phys
        + """"interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true
        },
        "nodes": {
            "font": {"size": 12, "face": "sans-serif"},
            "borderWidthSelected": 3
        },
        "edges": {
            "smooth": {"type": "continuous", "roundness": 0.3},
            "font": {"size": 10, "align": "middle"}
        }"""
        + "}"
    )


# ── Convenience entry point ─────────────────────────────────────────────


def visualize_bank(
    db_path: str,
    output: str = "memory_graph.html",
    fmt: str = "html",
    title: str = "ASEM Memory Graph",
    physics: bool = True,
    infer_types: bool = True,
    stats: bool = False,
) -> Dict[str, Any]:
    """One-shot: load bank -> build graph -> render.

    Args:
        db_path: Path to the SQLite memory bank.
        output: Output file path.
        fmt: ``"html"`` (interactive) or ``"png"`` (static).
        title: Chart title.
        physics: Enable pyvis force-directed physics.
        infer_types: Run heuristic edge-type inference.
        stats: Print graph metrics to stdout.

    Returns:
        Dict of graph metrics.
    """
    bank = MemoryBank.load(db_path)

    builder = MemoryGraphBuilder(bank)
    graph = builder.build_graph()

    if graph.number_of_nodes() == 0:
        _log.warning("No notes in memory bank — nothing to render")
        return {"nodes": 0, "edges": 0}

    if infer_types:
        builder.infer_edge_types()

    if fmt == "html":
        render_interactive(graph, output, title=title, physics_enabled=physics)
        # Legend is already injected by render_interactive
    elif fmt == "png":
        render_static(graph, output, title=title)
    else:
        raise ValueError(f"Unknown format: {fmt!r}. Use 'html' or 'png'.")

    metrics = builder.compute_metrics()
    if stats:
        print(_format_metrics(metrics))
    return metrics


def _format_metrics(metrics: Dict[str, Any]) -> str:
    lines = [
        "=" * 48,
        "  ASEM Memory Graph — Metrics",
        "=" * 48,
        f"  Nodes:                   {metrics['nodes']}",
        f"  Edges:                   {metrics['edges']}",
        f"  Density:                 {metrics.get('density', 0):.4f}",
        f"  Avg. clustering coeff:  {metrics.get('avg_clustering', 0):.4f}",
        f"  Connected components:    {metrics.get('num_connected_components', 0)}",
        f"  Largest component size:  {metrics.get('largest_component_size', 0)}",
    ]
    top = metrics.get("top_degree_nodes", [])
    if top:
        lines.append("  Top nodes (degree centrality):")
        for node_id, cent in top:
            lines.append(f"    {node_id[:12]}…  {cent:.4f}")
    lines.append("=" * 48)
    return "\n".join(lines)
