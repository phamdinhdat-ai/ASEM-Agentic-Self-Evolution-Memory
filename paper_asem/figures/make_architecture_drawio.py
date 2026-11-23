"""Generate the ASEM/CARE architecture as a .drawio file plus a self-contained HTML viewer.

Follows the drawio-diagram skill (~/.agents/skills/drawio-diagram): skill palette, legend row
at the top, containers for groups, orthogonal edges, infrastructure bar at the bottom,
sticky-note annotations for evidence. Status is encoded by element type:
    Capability (green)          evaluated end to end
    Tech Preview (purple)       implemented and tested; QA evaluation running or retrieval-only
    Tech Preview, dashed        specified, not in the pipeline
Numbers in annotations come from analysis/*.md and analysis/*.json of 17/11/2026.

    python paper_asem/figures/make_architecture_drawio.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from xml.sax.saxutils import quoteattr

OUT = Path(__file__).resolve().parent
NAME = "asem_architecture_v4"

STYLE = {
    "container": "rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;verticalAlign=top;fontSize=12;fontStyle=1;",
    "capability": "rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=10;verticalAlign=top;",
    "external": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;fontColor=#333333;strokeColor=#666666;verticalAlign=top;fontSize=10;",
    "preview": "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;verticalAlign=top;",
    "spec": "rounded=1;whiteSpace=wrap;html=1;fillColor=#e1d5e7;strokeColor=#9673a6;fontSize=10;verticalAlign=top;dashed=1;dashPattern=4 4;opacity=80;",
    "process": "rounded=0;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;fontSize=10;",
    "outcome": "rounded=0;whiteSpace=wrap;html=1;fillColor=#008a00;strokeColor=#005700;fontColor=#ffffff;fontSize=10;",
    "note": ("shape=note;whiteSpace=wrap;html=1;backgroundOutline=1;fontColor=#000000;darkOpacity=0.05;"
             "fillColor=#FFF9B2;strokeColor=none;fillStyle=solid;direction=west;gradientDirection=north;"
             "gradientColor=#FFF2A1;shadow=1;size=20;pointerEvents=1;fontSize=10;align=left;spacingLeft=6;verticalAlign=top;"),
    "infra": "rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;fontColor=#333333;fontSize=10;",
    "title": "text;html=1;align=left;verticalAlign=middle;fontSize=16;fontStyle=1;strokeColor=none;fillColor=none;",
}
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;fontSize=9;endArrow=block;endFill=1;"

cells: list[str] = []
boxes: dict[str, tuple[int, int, int, int, str]] = {}


def label(title: str, body: str = "") -> str:
    return f"<b>{title}</b>" + (f"<br><font style=\"font-size: 9px;\">{body}</font>" if body else "")


def vertex(cid: str, value: str, kind: str, x: int, y: int, w: int, h: int, extra: str = "") -> None:
    boxes[cid] = (x, y, w, h, kind)
    cells.append(f'<mxCell id="{cid}" value={quoteattr(value)} style="{STYLE[kind]}{extra}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" /></mxCell>')


def edge(cid: str, src: str, dst: str, text: str = "", dashed: bool = False, exit_: str = "", entry: str = "") -> None:
    style = EDGE + ("dashed=1;" if dashed else "") + exit_ + entry
    cells.append(f'<mxCell id="{cid}" value={quoteattr(text)} style="{style}" edge="1" parent="1" '
                 f'source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry" /></mxCell>')


# ---- title and legend (top) ----
vertex("title", "ASEM / CARE architecture (draft v4, 17 Nov 2026): what is built and what is evaluated",
       "title", 20, 6, 1100, 26)
legend = [("capability", "Evaluated end to end"), ("preview", "Implemented, evaluation running"),
          ("spec", "Specified only"), ("process", "Code check / gate"), ("external", "Model / infra"),
          ("outcome", "Output"), ("note", "Evidence")]
for i, (kind, text) in enumerate(legend):
    vertex(f"key-{i}", f"<span>{text}</span>", kind, 20 + i * 190, 40, 180, 22,
           "verticalAlign=middle;fontStyle=2;" if kind != "note" else "verticalAlign=middle;")

# ---- write path ----
vertex("wpath", "Write path (ASEM-Evolve)", "container", 20, 80, 1560, 180)
vertex("stream", label("Session stream", "turns s=(id, session, date, speaker, text)"), "external", 40, 120, 190, 70)
vertex("archive", label("Archive A<sub>t</sub>", "append-only original turns<br>the only citable source"), "capability", 270, 115, 210, 80)
vertex("ops", label("Guarded operators", "ADD / UPDATE / LINK<br>quote &#8849; turn, value &#8849; quote,<br>dates parsed and ordered; 15 unit tests"),
       "preview", 520, 110, 270, 95)
vertex("derived", label("Derived layer D<sub>t</sub>", "versioned notes (key, value, sources, valid-from)<br>supersede chains; replayable from A<sub>t</sub>"),
       "preview", 830, 110, 280, 95)
vertex("more", label("SPLIT / MERGE / PRUNE / REORG", "triggered by label-free read-path signals"), "spec", 1150, 115, 230, 80)
vertex("n-write", "<b>Proved (Prop. 1-3)</b><br>every note grounded in A<sub>t</sub><br>one current version per key<br>layer rebuildable from the log",
       "note", 1400, 100, 165, 110)
edge("e-s-a", "stream", "archive")
edge("e-a-o", "archive", "ops", dashed=True)
edge("e-o-d", "ops", "derived", dashed=True)
edge("e-m-o", "more", "ops", dashed=True, exit_="exitX=0.5;exitY=1;exitDx=0;exitDy=0;",
     entry="entryX=0.5;entryY=1;entryDx=0;entryDy=0;")

# ---- layered memory graph ----
vertex("gpath", "Layered memory graph G<sub>t</sub> (built per history, no LLM)", "container", 20, 290, 1560, 250)
vertex("enc", label("Qwen3-Embedding-0.6B", "GPU, fp16, last-token pooling"), "external", 40, 370, 200, 65)
vertex("l3", label("L3 communities", "Louvain, seeded"), "preview", 290, 320, 260, 42)
vertex("l2", label("L2 sessions"), "preview", 290, 372, 260, 34)
vertex("l1", label("L1 entities", "capitalized, 2-5% of turns"), "preview", 290, 416, 260, 42)
vertex("l0", label("L0 turns", "next edges, semantic 5-NN"), "preview", 290, 468, 260, 42)
vertex("ppr", label("Typed PPR influence",
                    "&#960; = &#945;s<sub>q</sub> + (1-&#945;)&#960;P, 5 dense seeds<br>rank = 0.7 z(dense) + 0.3 z(&#960;)<br>"
                    "mass beyond h hops &#8804; (1-&#945;)<sup>h+1</sup>"), "preview", 610, 330, 330, 100)
vertex("notesL", label("Notes of D<sub>t</sub> as graph nodes", "cite edges to their source turns"), "spec", 610, 450, 330, 60)
vertex("n-graph", "<b>Dev recall, LoCoMo (1,531 q)</b><br>top-8: dense 0.617, graph 0.618<br>top-18: dense 0.725, graph 0.736<br>"
                  "community round-robin: 0.415<br>per history: ~588 turns, 93 entities,<br>27 sessions, 10 communities",
       "note", 990, 320, 300, 150)
edge("e-a-enc", "archive", "enc", dashed=True, exit_="exitX=0.5;exitY=1;exitDx=0;exitDy=0;",
     entry="entryX=0.5;entryY=0;entryDx=0;entryDy=0;")
edge("e-enc-l0", "enc", "l0")
edge("e-enc-l1", "enc", "l1")
edge("e-l2-ppr", "l2", "ppr", dashed=True)
edge("e-n-l0", "notesL", "l0", dashed=True)
edge("e-d-n", "derived", "notesL", dashed=True, exit_="exitX=0.5;exitY=1;exitDx=0;exitDy=0;",
     entry="entryX=1;entryY=0.5;entryDx=0;entryDy=0;")

# ---- read path ----
vertex("rpath", "Read path (CARE with dense seeds and graph-aware repair; at most two reader calls)", "container", 20, 570, 1560, 320)
vertex("agg", label("Aggregation over notes", "count / sum / difference by code"), "spec", 40, 615, 200, 70)
vertex("expand", label("Repair expansion",
                       "lexical: BM25, stemmed, neighbours, searches Q<br>&#8746; dense top-6 (hybrid arm)<br>&#8746; graph-PPR top-6 (graph arm)"),
       "preview", 300, 605, 300, 95)
vertex("reader2", label("Reader R<sub>&#952;</sub>", "once more"), "capability", 650, 620, 150, 60)
vertex("out", label("Answer or abstention", "citations projected to archive turns"), "outcome", 860, 615, 200, 70)
vertex("q", label("Question q"), "external", 40, 790, 120, 55)
vertex("seed", label("First bundle", "BM25 or dense top-8, &#8804; 10,000 chars"), "capability", 190, 785, 190, 65)
vertex("reader", label("Reader R<sub>&#952;</sub> (Qwen3-8B)", "JSON: answer, cited IDs, searches"), "capability", 410, 785, 210, 65)
vertex("check", label("Code checks", "citation in bundle; speaker / event binding"), "process", 650, 785, 200, 65)
vertex("gate", label("Gate &#964;(q)", "invalid, or abstained with searches or risk cue"), "process", 880, 785, 190, 65)
vertex("n-read", "<b>LoCoMo, 1,986 q, same reader</b><br>CARE (BM25 + lexical repair): 0.5256<br>dense first pass: 0.4730 (+0.014 vs BM25, 9/10 conv.)<br>"
                 "dense + lexical repair: 0.5169 (-0.009, CI includes 0)<br>repair: +0.067 but -46 correct refusals<br>"
                 "6 verifiers: best &#945;/&#946; = 2.31 &lt; 3.89 needed; all lose<br><i>factorial with hybrid / graph arms: running</i>",
       "note", 1110, 600, 450, 160)
edge("e-q-s", "q", "seed")
edge("e-s-r", "seed", "reader")
edge("e-r-c", "reader", "check")
edge("e-c-g", "check", "gate")
edge("e-g-o", "gate", "out", "&#964; = 0", exit_="exitX=0.5;exitY=0;exitDx=0;exitDy=0;",
     entry="entryX=0.5;entryY=1;entryDx=0;entryDy=0;")
edge("e-g-x", "gate", "expand", "&#964; = 1", exit_="exitX=0;exitY=0;exitDx=0;exitDy=0;",
     entry="entryX=0.5;entryY=1;entryDx=0;entryDy=0;")
edge("e-x-r2", "expand", "reader2")
edge("e-r2-o", "reader2", "out")
edge("e-agg-x", "agg", "expand", dashed=True)
edge("e-ppr-x", "ppr", "expand", dashed=True, exit_="exitX=0.25;exitY=1;exitDx=0;exitDy=0;",
     entry="entryX=0.75;entryY=0;entryDx=0;entryDy=0;")

# ---- infrastructure bar ----
vertex("infra", "<b>Infrastructure</b>: Kaggle 2&#215;T4 &#183; vLLM (fp16, 8,192-token window, greedy, max-num-seqs 1) &#183; "
                "Qwen3-8B and Qwen3-Embedding-0.6B at pinned revisions &#183; open weights only &#183; reruns reproduce saved answers 98/98",
       "infra", 20, 910, 1560, 36)

model = ('<mxGraphModel dx="1600" dy="1000" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
         'fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="970" math="0" shadow="0"><root>'
         '<mxCell id="0" /><mxCell id="1" parent="0" />' + "".join(cells) + "</root></mxGraphModel>")
doc = f'<mxfile host="Electron" version="28.0.6"><diagram name="ASEM architecture v4" id="asem-arch-v4">{model}</diagram></mxfile>\n'


def overlaps() -> list[tuple[str, str]]:
    """Leaf boxes that overlap (containers and the title are allowed to contain others)."""
    leaves = {k: v for k, v in boxes.items() if v[4] not in ("container", "title")}
    out = []
    names = sorted(leaves)
    for i, a in enumerate(names):
        ax, ay, aw, ah, _ = leaves[a]
        for b in names[i + 1:]:
            bx, by, bw, bh, _ = leaves[b]
            if ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah:
                out.append((a, b))
    return out


if __name__ == "__main__":
    import xml.dom.minidom
    xml.dom.minidom.parseString(doc)                     # raises if the XML is malformed
    bad = overlaps()
    if bad:
        raise SystemExit(f"overlapping boxes: {bad}")
    (OUT / f"{NAME}.drawio").write_text(doc, encoding="utf-8")
    config = {"highlight": "#0000ff", "nav": True, "resize": True, "toolbar": "zoom layers lightbox",
              "edit": "_blank", "xml": model}
    page = ("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<title>ASEM architecture v4</title>\n"
            "<style>body{margin:0;background:#f5f5f5;display:flex;justify-content:center;align-items:center;"
            "min-height:100vh}.container{background:white;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.1);"
            "overflow:hidden}</style>\n</head>\n<body>\n<div class=\"container\">\n"
            f"<div class=\"mxgraph\" style=\"max-width:100%;border:1px solid transparent;\" "
            f"data-mxgraph=\"{html.escape(json.dumps(config), quote=True)}\"></div>\n</div>\n"
            "<script src=\"https://viewer.diagrams.net/js/viewer-static.min.js\"></script>\n</body>\n</html>\n")
    (OUT / f"{NAME}.drawio.html").write_text(page, encoding="utf-8")
    print(json.dumps({"cells": len(cells), "vertices": len(boxes), "files": [f"{NAME}.drawio", f"{NAME}.drawio.html"]}))
