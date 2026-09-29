# -*- coding: utf-8 -*-
"""ASEM-THG sample-ingestion demo.

Ingests the first N sessions of LoCoMo conv-26 through
``SinglePassSessionIngestor`` (ASEM-THG) and dumps the resulting system:
notes (fact + subject/predicate/object triplet + entities + speaker + date),
the raw LLM extraction, and the temporal hyper-graph edges.

Writes scratch_diag/asem_thg_demo.md. No repo code is modified.

Usage:
    python scratch_diag/demo_asem_thg.py [n_sessions]
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---- .env (same loader the runners use) ----------------------------------
dotenv = os.path.join(ROOT, ".env")
if os.path.exists(dotenv):
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv, override=False)
    except ImportError:
        for line in open(dotenv, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from asem.backends import build_backend
from asem.config import ASEMConfig
from asem.hyper_graph import TemporalHyperGraph, Triplet
from asem.memory_bank import MemoryBank
from asem.single_pass_ingest import SinglePassSessionIngestor
from eval.benchmark_runner import extract_sessions_from_conv

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DEMO_DIR = os.path.join(ROOT, "scratch_diag")
BANK_PATH = os.path.join(DEMO_DIR, "asem_thg_demo_bank.sqlite")
if os.path.exists(BANK_PATH):
    os.remove(BANK_PATH)

out = io.StringIO()
def p(*a): print(*a, file=out)


class RecordingTHG(TemporalHyperGraph):
    """Records the (subject, predicate, object) triplet fed for each note."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.triplets: Dict[str, Triplet] = {}

    def __bool__(self) -> bool:
        # TemporalHyperGraph defines __len__ (0 when empty) -> an empty graph is
        # FALSY, and SinglePassSessionIngestor does `hyper_graph or
        # TemporalHyperGraph()`, silently discarding our recorder. Force truthy.
        return True

    def add_note(self, note, triplet=None):  # noqa: ANN001
        if triplet is not None:
            self.triplets[note.id] = triplet
        return super().add_note(note, triplet)


class DemoIngestor(SinglePassSessionIngestor):
    """Also keeps the raw extraction so the demo can show it."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.last_extracted: Optional[Tuple[List[Dict[str, Any]], str]] = None

    def _extract_facts(self, prompt, dialogue_turns):  # noqa: ANN001
        facts, source = super()._extract_facts(prompt, dialogue_turns)
        self.last_extracted = (facts, source)
        return facts, source


def main() -> None:
    # ---- backend (endpoint serves deepseek-flash / deepseek-v4-pro) ------
    cfg = ASEMConfig.load("configs/models/deepseek_api.yaml")
    lc = cfg.inference.get("langchain", {})
    model = lc.get("model")
    lc["model"] = "deepseek-flash"          # endpoint-supported name
    cfg.inference["langchain"] = lc
    backend = build_backend(cfg.inference)
    p("# ASEM-THG sample ingestion — demo\n")
    p(f"- backend: `{cfg.inference.get('backend')}` / model `{lc.get('model')}` "
      f"(config declared `{model}` — stale, endpoint serves only deepseek-flash/deepseek-v4-pro)")
    p(f"- embedder: `{lc.get('embedder_name')}`")

    # ---- conversation ----------------------------------------------------
    raw = json.load(open("datasets/locomo/locomo10.json", encoding="utf-8"))
    conv = (raw if isinstance(raw, list) else [raw])[0]
    conv_data = conv.get("conversation", conv)
    conv_id = conv.get("conversation_id") or conv.get("sample_id") or "conv-26"
    sessions = extract_sessions_from_conv(conv_data)
    p(f"- conversation: `{conv_id}` | {len(sessions)} sessions | "
      f"{sum(len(s['turns']) for s in sessions)} turns total")
    p(f"- ingesting first **{N}** session(s)\n")

    # ---- ASEM-THG components --------------------------------------------
    thg = RecordingTHG(semantic_tau=0.70, max_entity_links=5, max_semantic_links=5)
    bank = MemoryBank(BANK_PATH)
    ing = DemoIngestor(backend, hyper_graph=thg, q0=0.50, max_retries=1)

    p("## Ingestion run\n")
    p("| # | session | date | turns | LLM source | facts | bank size |")
    p("|---|---|---|---:|---|---:|---:|")
    session_facts: Dict[str, List[Dict[str, Any]]] = {}
    for i, s in enumerate(sessions[:N]):
        facts_before = bank.size()
        notes = ing.ingest_session(
            list(s["turns"]), bank,
            session_date=s.get("date"), session_id=s.get("session_id"),
        )
        facts, source = ing.last_extracted or ([], "?")
        session_facts[s["session_id"]] = facts
        p(f"| {i+1} | {s['session_id']} | {s.get('date','')} | {len(s['turns'])} | "
          f"{source} | {len(notes)} | {bank.size()} |")

    # ---- raw LLM extraction (first session) -----------------------------
    first = sessions[0]
    p(f"\n## Raw ASEM-THG extraction — {first['session_id']} ({first.get('date','')})\n")
    p("Single LLM call -> atomic facts with `(subject, predicate, object)` triplets:\n")
    p("```json")
    p(json.dumps(session_facts.get(first["session_id"], [])[:8], indent=2, ensure_ascii=False))
    p("```")

    # ---- persisted notes -------------------------------------------------
    con = sqlite3.connect(BANK_PATH)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute("SELECT * FROM notes").fetchall()]
    con.close()
    rows.sort(key=lambda r: (r.get("session_id") or "", r.get("t") or "", r.get("c") or ""))

    p(f"\n## Persisted notes (bank: {len(rows)})\n")
    p("| # | date | fact | subject | predicate | object | entities | speaker |")
    p("|---:|---|---|---|---|---|---|---|")
    trip = thg.triplets
    for i, r in enumerate(rows):
        t = trip.get(r["id"])
        subj = t.subject if t else ""
        pred = t.predicate if t else ""
        obj = t.object if t else ""
        ents = ", ".join(json.loads(r["entities"] or "[]"))
        fact = (r["c"] or "").replace("|", "/")
        p(f"| {i+1} | {r['session_date'] or ''} | {fact[:110]} | {subj} | {pred} | {obj} | {ents} | {r['speaker'] or ''} |")

    # ---- edge summary ----------------------------------------------------
    rel_counts: Dict[str, int] = {}
    superseded_pairs: List[Tuple[str, str]] = []
    id2fact = {r["id"]: r["c"] for r in rows}
    for r in rows:
        for lnk in json.loads(r["L"] or "[]"):
            rel = lnk.get("relation", "?")
            rel_counts[rel] = rel_counts.get(rel, 0) + 1
            if rel == "superseded_by":
                superseded_pairs.append((r["id"], lnk["target_id"]))

    p("\n## Hyper-graph edges\n")
    p("| relation | directed link records | undirected (÷2) |")
    p("|---|---:|---:|")
    total = 0
    for rel, c in sorted(rel_counts.items(), key=lambda kv: -kv[1]):
        total += c
        p(f"| {rel} | {c} | {c // 2} |")
    p(f"| **total** | **{total}** | **{total // 2}** |")

    p("\n### Graph-level stats\n")
    p(f"- fact nodes in hyper-graph: {len(thg)}")
    p(f"- entity nodes: {len(thg._entity_nodes)}")
    if thg._graph is not None:
        p(f"- networkx graph: {thg._graph.number_of_nodes()} nodes, "
          f"{thg._graph.number_of_edges()} edges")
    p(f"- subject/predicate versioning index entries: {len(thg._subj_pred_index)}")

    if superseded_pairs:
        p("\n### Example `superseded_by` (temporal versioning, zero LLM)\n")
        seen = set()
        shown = 0
        for a, b in superseded_pairs:
            key = frozenset((a, b))
            if key in seen or shown >= 5:
                continue
            seen.add(key)
            p(f"- `{id2fact.get(a, a)[:80]}`")
            p(f"  ← superseded_by → `{id2fact.get(b, b)[:80]}`")
            shown += 1

    p("\n### Entity nodes (sample)\n")
    p("`" + "`, `".join(sorted(list(thg._entity_nodes))[:40]) + "`")

    with open(os.path.join(DEMO_DIR, "asem_thg_demo.md"), "w", encoding="utf-8") as f:
        f.write(out.getvalue())
    print(f"OK: {len(rows)} notes from {N} session(s); report -> scratch_diag/asem_thg_demo.md")
    for rel, c in sorted(rel_counts.items(), key=lambda kv: -kv[1]):
        print(f"   edge {rel}: {c // 2} undirected")


if __name__ == "__main__":
    main()
