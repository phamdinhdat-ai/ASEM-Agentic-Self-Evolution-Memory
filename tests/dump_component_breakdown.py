"""Lead's follow-up: 4-way cost split, chain base rate + degradation, drop dists.

Measures each W10 addition in ISOLATION so the +7% is attributable, and reports
the chain base rate including how often the chain degraded silently to nothing.
"""
from __future__ import annotations

import io
import json
import statistics
import sys

sys.path.insert(0, ".")

import asem.answer_agent as aa
from asem.answer_agent import _render_notes_block, detect_question_kind
from asem.token_budget import estimate_tokens
from tests.dump_efficiency import (load_locomo, load_longmemeval, locomo_notes,
                                   lme_notes, rank)

OUT = io.StringIO()


def p(*a):
    print(*a, file=OUT)


_orig_chains = aa._render_chains
_orig_label = aa._session_label


def render(notes, kind, *, labels=True, chains=True, anchors=True, positions=True):
    """Render with individual W10 components enabled/disabled."""
    if not labels:
        aa._session_label = lambda n: ""
    if not chains:
        aa._render_chains = lambda n, i: []
    try:
        if anchors and positions:
            return _render_notes_block(notes, context_style="thg_v1", question_kind=kind)
        # Force the components off by narrowing the question kind, which is how
        # the renderer gates them.
        k = kind
        if not anchors and k == "arithmetic":
            k = "single_fact"
        if not positions and k == "ordering":
            k = "single_fact"
        return _render_notes_block(notes, context_style="thg_v1", question_kind=k)
    finally:
        aa._render_chains = _orig_chains
        aa._session_label = _orig_label


def collect():
    rows = []
    for ds, items, nfn in (
        ("locomo", load_locomo("datasets/locomo/locomo10.json", 60), locomo_notes),
        ("longmemeval", load_longmemeval("datasets/longmemeval/longmemeval_oracle.json", 60), lme_notes),
    ):
        for it in items:
            q = it.get("question") or ""
            if not q:
                continue
            notes = rank(nfn(it), q, k=8)
            kind = detect_question_kind(q)
            base = len(_render_notes_block(notes, context_style="legacy"))
            no_labels = len(render(notes, kind, labels=False))
            no_chains = len(render(notes, kind, chains=False))
            full = len(render(notes, kind))
            # component costs, each measured as (full - component_removed)
            label_cost = full - no_labels
            chain_cost = full - no_chains
            # anchors / positions: only meaningful on their own kinds
            anchor_cost = 0
            if kind == "arithmetic":
                anchor_cost = full - len(render(notes, kind, anchors=False))
            pos_cost = 0
            if kind == "ordering":
                pos_cost = full - len(render(notes, kind, positions=False))
            # counts
            nchains = len(aa._render_chains(notes, {n.id: i + 1 for i, n in enumerate(notes)}))
            any_L = sum(1 for n in notes if n.L)
            rows.append({
                "ds": ds, "kind": kind, "base": base, "full": full,
                "label_cost": label_cost, "chain_cost": chain_cost,
                "anchor_cost": anchor_cost, "pos_cost": pos_cost,
                "nchains": nchains, "any_L": any_L, "nnotes": len(notes),
            })
    return rows


rows = collect()

p("=" * 92)
p("A. FOUR-WAY COST SPLIT (mean chars, and share of the legacy baseline)")
p("=" * 92)
p("")
p("  Each component is measured by REMOVING it: cost = full - (full without it).")
p("  Components can overlap in the header, so label+chain+anchors+positions may")
p("  differ slightly from full-legacy; both numbers are reported.")
p("")
p("  %-12s %8s %8s %10s %10s %10s %10s"
  % ("dataset", "legacy", "thg_v1", "label", "chains", "anchors", "position"))
p("  " + "-" * 78)
for ds in ("locomo", "longmemeval"):
    sub = [r for r in rows if r["ds"] == ds]
    base = statistics.fmean([r["base"] for r in sub])
    full = statistics.fmean([r["full"] for r in sub])
    lab = statistics.fmean([r["label_cost"] for r in sub])
    ch = statistics.fmean([r["chain_cost"] for r in sub])
    an = statistics.fmean([r["anchor_cost"] for r in sub])
    po = statistics.fmean([r["pos_cost"] for r in sub])
    p("  %-12s %8.0f %8.0f %+9.0f %+10.0f %+10.0f %+10.0f"
      % (ds, base, full, lab, ch, an, po))
    p("  %-12s %7s %7.2f%% %+8.2f%% %+9.2f%% %+9.2f%% %+9.2f%%"
      % ("  as % of legacy", "", 100 * (full - base) / base,
         100 * lab / base, 100 * ch / base, 100 * an / base, 100 * po / base))
    p("")

# Anchor/position rows only exist on their kinds; report conditional means too.
p("  Conditional (only questions where the component fires):")
for ds in ("locomo", "longmemeval"):
    sub = [r for r in rows if r["ds"] == ds]
    ar = [r["anchor_cost"] for r in sub if r["kind"] == "arithmetic"]
    orr = [r["pos_cost"] for r in sub if r["kind"] == "ordering"]
    p("    %-12s anchors: n=%3d mean %+6.0f chars | positions: n=%3d mean %+6.0f chars"
      % (ds, len(ar), statistics.fmean(ar) if ar else 0,
         len(orr), statistics.fmean(orr) if orr else 0))
p("")

p("=" * 92)
p("B. CHAIN BASE RATE AND SILENT DEGRADATION")
p("=" * 92)
p("")
for ds in ("locomo", "longmemeval"):
    sub = [r for r in rows if r["ds"] == ds]
    n = len(sub)
    fires = sum(1 for r in sub if r["nchains"] > 0)
    zero_because_empty_L = sum(1 for r in sub if r["nchains"] == 0 and r["any_L"] == 0)
    zero_despite_L = sum(1 for r in sub if r["nchains"] == 0 and r["any_L"] > 0)
    p("  %s (N=%d)" % (ds, n))
    p("    chains rendered (>=1)          : %3d/%d = %.3f" % (fires, n, fires / n))
    p("    chains == 0 BECAUSE note.L empty: %3d/%d = %.3f  <- silent degradation"
      % (zero_because_empty_L, n, zero_because_empty_L / n))
    p("    chains == 0 DESPITE note.L set  : %3d/%d = %.3f  <- no in-block bridge"
      % (zero_despite_L, n, zero_despite_L / n))
    p("    mean chains per question        : %.2f" % statistics.fmean([r["nchains"] for r in sub]))
    p("    questions where NO note had L   : %d/%d"
      % (sum(1 for r in sub if r["any_L"] == 0), n))
    p("")
p("  INTERPRETATION: a chains==0 row is NOT a dropped bridge in this corpus.")
p("  It is either (a) note.L genuinely empty (no graph object at load), or")
p("  (b) no link whose TARGET is inside the 8-note block. Both are reported")
p("  above so the two causes are not conflated.")

p("")
p("=" * 92)
p("C. DROPPED-CONTENT DISTRIBUTION (W10.2 accounting, from the 120-row report)")
p("=" * 92)
p("")
try:
    rep = json.load(open("tests/efficiency_report.json", encoding="utf-8"))
    for ds, rws in rep.items():
        dsent = [r["dropped_sent"] for r in rws]
        dchar = [r["dropped_chars"] for r in rws]
        kept = [r["thg_kept"] for r in rws]

        def d(v):
            s = sorted(v)
            return (s[0], int(statistics.median(s)),
                    s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))], s[-1])
        p("  %s (N=%d)" % (ds, len(rws)))
        p("    dropped_sentences min/med/p90/max: %s (total %d)"
          % (d(dsent), sum(dsent)))
        p("    dropped_chars     min/med/p90/max: %s (total %d)"
          % (d(dchar), sum(dchar)))
        p("    notes rendered    min/med/p90/max: %s  (cap=8)"
          % (d(kept),))
        binding = sum(1 for r in rws if r["dropped_sent"] > 0)
        p("    questions where content was dropped: %d/%d = %.3f"
          % (binding, len(rws), binding / len(rws)))
        p("")
except FileNotFoundError:
    p("  (tests/efficiency_report.json missing)")

with open("tests/w10_component_breakdown.txt", "w", encoding="utf-8") as fh:
    fh.write(OUT.getvalue())
print("written tests/w10_component_breakdown.txt")
