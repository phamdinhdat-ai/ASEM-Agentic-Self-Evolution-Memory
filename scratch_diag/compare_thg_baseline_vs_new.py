# -*- coding: utf-8 -*-
"""Compare the ASEM-THG baseline eval vs the post-P0/new-context eval.

Reads:
  data/benchmarks/results/static/locomo10/ds_thg__deepseek_v4_flash__BASELINE.json
  data/benchmarks/results/static/locomo10/ds_thg__deepseek_v4_flash.json

Writes scratch_diag/thg_context_delta.md
"""
from __future__ import annotations

import io, json, os

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
RES = os.path.join(ROOT, "data/benchmarks/results/static/locomo10")
BASE = os.path.join(RES, "ds_thg__deepseek_v4_flash__BASELINE.json")
NEW = os.path.join(RES, "ds_thg__deepseek_v4_flash.json")

out = io.StringIO()
def p(*a): print(*a, file=out)

def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def cell(d, key):
    v = d.get(key)
    return None if v is None else 100 * v

if not os.path.exists(NEW):
    p(f"new results not written yet: {NEW}")
else:
    b = load(BASE)["systems"]["ASEM-THG"]
    n = load(NEW)["systems"]["ASEM-THG"]

    p("# ASEM-THG — baseline vs post-P0 + natural-language context\n")
    p("Metrics: EM% / ROUGE-L% / BERTScore-F1% (199 conv-26 QA)\n")

    p("## Overall\n")
    p("| metric | baseline | new | Δ |")
    p("|---|--:|--:|--:|")
    for k, label in [("em", "EM"), ("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore-F1")]:
        bv, nv = cell(b["overall"], k), cell(n["overall"], k)
        p(f"| {label} | {bv:.1f} | {nv:.1f} | {nv - bv:+.1f} |")
    p("")

    p("## Per category\n")
    p("| category | n | EM base | EM new | Δ EM | ROUGE base | ROUGE new | Δ ROUGE |")
    p("|---|--:|--:|--:|--:|--:|--:|--:|")
    cats = sorted(set(b["per_category"]) | set(n["per_category"]))
    for c in cats:
        bc = b["per_category"].get(c, {})
        nc = n["per_category"].get(c, {})
        if not bc and not nc:
            continue
        cnt = nc.get("n") or bc.get("n") or 0
        be, ne = cell(bc, "em"), cell(nc, "em")
        br, nr = cell(bc, "rougeL"), cell(nc, "rougeL")
        f = lambda x: "  –  " if x is None else f"{x:.1f}"
        de = "  –  " if (be is None or ne is None) else f"{ne - be:+.1f}"
        dr = "  –  " if (br is None or nr is None) else f"{nr - br:+.1f}"
        p(f"| {c} | {int(cnt)} | {f(be)} | {f(ne)} | {de} | {f(br)} | {f(nr)} | {dr} |")
    p("")
    p("> Note: baseline = distillation mode, old label-style context. new = direct mode + "
      "temporal QA prompt + same-entity/superseded_by weights + natural-language context.")

with open(os.path.join(ROOT, "scratch_diag", "thg_context_delta.md"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print(out.getvalue())
