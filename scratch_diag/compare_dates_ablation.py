# -*- coding: utf-8 -*-
"""Scratch (delete after use): compare the no-dates vs with-dates fair-play runs.

Reads:
  data/benchmarks/results/fairplay_locomo10_conv26.json        (baseline, no dates)
  data/benchmarks/results/fairplay_locomo10_conv26_dates.json  (--with-dates)

Writes a markdown ablation table to scratch_diag/dates_ablation_report.md
"""
import io, json, os

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
RES = os.path.join(ROOT, "data", "benchmarks", "results")

def load(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

base = load("fairplay_locomo10_conv26.json")
datd = load("fairplay_locomo10_conv26_dates.json")

out = io.StringIO()
def p(*a): print(*a, file=out)

METHODS = ["FastASEM", "ASEMv2", "SimRetrieval", "FullContext"]
METRICS = [("em", "EM%"), ("rouge_l", "ROUGE-L%"), ("bertscore_f1", "BERT-F1%"), ("judge", "Judge%")]

def fmt(v):
    if v is None:
        return "  -  "
    return f"{100 * v:5.1f}"

p("# Temporal-metadata ablation — no dates vs with dates\n")

if base is None or datd is None:
    p(f"Missing one of the summaries. base={base is not None}, dates={datd is not None}")
    p("Run:  python scripts/run_fair_play.py            # baseline")
    p("      python scripts/run_fair_play.py --with-dates")
else:
    bn = base.get("metadata", {}).get("total_questions")
    dn = datd.get("metadata", {}).get("total_questions")
    p(f"- baseline questions: {bn} | with-dates questions: {dn} "
      f"(with_dates={datd.get('metadata', {}).get('with_dates')})")
    p("")

    # ---- Overall ----
    p("## Overall\n")
    hdr = f"| {'Method':<14} |" + "".join(f" {m[1]}: no / with (Δ) |" for m in METRICS)
    p(hdr)
    p("|" + "---|" * (len(METRICS) + 1))
    for name in METHODS:
        b = base.get("overall", {}).get(name, {})
        d = datd.get("overall", {}).get(name, {})
        cells = []
        for key, _ in METRICS:
            bv, dv = b.get(key), d.get(key)
            if bv is None or dv is None:
                cells.append("  -  ")
            else:
                cells.append(f"{fmt(bv)} / {fmt(dv)} ({(dv - bv) * 100:+.1f})")
        p(f"| {name:<14} | " + " | ".join(cells) + " |")
    p("")

    # ---- Per-category EM ----
    p("## Per-category EM%\n")
    cats = ["Temporal Reasoning", "Conversational Context", "Single-Hop", "Multi-Hop / Commonsense"]
    p("| Method | Category | no dates | with dates | Δ |")
    p("|---|---|---:|---:|---:|")
    for name in METHODS:
        for c in cats:
            b = base.get("by_category", {}).get(name, {}).get(c, {})
            d = datd.get("by_category", {}).get(name, {}).get(c, {})
            bv, dv = b.get("em"), d.get("em")
            if bv is None or dv is None:
                continue
            p(f"| {name} | {c} | {100*bv:.1f} | {100*dv:.1f} | {(dv-bv)*100:+.1f} |")
    p("")

    # ---- Default direction check: FastASEM should be identical (no dates used) ----
    p("## Notes\n")
    p("- FastASEM is re-scored from the same existing preds in both runs, so its row "
      "should be identical (a sanity check).")
    p("- The interesting quantity is **Δ per method**: how much each method converts the "
      "session-date metadata into temporal answers.")

with open(os.path.join(ROOT, "scratch_diag", "dates_ablation_report.md"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("wrote scratch_diag/dates_ablation_report.md")
