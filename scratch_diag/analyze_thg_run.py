# -*- coding: utf-8 -*-
"""Summarize an in-progress/finished ASEM-THG static eval preds file by category:
n, how many answer "I don't know", and mean EM / em_loose.
"""
import io, json, os, re, sys

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl")

rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
by = {}
for r in rows:
    by.setdefault(r.get("category_name", "?"), []).append(r)

def is_idk(s):
    return bool(re.search(r"\bi\s+don'?t\s+know\b", (s or "").lower())) or not (s or "").strip()

out = io.StringIO()
def p(*a): print(*a, file=out)
p(f"# in-progress preds: {os.path.basename(path)}  ({len(rows)} answered)\n")
p("| category | n | 'I don't know' | % idk | EM | em_loose |")
p("|---|--:|--:|--:|--:|--:|")
tot = len(rows)
tot_idk = sum(1 for r in rows if is_idk(r.get("pred")))
for c, rs in sorted(by.items()):
    n = len(rs)
    idk = sum(1 for r in rs if is_idk(r.get("pred")))
    em = sum(r.get("em") or 0 for r in rs) / n
    el = sum(r.get("em_loose") or 0 for r in rs) / n
    p(f"| {c} | {n} | {idk} | {100*idk/n:.0f}% | {100*em:.1f} | {100*el:.1f} |")
p(f"| **total** | **{tot}** | **{tot_idk}** | **{100*tot_idk/max(tot,1):.0f}%** | "
  f"**{100*sum(r.get('em') or 0 for r in rows)/max(tot,1):.1f}** | "
  f"**{100*sum(r.get('em_loose') or 0 for r in rows)/max(tot,1):.1f}** |")
p("")
p("## Sample 'I don't know' answers with their gold (non-adversarial first)\n")
shown = 0
for c, rs in sorted(by.items()):
    if c == "adversarial":
        continue
    for r in rs:
        if is_idk(r.get("pred")) and shown < 12:
            p(f"- [{c}] Q: {r.get('question')}\n  gold: {r.get('ref')!r}")
            shown += 1
with open(os.path.join(ROOT, "scratch_diag", "thg_run_idk_analysis.md"), "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print(out.getvalue())
