"""W10 EVIDENCE 2b — provenance cost, marker firing, and the distribution check."""
from __future__ import annotations

import io
import json
import statistics
import sys

sys.path.insert(0, ".")
from tests.dump_efficiency import (CONTEXT_WINDOW, MAX_TOKENS, _FullContextStub,
                                   estimate_tokens, load_locomo, load_longmemeval,
                                   locomo_history, locomo_notes, lme_history,
                                   lme_notes, rank, asem_prompt)
from asem.answer_agent import detect_question_kind, _render_notes_block

OUT = io.StringIO()


def p(*a):
    print(*a, file=OUT)


p("=" * 96)
p("PROVENANCE AND MARKER COST — thg_v1 vs legacy on the SAME note list")
p("=" * 96)
p("")

rows = []
for tag, items, notes_fn in (
    ("locomo", load_locomo("datasets/locomo/locomo10.json", 40), locomo_notes),
    ("longmemeval", load_longmemeval("datasets/longmemeval/longmemeval_oracle.json", 40), lme_notes),
):
    for it in items:
        q = it.get("question") or ""
        if not q:
            continue
        notes = notes_fn(it)
        r = rank(notes, q, k=8)
        kind = detect_question_kind(q)
        lp, _lk, _s = asem_prompt(r, q, "legacy")
        tp, _tk, st = asem_prompt(r, q, "thg_v1")
        rows.append({
            "ds": tag, "kind": kind,
            "lc": len(lp), "tc": len(tp),
            "lt": estimate_tokens(lp), "tt": estimate_tokens(tp),
            "chains": st.chains_rendered if st else 0,
            "sessions": st.sessions_labelled if st else 0,
            "notes": st.notes_rendered if st else 0,
        })

p("TOTAL N = %d questions (40 per dataset)" % len(rows))
p("")
p("  dataset      N   legacy tok  thg_v1 tok   delta tok   delta %%")
p("  " + "-" * 62)
for ds in ("locomo", "longmemeval"):
    sub = [r for r in rows if r["ds"] == ds]
    lt = statistics.fmean([r["lt"] for r in sub])
    tt = statistics.fmean([r["tt"] for r in sub])
    p("  %-11s %3d %11.0f %11.0f %10.0f %8.2f%%"
      % (ds, len(sub), lt, tt, tt - lt, 100 * (tt - lt) / lt))
allt = [r["lt"] for r in rows]
allh = [r["tt"] for r in rows]
p("  %-11s %3d %11.0f %11.0f %10.0f %8.2f%%"
  % ("COMBINED", len(rows), statistics.fmean(allt), statistics.fmean(allh),
     statistics.fmean(allh) - statistics.fmean(allt),
     100 * (statistics.fmean(allh) - statistics.fmean(allt)) / statistics.fmean(allt)))
p("")
p("Threshold check (task-2: fall back to a short ordinal if provenance >8%%):")
worst = max((100 * (r["tt"] - r["lt"]) / r["lt"], r) for r in rows)
p("  worst per-question delta: %+.2f%%" % worst[0])
p("  => %s" % ("WITHIN budget, full session label kept"
               if worst[0] <= 8 else "OVER 8% -> would need the ordinal fallback"))

p("")
p("=" * 96)
p("MARKER FIRING on real questions (thg_v1 only)")
p("=" * 96)
p("")
kinds = {}
for r in rows:
    kinds.setdefault(r["kind"], []).append(r)
p("  %-17s %5s  %s" % ("kind", "n", "structural marker expected"))
p("  " + "-" * 62)
expected = {
    "arithmetic": "Date anchors:",
    "ordering": "(position N of M)",
    "knowledge_update": "superseded + current-first",
    "multi_hop": "Evidence chains:",
    "count": "(none - generic)",
    "single_fact": "(none - generic)",
}
for k in sorted(kinds, key=lambda x: -len(kinds[x])):
    p("  %-17s %5d  %s" % (k, len(kinds[k]), expected.get(k, "")))

p("")
p("Provenance coverage:")
p("  blocks labelled / blocks rendered = %d / %d = %.3f"
  % (sum(r["sessions"] for r in rows), sum(r["notes"] for r in rows),
     sum(r["sessions"] for r in rows) / max(1, sum(r["notes"] for r in rows))))
p("  questions with >=1 chain        = %d / %d = %.3f"
  % (sum(1 for r in rows if r["chains"]), len(rows),
     sum(1 for r in rows if r["chains"]) / len(rows)))
p("  total chains rendered           = %d" % sum(r["chains"] for r in rows))

p("")
p("=" * 96)
p("DISTRIBUTION CHECK (lead's request) — thg_v1 vs FullContext, per question")
p("=" * 96)
p("")

fc = _FullContextStub()
for tag, items, hfn, nfn, qkey in (
    ("locomo", load_locomo("datasets/locomo/locomo10.json", 60), locomo_history,
     locomo_notes, "question"),
    ("longmemeval", load_longmemeval("datasets/longmemeval/longmemeval_oracle.json", 60),
     lme_history, lme_notes, "question"),
):
    tpl = "You are a memory QA assistant.\n\n{query}\n\nCONTEXT:\n{context}\n\nAnswer:"
    tt, ft = [], []
    for it in items:
        q = it.get(qkey) or ""
        if not q:
            continue
        notes = nfn(it)
        tp, _k, _s = asem_prompt(rank(notes, q, k=8), q, "thg_v1")
        fp, _fk = fc.prompt(hfn(it), q, tpl)
        tt.append(estimate_tokens(tp))
        ft.append(estimate_tokens(fp))

    def d(v):
        s = sorted(v)
        return (s[0], int(statistics.median(s)),
                s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))], s[-1])

    dt, df = d(tt), d(ft)
    p("  %s (N=%d)" % (tag, len(tt)))
    p("    %-10s %10s %10s %8s" % ("", "thg_v1", "FullContext", "ratio"))
    for i, name in enumerate(("min", "median", "p90", "max")):
        p("    %-10s %10d %10d %8.3f" % (name, dt[i], df[i], dt[i] / df[i]))
    p("    p90 thg_v1 EXCEEDS p90 FullContext: %s" % (dt[2] > df[2]))
    p("")

with open("tests/provenance_marker_report.txt", "w", encoding="utf-8") as fh:
    fh.write(OUT.getvalue())
print("written tests/provenance_marker_report.txt")
