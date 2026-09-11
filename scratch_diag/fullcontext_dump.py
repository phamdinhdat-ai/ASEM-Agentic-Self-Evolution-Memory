# -*- coding: utf-8 -*-
"""Scratch (delete after use): reconstruct the EXACT FullContext context for
LoCoMo conv-26 and dump sample QA evidence to scratch_diag/.

FullContext passes the ENTIRE transcript (all_turns) as context for every
question. This script shows:
  A) mechanics + stats
  B) full transcript head/tail (as FullContext sees it)
  C) per-sample QA: category, query, ref, FastASEM pred, FullContext pred,
     and the transcript "evidence window" (turns matching answer keywords)
"""
import io, json, os, re, sys

ROOT = r"C:\Users\Dat Pham\Documents\datpd\master-phenikaa\thesis_master\ASEM-Masters\ASEM-Agentic-Self-Evolution-Memory"
sys.path.insert(0, ROOT)
from eval.benchmark_runner import extract_sessions_from_conv, CATEGORY_NAMES

out = io.StringIO()
def p(*a): print(*a, file=out)

# ---- load conv-26 (index 0) ----
data = json.load(open(ROOT + r"\datasets\locomo\locomo10.json", encoding="utf-8"))
convs = data if isinstance(data, list) else [data]
conv = convs[0]
conv_data = conv.get("conversation", conv)
sessions = extract_sessions_from_conv(conv_data)
all_turns = [t for s in sessions for t in s["turns"]]

p("# FullContext context — LoCoMo conv-26\n")
p(f"- sessions: {len(sessions)} | turns: {len(all_turns)} | total chars: {len(chr(10).join(all_turns))}")
p(f"- session ids: {[s['session_id'] for s in sessions]}")
p("")

# ---- A) mechanics ----
p("## A) How FullContext builds the prompt\n")
p("Template (`_FULL_CONTEXT_PROMPT` in eval/systems.py):")
p("```")
p("Use the conversation excerpts below to answer the question. "
  "Reply with only the answer — a few words or one sentence, no explanation.")
p("")
p("Conversation:")
p("{context}")
p("")
p("Question: {query}")
p("")
p("Answer:")
p("```")
p("")
p("- `context = \"\\n\".join(all_turns)` — the ENTIRE transcript (419 turns), "
  "identical for every question (max_history_turns=0 = no truncation).")
p("- Turn format from `extract_sessions_from_conv`: `[Speaker] text` + ` (photo: blip)`")
p("")

# ---- B) head / tail of the context ----
p("## B) Transcript as FullContext sees it\n")
p("### First 12 turns:")
p("```")
for t in all_turns[:12]:
    p(t)
p("```\n")
p("### Last 8 turns:")
p("```")
for t in all_turns[-8:]:
    p(t)
p("```\n")
p("(Full 419-turn transcript written separately to `scratch_diag/fullcontext_conv26_full.txt`)\n")

with open(ROOT + r"\scratch_diag\fullcontext_conv26_full.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(all_turns))

# ---- C) per-sample QA ----
p("## C) Sample QA + FullContext evidence\n")

preds = {}
for line in open(ROOT + r"\data\benchmarks\results\preds\locomo10_FastASEM_conv0_qa.jsonl", encoding="utf-8"):
    r = json.loads(line)
    preds[r["idx"]] = r
fullpreds = {}
for line in open(ROOT + r"\data\benchmarks\results\preds\fairplay_FullContext_conv26.jsonl", encoding="utf-8"):
    r = json.loads(line)
    fullpreds[r["idx"]] = r

samples = [0, 3, 11, 37, 46, 85, 90, 97]

def evidence_window(turns, answer, q, width=4):
    """Return turns near lines whose text overlaps answer/question keywords."""
    keywords = [w for w in re.findall(r"[A-Za-z]{4,}", str(answer) + " " + q)
                if w.lower() not in {"what", "when", "where", "why", "which", "would", "have",
                                     "does", "did", "from", "that", "with", "this", "they",
                                     "their", "there", "about", "been", "were", "being", "into",
                                     "your", "youre", "caroline", "melanie", "melanies"}]
    kw = [k.lower() for k in keywords]
    hits = []
    for i, t in enumerate(turns):
        low = t.lower()
        score = sum(1 for k in kw if k in low)
        if score:
            hits.append((score, i))
    hits.sort(reverse=True, key=lambda x: x[0])
    top = [i for _, i in hits[:4]]
    top.sort()
    if not top:
        return "(no strong keyword hits found in transcript)"
    lines = []
    lo, hi = max(0, top[0] - 1), min(len(turns), top[-1] + 2)
    for i in range(lo, hi):
        mark = ">>" if i in top else "  "
        lines.append(f"{mark} #{i}: {turns[i][:220]}")
    return "\n".join(lines)

for idx in samples:
    f = preds[idx]; u = fullpreds[idx]
    p(f"### Q idx {idx} — [{f.get('category_name')}]")
    p(f"- **Query:** {f['query']}")
    p(f"- **Reference:** {f['ref']}")
    p(f"- **FullContext pred:** {u['pred']}")
    p(f"- **FastASEM pred:** {f['pred']}")
    p(f"- **FullContext context used:** ALL 419 turns (whole transcript)\n")
    p("Evidence window (transcript lines most relevant to the answer):")
    p("```")
    p(evidence_window(all_turns, u['pred'], f['query']))
    p("```\n")

with open(ROOT + r"\scratch_diag\fullcontext_conv26_samples.md", "w", encoding="utf-8") as f:
    f.write(out.getvalue())
print("wrote scratch_diag/fullcontext_conv26_samples.md and fullcontext_conv26_full.txt")
