"""W10 EVIDENCE 2 — thg_v1 vs FullContext token efficiency, BOTH datasets.

FAIRNESS (ADDITION A): both arms are budget-matched through the SAME shared
mechanism. ASEM-THG's context is fitted by `AnswerAgent._fit` and FullContext's
by `_BudgetedAnswer.fit_context`, both of which resolve the budget with
`asem.token_budget.resolve_budget(backend, context_window, max_tokens)` — i.e.
`context_window - max_tokens - SAFETY_MARGIN_TOKENS`. Both use 32768 / 512 from
configs/models/deepseek_v41_flash.yaml, giving the same 32192-token budget.

Two distinct context sizes are reported, never conflated:
  (1) RENDERED PROMPT  — the string actually sent (what costs tokens).
  (2) CANDIDATE POOL   — what retrieval produced BEFORE trimming; FullContext's
                         analogue is len(history).

CONTENT ASSERTIONS: if you add a "is this text present in the rendered block?"
check, route it through `tests/assert_content.py` (`contains` / `assert_contains`
/ `assert_absent`). A raw `"text" in block` is False for any `_wrap`-broken line,
so it passes only while a fixture happens to be short — green for the wrong
reason. This script currently makes no such check; keep it that way or use the
helper.

Real questions are read from the on-disk corpora; the NOTE LIST is constructed
in-code (no bank, no API), and the retrieval order is a deterministic lexical
overlap ranking so the run is reproducible without an embedder.

Run:
  $py="C:\\Users\\Dat Pham\\anaconda3\\envs\\memory-r1\\python.exe"
  $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONPATH="."
  & $py tests/dump_efficiency.py --n 60 --dataset both
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics
import sys
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from asem.answer_agent import AnswerAgent, _render_notes_block, detect_question_kind
from asem.note import LinkRecord, Note
from asem.token_budget import estimate_tokens, resolve_budget

CONFIG = "configs/models/deepseek_v41_flash.yaml"
# Budget-matched pair: the same numbers `_BudgetedAnswer` and `AnswerAgent` read.
CONTEXT_WINDOW = 32768
MAX_TOKENS = 512
SAFETY = 64
BUDGET = CONTEXT_WINDOW - MAX_TOKENS - SAFETY          # 32192

OUT = io.StringIO()


def p(*a):
    print(*a, file=OUT)


# ---------------------------------------------------------------------------
# corpora
# ---------------------------------------------------------------------------
def load_locomo(path, limit):
    """LoCoMo10 QA pairs + the conversation turns they belong to."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = []
    for conv in data:
        convo = conv.get("conversation") or {}
        qs = conv.get("qa") or conv.get("questions") or []
        sess_keys = sorted(
            (k for k in convo if re.match(r"session_\d+$", str(k))),
            key=lambda k: int(str(k).split("_")[1]),
        )
        for q in qs:
            question = q.get("question") or q.get("query") or ""
            if not question:
                continue
            items.append({
                "question": question,
                "answer": str(q.get("answer") or q.get("adversarial_answer") or ""),
                "category": q.get("category"),
                "convo": convo,
                "sess_keys": sess_keys,
                "sid": conv.get("sample_id") or conv.get("conversation_id") or "conv",
            })
            if limit and len(items) >= limit:
                return items
    return items


def locomo_history(item):
    """FullContext-style history for one LoCoMo item: all turns, session-tagged."""
    header_re = re.compile(r"session_(\d+)_date_time$")
    history = []
    for key in item["sess_keys"]:
        num = int(str(key).split("_")[1])
        date = item["convo"].get(f"session_{num}_date_time", "")
        history.append(f"[Session {num}" + (f" — {date}" if date else "") + "]")
        for turn in item["convo"].get(key, []):
            spk = turn.get("speaker", "Unknown")
            txt = turn.get("text", "")
            blip = turn.get("blip_caption", "")
            line = f"[{spk}] {txt}"
            if blip:
                line += f" (image: {blip})"
            history.append(line)
    return history


def locomo_notes(item, max_notes=40):
    """Build notes in-code for one LoCoMo item (deterministic, no API)."""
    notes, t0 = [], datetime(2023, 4, 1)
    seen = set()
    for key in item["sess_keys"]:
        num = int(str(key).split("_")[1])
        date = item["convo"].get(f"session_{num}_date_time", "")
        for turn in item["convo"].get(key, []):
            txt = (turn.get("text") or "").strip()
            if not txt or txt in seen:
                continue
            seen.add(txt)
            spk = turn.get("speaker", "Unknown")
            notes.append(Note(
                id=f"{item['sid']}_{num}_{len(notes)}",
                c=f"[{spk}] {txt}", t=t0 + timedelta(days=num * 30),
                K=[w.lower() for w in re.findall(r"[A-Za-z]{4,}", txt)[:4]],
                G=[], X=txt, e=None, L=[], z=None, q=0.5,
                session_id=f"session_{num}",
                session_date=date or (t0 + timedelta(days=num * 30)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"),
                entities=list({w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", txt)}),
                speaker=spk,
            ))
            if len(notes) >= max_notes:
                break
        if len(notes) >= max_notes:
            break
    # deterministic link set so provenance/chain rendering has something to show
    for i in range(len(notes) - 1):
        notes[i].L.append(LinkRecord(target_id=notes[i + 1].id, relation="temporal"))
    return notes


def load_longmemeval(path, limit):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    items = []
    for it in data:
        items.append(it)
        if limit and len(items) >= limit:
            break
    return items


def lme_history(item):
    """FullContext analogue — mirrors eval/longmemeval.py:history_text()."""
    history = []
    sessions = item.get("haystack_sessions") or []
    dates = item.get("haystack_dates") or []
    for i, raw in enumerate(sessions):
        date = dates[i] if i < len(dates) else ""
        history.append(f"[Session {i + 1}" + (f" — {date}" if date else "") + "]")
        for turn in raw:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history.append(f"[{role}] {content}")
    return history


def lme_notes(item, max_notes=40):
    notes, t0 = [], datetime(2023, 4, 1)
    sessions = item.get("haystack_sessions") or []
    dates = item.get("haystack_dates") or []
    for i, raw in enumerate(sessions):
        date = dates[i] if i < len(dates) else ""
        sid = f"session_{i + 1}"
        for turn in raw:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            role = turn.get("role", "user")
            notes.append(Note(
                id=f"s{i}_{len(notes)}",
                c=f"[{role}] {content}", t=t0 + timedelta(days=i * 7),
                K=[w.lower() for w in re.findall(r"[A-Za-z]{4,}", content)[:4]],
                G=[], X=content, e=None, L=[], z=None, q=0.5,
                session_id=sid, session_date=date or None,
                entities=list({w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", content)}),
                speaker=role,
            ))
            if len(notes) >= max_notes:
                break
        if len(notes) >= max_notes:
            break
    for i in range(len(notes) - 1):
        notes[i].L.append(LinkRecord(target_id=notes[i + 1].id, relation="temporal"))
    return notes


# ---------------------------------------------------------------------------
# ranking: deterministic lexical overlap (stand-in for the retriever)
# ---------------------------------------------------------------------------
def rank(notes, question, k=8):
    q = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", question or "")}
    scored = []
    for i, n in enumerate(notes):
        text = f"{n.X} {n.c} {' '.join(n.K or [])}".lower()
        overlap = len(q & {w for w in re.findall(r"[a-z]{3,}", text)})
        scored.append((-overlap, i, n))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [n for _s, _i, n in scored[:k]]


# ---------------------------------------------------------------------------
# the two context builders, budget-matched
# ---------------------------------------------------------------------------
class _StubBackend:
    """Declares only what the budget arithmetic reads."""
    default_max_tokens = None

    def generate(self, prompt, **kw):        # never called
        raise AssertionError("no generation in this measurement")


def asem_prompt(notes, question, style):
    """(1) rendered prompt via AnswerAgent._fit — the real production path."""
    agent = AnswerAgent(
        backend=_StubBackend(),
        prompt_template="{query}\n\nCANDIDATES:\n{candidates}",
        baseline_prompt_template="{query}\n\nCONTEXT:\n{context}",
        direct_mode=True,
        max_tokens=MAX_TOKENS,
        context_window=CONTEXT_WINDOW,
        max_context_notes=8,
        content_char_limit=200,
        context_style=style,
    )
    prompt, kept = agent._fit(
        agent._select(notes),
        lambda ns: agent.baseline_prompt_template.format(
            query=question,
            context=agent._render_with_stats(ns, query=question),
        ),
    )
    return prompt, kept, agent.last_render_stats


class _FullContextStub:
    """FullContext's context builder INLINED from eval/baselines.py.

    `eval/baselines.py` imports asem.pipeline etc. which is heavier than needed
    here; this mirrors `FullContext.answer` + `_BudgetedAnswer.fit_context`
    exactly, including `drop="oldest"`, `head_keep=5` and `shrink=clip_block`.
    """
    head_turns = 5

    def __init__(self):
        self.budget = resolve_budget(_StubBackend(), CONTEXT_WINDOW, MAX_TOKENS)

    def prompt(self, history, question, template):
        from asem.token_budget import clip_block, fit_items
        # NOTE the return order: fit_items returns (prompt, kept_items).
        prompt, kept = fit_items(
            list(history),
            lambda ks: template.format(
                query=question,
                context="\n".join(ks) if ks else "(no prior conversation)",
            ),
            budget=self.budget,
            min_keep=1,
            drop="oldest",
            head_keep=self.head_turns,
            shrink=clip_block,
            label="FullContext",
        )
        return prompt, kept


def dist(values):
    if not values:
        return {}
    s = sorted(values)
    return {
        "min": s[0],
        "median": int(statistics.median(s)),
        "p90": s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))],
        "max": s[-1],
        "mean": round(statistics.fmean(s), 1),
    }


def measure(dataset, items, history_fn, notes_fn, fc_template, n):
    fc = _FullContextStub()
    rows = []
    kinds = {}
    for item in items[:n]:
        question = item["question"] if dataset == "locomo" else item.get("question", "")
        if not question:
            continue
        history = history_fn(item)
        notes = notes_fn(item)
        pool = len(notes)
        ranked = rank(notes, question, k=8)

        legacy_prompt, legacy_kept, _ = asem_prompt(ranked, question, "legacy")
        thg_prompt, thg_kept, st = asem_prompt(ranked, question, "thg_v1")
        fc_prompt, fc_kept = fc.prompt(history, question, fc_template)

        kind = detect_question_kind(question)
        kinds[kind] = kinds.get(kind, 0) + 1
        rows.append({
            "dataset": dataset,
            "kind": kind,
            "q": question[:60],
            "pool": pool,
            "fc_pool": len(history),
            "legacy_chars": len(legacy_prompt),
            "fc_chars": len(fc_prompt),
            "thg_chars": len(thg_prompt),
            "legacy_tok": estimate_tokens(legacy_prompt),
            "fc_tok": estimate_tokens(fc_prompt),
            "thg_tok": estimate_tokens(thg_prompt),
            "thg_kept": len(thg_kept),
            "legacy_kept": len(legacy_kept),
            "fc_kept": len(fc_kept),
            "provenance": (st.sessions_labelled if st else 0),
            "chains": (st.chains_rendered if st else 0),
            "dropped_sent": (st.dropped_sentences if st else 0),
            "dropped_chars": (st.dropped_chars if st else 0),
        })
    return rows, kinds


def table(title, rows):
    p("")
    p("=" * 100)
    p(title)
    p("=" * 100)
    if not rows:
        p("  (no rows)")
        return
    thg = [r["thg_tok"] for r in rows]
    fc = [r["fc_tok"] for r in rows]
    leg = [r["legacy_tok"] for r in rows]
    p("N = %d questions" % len(rows))
    p("")
    p("  metric                        thg_v1      legacy    FullContext    thg/FC")
    p("  --------------------------------------------------------------------------")
    dt, df, dl = dist(thg), dist(fc), dist(leg)
    for stat in ("min", "median", "p90", "max", "mean"):
        ratio = (dt[stat] / df[stat]) if df.get(stat) else float("nan")
        p("  tokens %-6s %19s %11s %13s %8.3f"
          % (stat, dt[stat], dl[stat], df[stat], ratio))
    p("")
    p("  chars: thg_v1 median=%d  legacy median=%d  FullContext median=%d"
      % (dist([r["thg_chars"] for r in rows])["median"],
         dist([r["legacy_chars"] for r in rows])["median"],
         dist([r["fc_chars"] for r in rows])["median"]))
    p("")
    p("  CANDIDATE POOL (before trimming):")
    p("    ASEM-THG notes available   : median=%d  mean=%.1f  max=%d"
      % (dist([r["pool"] for r in rows])["median"],
         statistics.fmean([r["pool"] for r in rows]),
         max(r["pool"] for r in rows)))
    p("    FullContext history items  : median=%d  mean=%.1f  max=%d"
      % (dist([r["fc_pool"] for r in rows])["median"],
         statistics.fmean([r["fc_pool"] for r in rows]),
         max(r["fc_pool"] for r in rows)))
    p("    notes actually rendered    : thg_v1 median=%d, legacy median=%d"
      % (dist([r["thg_kept"] for r in rows])["median"],
         dist([r["legacy_kept"] for r in rows])["median"]))
    p("")
    wins = sum(1 for r in rows if r["thg_tok"] <= r["fc_tok"])
    p("  per-question: thg_v1 <= FullContext on %d/%d = %.3f"
      % (wins, len(rows), wins / len(rows)))
    # The honest tail check the lead asked for.
    over = [r for r in rows if r["thg_tok"] > r["fc_tok"]]
    p("  questions where thg_v1 EXCEEDS FullContext: %d" % len(over))
    for r in over[:5]:
        p("    %-28s thg=%6d fc=%6d  %s"
          % (r["q"], r["thg_tok"], r["fc_tok"], r["kind"]))
    p("")
    p("  provenance/chain coverage:")
    prov = sum(1 for r in rows if r["provenance"])
    p("    blocks carried a session label : %d/%d questions" % (prov, len(rows)))
    p("    note.L non-empty (chains)      : %d/%d questions"
      % (sum(1 for r in rows if r["chains"]), len(rows)))
    p("    total chains rendered          : %d" % sum(r["chains"] for r in rows))
    p("    dropped sentences (thg_v1)     : median=%d  total=%d"
      % (dist([r["dropped_sent"] for r in rows])["median"],
         sum(r["dropped_sent"] for r in rows)))
    p("")
    p("  question-kind mix: %s" % dict(sorted(kinds_all.items())))
    p("")
    p("  per-question detail (first 25):")
    p("    %-32s %-15s %8s %8s %8s %8s" % ("question", "kind", "thg_tk", "leg_tk", "fc_tok", "thg/fc"))
    for r in rows[:25]:
        p("    %-32s %-15s %8d %8d %8d %8.3f"
          % (r["q"][:32], r["kind"], r["thg_tok"], r["legacy_tok"], r["fc_tok"],
             r["thg_tok"] / max(1, r["fc_tok"])))


kinds_all = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--dataset", default="both", choices=["both", "locomo", "longmemeval"])
    args = ap.parse_args()

    p("=" * 100)
    p("W10 EVIDENCE 2 — rendered-prompt token efficiency, budget-matched")
    p("=" * 100)
    p("config        : %s" % CONFIG)
    p("context_window: %d   max_tokens: %d   safety: %d   BUDGET: %d tokens"
      % (CONTEXT_WINDOW, MAX_TOKENS, SAFETY, BUDGET))
    p("both arms resolve this budget through asem.token_budget.resolve_budget")
    p("FullContext: drop='oldest', head_keep=5, shrink=clip_block (eval/baselines.py)")
    p("ASEM-THG   : drop='tail',  min_keep=1  (AnswerAgent._fit)")
    p("NOTE: note lists are built IN-CODE; retrieval order is deterministic lexical")
    p("      overlap, so no embedder/bank/API is needed. Question TEXT and FullContext")
    p("      history come from the real corpora.")

    fc_template = "You are a memory QA assistant.\n\n{query}\n\nCONTEXT:\n{context}\n\nAnswer:"
    locomo_path = "datasets/locomo/locomo10.json"
    lme_path = "datasets/longmemeval/longmemeval_oracle.json"

    all_rows = {}
    if args.dataset in ("both", "locomo"):
        if os.path.exists(locomo_path):
            items = load_locomo(locomo_path, args.n)
            rows, kinds = measure("locomo", items, locomo_history, locomo_notes,
                                  fc_template, args.n)
            kinds_all.update(kinds)
            all_rows["locomo"] = rows
            table("LoCoMo10 (n=%d of %d loaded)" % (len(rows), len(items)), rows)
        else:
            p("\n!! %s not found — skipped" % locomo_path)

    if args.dataset in ("both", "longmemeval"):
        if os.path.exists(lme_path):
            items = load_longmemeval(lme_path, args.n)
            rows, kinds = measure("longmemeval", items, lme_history, lme_notes,
                                  fc_template, args.n)
            kinds_all.update(kinds)
            all_rows["longmemeval"] = rows
            table("LongMemEval oracle (n=%d)" % len(rows), rows)
        else:
            p("\n!! %s not found — skipped" % lme_path)

    with open("tests/efficiency_report.json", "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=1)
    with open("tests/context_efficiency_report.txt", "w", encoding="utf-8") as fh:
        fh.write(OUT.getvalue())
    print("written tests/context_efficiency_report.txt and tests/efficiency_report.json")


if __name__ == "__main__":
    main()
