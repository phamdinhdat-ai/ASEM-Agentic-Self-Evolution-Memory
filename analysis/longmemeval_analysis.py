"""LongMemEval analysis: paired ablation test + per-category breakdown.

Companion to analysis/ws4_paired_significance.py, which does the same job for
LoCoMo. LongMemEval needs its own tool because its SHAPE is different:

  * Each of the 500 questions owns a PRIVATE haystack. There is no shared bank,
    so a "conversation" is not the unit of analysis -- the QUESTION is.
  * Questions carry a `question_type` (temporal-reasoning, multi-session,
    knowledge-update, single-session-user, single-session-assistant,
    single-session-preference), and the user's claim is specifically about
    long-term-memory and multi-hop behaviour, so the per-type split is the
    interesting slice, not an afterthought.
  * Both arms answer the SAME 500 questions, so they pair by question id.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
  * It does not report `session_recall`. On the LongMemEval oracle haystack every
    session is a gold session, so session_recall is trivially 1.00 on 100% of
    items and is not evidence of anything.
  * It does not emit an accuracy rate when either arm has a judge failure
    (`judge_correct is None`). A rate computed over a failed measurement is the
    exact defect that produced the silent judge=0.0 outage.

Run:
    $py analysis/longmemeval_analysis.py --tag-thgv1 thg_v1_lme --tag-legacy legacy_lme
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "data" / "benchmarks" / "results" / "longmemeval"


def load_scores(tag: str) -> dict[str, dict]:
    """Per-question judge/BERTScore rows, keyed by question id.

    RAISES when the scores file is ABSENT rather than returning `{}`.

    An earlier version returned an empty dict here, and the consequence was the
    exact error class this module exists to prevent: with `scores/` missing the
    judge guard could not fire (there were no rows to inspect), so the tool
    printed `judge=0.0000 cov=0.00` and exited 0 -- a MISSING measurement
    rendered as a NUMBER. That is the same shape as the outage that produced a
    silent `judge: 0.0` in the first place.

    The real case: `run_longmemeval_eval.py` gates the judge on the `--judge`
    FLAG (:676-677, :722), NOT on `"judge" in --metrics`. Omitting it writes no
    `scores/*.jsonl` at all and the aggregate reports 0.0/0.0. Note the two
    runners differ -- `run_static_eval.py` enables the judge from
    `--metrics judge` alone -- so a command pattern learned on one silently
    half-works on the other.
    """
    pattern = f"{tag}/scores/*__ASEM-THG.jsonl"
    hits = sorted(RESULTS.glob(pattern))
    if not hits:
        raise SystemExit(
            f"no scores file for tag '{tag}' (looked for {pattern}).\n"
            "The judge phase did not run or did not write per-row verdicts, so there\n"
            "is nothing to test. Do NOT read an aggregate judge=0.0 as a result: check\n"
            "that the command passed the runner's judge FLAG (run_longmemeval_eval.py\n"
            "requires --judge; run_static_eval.py does not have that flag and enables\n"
            "the judge from --metrics judge alone)."
        )
    rows: dict[str, dict] = {}
    for path in hits:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                key = r.get("question_id") or r.get("idx")
                rows[str(key)] = r
    if not rows:
        raise SystemExit(f"scores file(s) for '{tag}' exist but contain no rows")
    return rows


def load_bertscore(tag: str) -> dict[str, float]:
    """Per-question BERTScore from the PREDS jsonl, keyed by question id.

    BERTScore is NOT in `scores/` -- that file carries only judge fields
    (`idx, pred, ref, judge_correct, judge_reasoning, judge_error`). It lives in
    `preds/`, which also carries `idx` so the two arms can still be paired.
    """
    hits = sorted(RESULTS.glob(f"{tag}/preds/*__ASEM-THG.jsonl"))
    out: dict[str, float] = {}
    for path in hits:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get("bertscore_f1")
            if v is not None:
                out[str(r.get("idx"))] = v
    return out


def load_overall(tag: str) -> dict:
    hits = sorted(RESULTS.glob(f"{tag}/{tag}__*.json"))
    if not hits:
        raise SystemExit(f"no result json for tag '{tag}'")
    d = json.loads(hits[0].read_text(encoding="utf-8"))
    return (d.get("systems") or {}).get("ASEM-THG", {})


def sign_test(k: int, n: int) -> float:
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n, 1.0)


def fmt(v, places: int = 4, width: int = 10) -> str:
    """Format a metric, printing `n/a` for None instead of crashing or showing 0.

    A missing metric must be VISIBLY missing. Rendering `None` as `0.0000` is the
    error class this project hit when a failed judge call became a clean-looking
    `judge: 0.0`, so an absent value is shown as `n/a` and never as a number.
    """
    if v is None:
        return f"{'n/a':>{width}}"
    return f"{v:>{width}.{places}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-thgv1", default="gb_full", help="results tag for context_style=thg_v1")
    ap.add_argument("--tag-legacy", default="gb_full_legacy", help="results tag for legacy")
    args = ap.parse_args()

    a_all = load_overall(args.tag_thgv1)
    print("=" * 78)
    print("LONGMEMEVAL -- ASEM-THG only (per user directive; no baseline system)")
    print("=" * 78)
    o = a_all.get("overall") or {}
    print(f"tag {args.tag_thgv1}: n={o.get('n')} em={fmt(o.get('em'))} f1={fmt(o.get('f1'))} "
          f"rougeL={fmt(o.get('rougeL'))} bertscore={fmt(o.get('bertscore_f1'))} "
          f"judge={fmt(o.get('judge'))} cov={fmt(o.get('judge_coverage'), 2, 5)}")

    # Refuse to present a judge figure whose coverage is not 1.0. A judge score
    # with coverage 0.0 is the ABSENCE of a measurement, not a score of zero, and
    # printing it beside populated metrics is what makes it look real.
    cov = o.get("judge_coverage")
    if "judge" in (o or {}) and (cov is None or cov != 1.0):
        print()
        print("!" * 78)
        print(f"JUDGE NOT MEASURED for tag '{args.tag_thgv1}': judge_coverage={cov}.")
        print("The judge figure above is NOT a score -- the judge phase did not run or")
        print("did not produce verdicts. If it is 0.0, the usual cause on this runner is")
        print("a MISSING --judge FLAG (run_longmemeval_eval.py gates the judge on the flag,")
        print("not on 'judge' in --metrics). EM/F1/rougeL/BERTScore above remain valid.")
        print("!" * 78)
        return 3

    pc = a_all.get("per_category") or {}
    if pc:
        print()
        print("per question_type (thg_v1):")
        print(f"  {'type':30s} {'n':>5s} {'EM':>7s} {'F1':>7s} {'BERTScore':>10s} {'judge':>7s} {'cov':>5s}")
        for k in sorted(pc, key=lambda x: -(pc[x].get("n") or 0)):
            v = pc[k]
            print(f"  {k:30s} {v.get('n'):>5.0f} {fmt(v.get('em'), 4, 7)} {fmt(v.get('f1'), 4, 7)} "
                  f"{fmt(v.get('bertscore_f1'), 4, 10)} {fmt(v.get('judge'), 4, 7)} "
                  f"{fmt(v.get('judge_coverage'), 2, 5)}")

    # ---- paired ablation, if the legacy arm exists ----
    try:
        b_all = load_overall(args.tag_legacy)
    except SystemExit:
        print()
        print(f"(legacy arm tag '{args.tag_legacy}' not present yet -- paired test skipped)")
        return 0

    sa, sb = load_scores(args.tag_thgv1), load_scores(args.tag_legacy)
    common = sorted(set(sa) & set(sb))
    print()
    print("=" * 78)
    print(f"PAIRED ABLATION TEST -- {len(common)} questions paired by question id")
    print("=" * 78)

    fail_a = [q for q in common if sa[q].get("judge_correct") is None]
    fail_b = [q for q in common if sb[q].get("judge_correct") is None]
    print(f"judge failures: thg_v1={len(fail_a)} legacy={len(fail_b)}")
    if fail_a or fail_b:
        print("REFUSING TO EMIT AN ACCURACY: judge failures present.")
        return 2

    both = sum(1 for q in common if sa[q]["judge_correct"] and sb[q]["judge_correct"])
    oa = sum(1 for q in common if sa[q]["judge_correct"] and not sb[q]["judge_correct"])
    ob = sum(1 for q in common if not sa[q]["judge_correct"] and sb[q]["judge_correct"])
    ne = sum(1 for q in common if not sa[q]["judge_correct"] and not sb[q]["judge_correct"])
    n = len(common)
    print(f"both={both}  thg_v1 only={oa}  legacy only={ob}  neither={ne}")
    print(f"judge accuracy  thg_v1={((both + oa) / n):.4f} ({both + oa}/{n})   "
          f"legacy={((both + ob) / n):.4f} ({both + ob}/{n})")
    disc = oa + ob
    p = sign_test(min(oa, ob), disc)
    print(f"discordant={disc} (thg_v1 {oa} vs legacy {ob})   exact two-sided p={p:.4f}")
    print("VERDICT: " + ("NOT significant -> ablation NOT resolved." if p >= 0.05
                         else f"significant -> {'thg_v1' if oa > ob else 'legacy'} ahead."))

    # BERTScore is NOT in the scores jsonl -- that file carries only judge fields
    # (`idx, pred, ref, judge_correct, judge_reasoning, judge_error`). BERTScore
    # lives in the PREDS jsonl, which additionally carries an `idx` we can pair on.
    bs_a, bs_b = load_bertscore(args.tag_thgv1), load_bertscore(args.tag_legacy)
    paired_bs = [q for q in common if q in bs_a and q in bs_b]
    if paired_bs:
        da = [bs_a[q] for q in paired_bs]
        db = [bs_b[q] for q in paired_bs]
        diff = [x - y for x, y in zip(da, db)]
        w = sum(1 for d in diff if d > 0)
        l = sum(1 for d in diff if d < 0)
        m = len(diff)
        print()
        print(f"BERTScore-F1, paired (n={m}):")
        print(f"  thg_v1 mean {st.mean(da):.4f}   legacy mean {st.mean(db):.4f}")
        print(f"  mean diff {st.mean(diff):+.4f}  median {st.median(diff):+.4f}  "
              f"win/lose/tie {w}/{l}/{m - w - l}")
        print(f"  paired sign p {sign_test(min(w, l), w + l):.4f}")
    else:
        print()
        print("BERTScore: no paired preds available (skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
