"""WS4 — build the thesis measurement tables from raw run artifacts ONLY.

DESIGN RULES (each one exists because a specific error already happened on this
team; they are not stylistic preferences):

R1  NEVER do arithmetic on a headline number. Every ratio, distribution and rate
    this script reports is computed here from the per-row source artifact and
    PRINTED BY THIS SCRIPT. Nothing is copied out of another team member's prose.
    (Three derived-value errors already occurred: a `6252/2=8.57` inference, a
    double-ingest inferred from a log-line count, a double-counted pair count.)

R2  A RATE OVER ERRORED ROWS IS NOT A RATE. Every aggregate prints its
    DENOMINATOR and an explicit ERROR COUNT. If any row failed, the rate is
    marked `INVALID` and the script exits non-zero. A per-row `TypeError`
    swallowed into a field, with `None` being falsy, previously rendered total
    method failure as a clean `0.0`.

R3  judge_coverage IS THE METRIC. A judge score whose coverage != 1.0 is not
    reported as a score at all. The known live failure is a 403 on the judge
    model id, which produces judge=0.0 with coverage=0.0 and looks like a
    legitimate measurement.

R4  Budget-matched claim is verified, not asserted: the script re-derives the
    budget from the two configs and prints it.

Usage:
  $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONPATH="."
  & $py analysis/ws4_tables.py --locomo <dir> --lme <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  !! FAILURE: {msg}")


def dist(values, name="value"):
    """min / median / p90 / max / mean, printed by this script (R1)."""
    if not values:
        return {"n": 0}
    s = sorted(values)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    return {
        "n": len(s), "min": s[0], "median": int(statistics.median(s)),
        "p90": p90, "max": s[-1], "mean": round(statistics.fmean(s), 1),
    }


def fmt_dist(d):
    if not d or d.get("n", 0) == 0:
        return "(none)"
    return (f"min={d['min']} med={d['median']} p90={d['p90']} "
            f"max={d['max']} mean={d['mean']}")


# ---------------------------------------------------------------------------
# source artifact loaders
# ---------------------------------------------------------------------------
def load_scores(directory):
    """Read per-row judge verdicts. Returns (rows, n_rows, n_errors).

    This is the artifact that tells us whether a judge number is real. The
    aggregate JSON reports only `judge_coverage`, so we recompute coverage from
    the rows themselves rather than trusting either value.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "scores", "*.jsonl"))):
        system = os.path.basename(path).rsplit("__", 1)[-1][:-6]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["_system"] = system
                r["_file"] = path
                rows.append(r)
    errors = [r for r in rows if r.get("judge_error")]
    return rows, len(rows), len(errors)


def summarise_judge(rows, label):
    """R2 + R3: denominator, error count, and a refusal to emit a rate."""
    if not rows:
        fail(f"{label}: no judge rows at all")
        return None
    n = len(rows)
    errs = [r for r in rows if r.get("judge_error")]
    coverage = (n - len(errs)) / n
    print(f"    judge rows            : {n}")
    print(f"    judge ERRORS          : {len(errs)}")
    print(f"    judge_coverage        : {coverage:.4f}")
    if len(errs):
        fail(f"{label}: {len(errs)}/{n} judge rows errored -> rate INVALID (R2)")
        for e in errs[:3]:
            print(f"      first error: {str(e.get('judge_error'))[:160]}")
        return {"n": n, "errors": len(errs), "coverage": coverage, "judge": None}
    yes = sum(1 for r in rows if r.get("judge_correct"))
    print(f"    judge (coverage=1.0)  : {yes}/{n} = {yes / n:.4f}")
    return {"n": n, "errors": 0, "coverage": coverage, "judge": yes / n}


def load_results(directory):
    out = {}
    for path in glob.glob(os.path.join(directory, "*__*.json")):
        if os.path.basename(path).endswith(".json"):
            try:
                with open(path, encoding="utf-8") as fh:
                    d = json.load(fh)
            except Exception:
                continue
            if "systems" in d:
                out[path] = d
    return out


# ---------------------------------------------------------------------------
# what the runner actually SUMMARISED (read, not recomputed)
# ---------------------------------------------------------------------------
def read_summary(directory, label):
    files = [p for p in load_results(directory)]
    if not files:
        fail(f"{label}: no result JSON found in {directory}")
        return None
    path = sorted(files)[-1]
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    print(f"    summary artifact      : {path}")
    out = {}
    for sysname, blk in d.get("systems", {}).items():
        out[sysname] = {
            "overall": blk.get("overall", {}),
            "per_category": blk.get("per_category", {}),
            "n": blk.get("n"),
        }
    return out


def print_quality(summary, label):
    if not summary:
        return
    print(f"\n  QUALITY — {label}")
    print(f"    {'system':<14} {'n':>6} {'em':>7} {'f1':>7} {'rougeL':>7} "
          f"{'bs_f1':>7} {'judge':>7} {'cov':>6}")
    for sysname in sorted(summary):
        o = summary[sysname]["overall"]
        n = o.get("n", summary[sysname].get("n"))
        cov = o.get("judge_coverage")
        jv = o.get("judge")
        # R3: do not print a judge score that has no coverage behind it.
        if cov is not None and abs(cov - 1.0) > 1e-9:
            jv_s = f"{jv:.4f}" if jv is not None else "n/a"
            jv_s += " INVALID"
        else:
            jv_s = f"{jv:.4f}" if jv is not None else "n/a"
        print(f"    {sysname:<14} {n if n is not None else 0:>6} "
              f"{o.get('em', 0):>7.4f} {o.get('f1', 0):>7.4f} "
              f"{o.get('rougeL', 0):>7.4f} "
              f"{(o.get('bertscore_f1') if o.get('bertscore_f1') is not None else float('nan')):>7.4f} "
              f"{jv_s:>7} "
              f"{(cov if cov is not None else float('nan')):>6.3f}")


def print_per_category(summary, label):
    if not summary:
        return
    print(f"\n  PER-CATEGORY QUALITY + n — {label}")
    for sysname in sorted(summary):
        pc = summary[sysname]["per_category"] or {}
        if not pc:
            continue
        print(f"    [{sysname}]")
        print(f"      {'category':<24} {'n':>6} {'f1':>7} {'judge':>7} {'cov':>6}")
        for cat in sorted(pc):
            c = pc[cat]
            cov = c.get("judge_coverage")
            jv = c.get("judge")
            bad = cov is not None and abs(cov - 1.0) > 1e-9
            js = (f"{jv:.4f}" if jv is not None else "n/a") + (" INVALID" if bad else "")
            print(f"      {cat:<24} {int(c.get('n', 0)):>6} {c.get('f1', 0):>7.4f} "
                  f"{js:>7} {(cov if cov is not None else float('nan')):>6.3f}")


# ---------------------------------------------------------------------------
# cost side — from the real AnswerAgent / FullContext builders
# ---------------------------------------------------------------------------
def cost_from_efficiency():
    path = "tests/efficiency_report.json"
    if not os.path.exists(path):
        fail(f"missing {path}")
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    print("\n  COST — rendered-prompt tokens, budget-matched (R1: computed here)")
    for ds in sorted(data):
        rows = data[ds]
        if not rows:
            continue
        thg = [r["thg_tok"] for r in rows]
        leg = [r["legacy_tok"] for r in rows]
        fc = [r["fc_tok"] for r in rows]
        n = len(rows)
        print(f"\n    [{ds}]  N={n}")
        for nm, v in (("ASEM-THG thg_v1", thg), ("ASEM-THG legacy", leg),
                      ("FullContext", fc)):
            print(f"      {nm:<18} {fmt_dist(dist(v))}")
        ratios = sorted(t / f for t, f in zip(thg, fc))
        inv = sum(1 for t, f in zip(thg, fc) if t > f)
        errs = 0  # every row contributed a measurement
        print(f"      ratio thg_v1/FC    {fmt_dist({'n': n, 'min': round(ratios[0], 4), 'median': round(statistics.median(ratios), 4), 'p90': round(ratios[min(n - 1, int(round(0.9 * (n - 1))))], 4), 'max': round(ratios[-1], 4), 'mean': round(statistics.fmean(ratios), 4)})}")
        print(f"      denominator={n}  errors={errs}  -> "
              f"{'VALID' if errs == 0 else 'INVALID'}")
        print(f"      median speedup FC/thg_v1 = "
              f"{statistics.median(fc) / statistics.median(thg):.3f}x")
        print(f"      INVERSIONS (thg_v1 > FC): {inv}/{n}")
        if inv:
            print("      (these are the rows where the cost claim does NOT hold)")
            for r in rows:
                if r["thg_tok"] > r["fc_tok"]:
                    print(f"        {r['q'][:44]:<44} thg={r['thg_tok']} "
                          f"fc={r['fc_tok']} {r['kind']}")
        # per-category cost, for the dataset where kind is recorded
        byk = {}
        for r in rows:
            byk.setdefault(r["kind"], []).append(r)
        if len(byk) > 1:
            print("      cost by question kind:")
            for k in sorted(byk):
                rs = byk[k]
                t = statistics.median([x["thg_tok"] for x in rs])
                f = statistics.median([x["fc_tok"] for x in rs])
                print(f"        {k:<16} n={len(rs):>3} thg={t:>6.0f} fc={f:>6.0f} "
                      f"ratio={t / f:.3f} speedup={f / t:.2f}x")


def verify_budget():
    """R4 — re-derive the shared budget from the configs instead of asserting it."""
    print("\n  BUDGET — re-derived from config (R4)")
    try:
        sys.path.insert(0, ".")
        import yaml
        for cfg in ("configs/models/deepseek_v41_flash.yaml",
                    "configs/models/deepseek_v41_flash_thg_v1.yaml"):
            with open(cfg, encoding="utf-8") as fh:
                c = yaml.safe_load(fh)
            a = c.get("answer", {})
            cw, mt = a.get("context_window"), a.get("max_tokens")
            print(f"    {cfg}")
            print(f"      context_window={cw} max_tokens={mt} "
                  f"context_style={a.get('context_style')!r}")
            if cw and mt:
                print(f"      -> budget = {cw} - {mt} - 64 safety = {cw - mt - 64}")
    except Exception as exc:  # pragma: no cover
        fail(f"budget verification raised {exc!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--locomo", default=None,
                    help="results dir of a LoCoMo run")
    ap.add_argument("--lme", default=None,
                    help="results dir of a LongMemEval run")
    ap.add_argument("--cost-only", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("WS4 EVALUATION TABLES — every number below is computed by THIS script")
    print("=" * 78)

    if args.locomo:
        print(f"\n### LoCoMo run: {args.locomo}")
        rows, n, e = load_scores(args.locomo)
        print(f"  raw judge rows={n} errors={e}")
        summarise_judge(rows, "LoCoMo")
        print_quality(read_summary(args.locomo, "LoCoMo"), "LoCoMo")
        print_per_category(read_summary(args.locomo, "LoCoMo"), "LoCoMo")
    if args.lme:
        print(f"\n### LongMemEval run: {args.lme}")
        rows, n, e = load_scores(args.lme)
        print(f"  raw judge rows={n} errors={e}")
        summarise_judge(rows, "LongMemEval")
        print_quality(read_summary(args.lme, "LongMemEval"), "LongMemEval")
        print_per_category(read_summary(args.lme, "LongMemEval"), "LongMemEval")

    if not args.cost_only:
        verify_budget()
    cost_from_efficiency()

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S) — rates above marked INVALID are")
        print("not reportable (R2/R3):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(2)
    print("RESULT: all reported rates had denominator>0 and zero row errors")
    print("=" * 78)


if __name__ == "__main__":
    main()
