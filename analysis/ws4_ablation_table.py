"""WS4 — the ablation + reproducibility table, read from run artifacts.

Every number is PRINTED BY THIS SCRIPT from per-run JSON artifacts. Nothing is
copied from prose and no headline number is derived by arithmetic on another
headline number (R1).

R2/R3 discipline: each arm reports its own n and error count. A judge score is
printed only alongside its coverage, and is marked INVALID when coverage != 1.0.

Run:
  $env:PYTHONPATH="."
  & $py analysis/ws4_ablation_table.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

ARMS = {
    "thg_v1 (context_style=thg_v1)": "data/benchmarks/results/ws4_ablation/thgv1",
    "legacy (context_style=legacy)": "data/benchmarks/results/ws4_ablation/legacy",
}
METRICS = ("em", "f1", "rougeL", "bertscore_f1", "judge")


def load_arm(directory):
    """Return the per-system aggregate block + the row-level judge artifact."""
    out = {}
    for path in glob.glob(os.path.join(directory, "**", "*.json"), recursive=True):
        with open(path, encoding="utf-8") as fh:
            try:
                d = json.load(fh)
            except Exception:
                continue
        if "systems" not in d:
            continue
        out[path] = d
    # row-level judge artifact (coverage recomputed from rows, not trusted)
    rows = []
    for path in glob.glob(os.path.join(directory, "**", "scores", "*.jsonl"),
                          recursive=True):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return out, rows


def judge_failures(rows):
    """Rows where the judge did NOT produce a verdict.

    IMPORTANT — the field name is misleading and this function exists to stop a
    second false alarm. In `scores/*.jsonl`, `judge_error` is NOT an exception
    slot: `eval/llm_as_a_judge.py` uses it to carry the judge's VERDICT
    CATEGORY for an incorrect answer (`missing_info`, `contradiction`,
    `vague`, ...). A row is a genuine failure only when the judge produced NO
    verdict at all, i.e. `judge_correct is None`. Counting non-null
    `judge_error` as an error inflates the error count with the judge's own
    legitimate "this answer is wrong" labels.

    A real outage (the 403 episode) leaves `judge_correct = None` AND a
    `judge_error` containing the transport/HTTP error text, so the two cases
    are separable by `judge_correct is None`.
    """
    return [r for r in rows if r.get("judge_correct") is None]


def main():
    print("=" * 92)
    print("WS4 ABLATION TABLE — ASEM-THG only; one variable: answer.context_style")
    print("=" * 92)

    summary = {}
    for label, directory in ARMS.items():
        files, rows = load_arm(directory)
        if not files:
            print(f"\n[{label}]  -> NO ARTIFACT at {directory}")
            continue
        path = sorted(files)[-1]
        d = json.load(open(path, encoding="utf-8"))
        if "ASEM-THG" not in d.get("systems", {}):
            print(f"\n[{label}]  -> artifact {path} has no ASEM-THG block; skipped")
            continue
        blk = d["systems"]["ASEM-THG"]
        o = blk["overall"]
        n_err = len(judge_failures(rows)) if rows else None
        summary[label] = {
            "path": path,
            "n": o.get("n"),
            "metrics": {m: o.get(m) for m in METRICS},
            "coverage": o.get("judge_coverage"),
            "rows": len(rows),
            "row_errors": n_err,
            "per_category": blk.get("per_category", {}),
        }
        print(f"\n[{label}]")
        print(f"  artifact      : {path}")
        n = o.get("n")
        print(f"  n             : {n}")
        for m in METRICS:
            v = o.get(m)
            print(f"  {m:<14}: {v:.4f}" if isinstance(v, float) else f"  {m:<14}: {v}")
        cov = o.get("judge_coverage")
        print(f"  judge_coverage: {cov}   <- R3: score is valid only if this is 1.0")
        print(f"  row-level judge rows : {len(rows)}")
        print(f"  row-level judge FAILURES (judge_correct is None): {n_err}"
              "   <- R2: any >0 invalidates the rate")
        n_wrong = sum(1 for r in rows if r.get("judge_correct") is False) if rows else 0
        print(f"  judge said INCORRECT  : {n_wrong}/{len(rows)}"
              "   (legitimate labels, NOT errors)")
        if n_err:
            print("  !! judge rate marked INVALID")
            for r in judge_failures(rows)[:3]:
                print(f"     {str(r.get('judge_error'))[:140]}")
        # per-category, printed with n so a slice can never be read as the whole
        pc = blk.get("per_category", {})
        if pc:
            print(f"  per-category ({len(pc)} categories):")
            print(f"    {'category':<16} {'n':>4} {'em':>7} {'f1':>7} {'bs_f1':>7} "
                  f"{'judge':>7}")
            for cat in sorted(pc):
                c = pc[cat]
                bf = c.get("bertscore_f1")
                print(f"    {cat:<16} {int(c.get('n', 0)):>4} {c.get('em', 0):>7.4f} "
                      f"{c.get('f1', 0):>7.4f} "
                      f"{(bf if bf is not None else float('nan')):>7.4f} "
                      f"{c.get('judge', 0):>7.4f}")

    # ---- the ablation delta, computed here (R1) ----
    a = summary.get("thg_v1 (context_style=thg_v1)")
    b = summary.get("legacy (context_style=legacy)")
    if a and b:
        print("\n" + "=" * 92)
        print("ABLATION DELTA  thg_v1 - legacy   (printed by this script, not inferred)")
        print("=" * 92)
        print(f"  {'metric':<14} {'thg_v1':>9} {'legacy':>9} {'delta':>9}  sign")
        for m in METRICS:
            va, vb = a["metrics"][m], b["metrics"][m]
            if va is None or vb is None:
                continue
            dl = va - vb
            sign = "thg_v1 BETTER" if dl > 0 else ("legacy BETTER" if dl < 0 else "TIE")
            print(f"  {m:<14} {va:>9.4f} {vb:>9.4f} {dl:>+9.4f}  {sign}")
        print(f"\n  n: thg_v1={a['n']}  legacy={b['n']}  "
              f"(same questions, same bank, same conversation/limit)")

    # ---- reproducibility check ----
    f = summary.get("thg_v1 FIRST RUN (results-dir default)")
    if a and f:
        same = all(
            (a["metrics"][m] or 0) == (f["metrics"][m] or 0) for m in METRICS
        )
        print("\n" + "=" * 92)
        print("REPRODUCIBILITY CHECK — same command, same config, run twice")
        print("=" * 92)
        print(f"  run 1 artifact: {f['path']}")
        print(f"  run 2 artifact: {a['path']}")
        for m in METRICS:
            print(f"    {m:<14} run1={f['metrics'][m]}  run2={a['metrics'][m]}")
        print(f"  identical: {same}")
        if not same:
            print("  !! NOT REPRODUCIBLE. answer.temperature is 0.0, but the")
            print("     ingestion/answer path still varies. Treat single-run")
            print("     differences below ~1 question (0.05 at n=20) as NOISE.")
        # deterministic metrics must still match exactly
        for m in ("bertscore_f1",):
            if (a["metrics"][m] or 0) != (f["metrics"][m] or 0):
                print(f"  note: {m} differs too, so the answers themselves changed")

    print("\n" + "=" * 92)
    print("LIMITATIONS (carry into the report verbatim)")
    print("=" * 92)
    print("  L1 N=20 questions from ONE conversation (locomo_0000), first 20 in")
    print("     file order. NOT a random sample and NOT the full 199-pair set.")
    print("  L2 Every claim above is scoped to ASEM-THG on bank tag thg_v2")
    print("     (mixed-window: hyper_graph.py changed on disk mid-build).")
    print("  L3 No baseline system was run, per the user's directive. The")
    print("     'less context' claim is supported by the SIZE measurement and the")
    print("     context-builder ratio, NOT by a FullContext quality comparison.")


if __name__ == "__main__":
    main()
