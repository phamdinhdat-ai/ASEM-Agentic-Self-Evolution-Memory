"""Summarize the CARE component ablation (docs/CARE_ablation_protocol_20261121.md).

For each benchmark: per-arm scores and cost, the contrast of every ablation arm with the full method
(paired bootstrap, Bonferroni over the eight ablation arms), a verdict per mechanism from the fixed rule,
the decomposition first pass -> retry only -> expansion only -> full repair, and the reproducibility check
of the ``care`` arm against the saved CARE answers. LoCoMo resamples conversations; LongMemEval questions.

    python -m scripts.summarize_care_ablation --benchmark locomo
    python -m scripts.summarize_care_ablation --benchmark lme
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import random

from asem.care.ablation import ABLATION_ARMS

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"locomo": 1986, "lme": 500}
MARGIN = 0.01                               # equivalence margin, fixed in the protocol
FAMILY = len(ABLATION_ARMS)                 # 8 ablation contrasts
BONFERRONI = 1 - 0.05 / FAMILY              # 99.375%
SEED, DRAWS = 20261121, 10000
REF = ("no information available", "not mentioned")
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
ARMS = ["first_pass", "care", *ABLATION_ARMS]


def refused(text: str) -> bool:
    return any(k in (text or "").lower() for k in REF) or not (text or "").strip()


def load(benchmark: str, root: Path) -> list[dict]:
    paths = sorted(glob.glob(str(root / f"{benchmark}-s*" / "**" / "report.json"), recursive=True))
    if len(paths) != 5:
        raise ValueError(f"{benchmark}: expected 5 shard reports, found {len(paths)}")
    rows = [r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
    if len({r["id"] for r in rows}) != len(rows) or len(rows) != EXPECTED[benchmark]:
        raise ValueError(f"{benchmark}: {len(rows)} rows, expected {EXPECTED[benchmark]} distinct")
    return rows


def interval(groups: dict[str, list[float]], level: float) -> list[float]:
    names, rng, draws = sorted(groups), random.Random(SEED), []
    for _ in range(DRAWS):
        pick = [groups[rng.choice(names)] for _ in names]
        draws.append(sum(sum(p) for p in pick) / sum(len(p) for p in pick))
    draws.sort()
    tail = (1 - level) / 2
    return [draws[int(tail * DRAWS)], draws[min(DRAWS - 1, int((1 - tail) * DRAWS))]]


def contrast(rows: list[dict], a: str, b: str, cluster) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[cluster(r)].append(r["systems"][a]["score"] - r["systems"][b]["score"])
    diffs = [d for g in groups.values() for d in g]
    out = {"mean": sum(diffs) / len(diffs), "ci95": interval(groups, 0.95), "ci_bonferroni": interval(groups, BONFERRONI),
           "better": sum(d > 1e-12 for d in diffs), "worse": sum(d < -1e-12 for d in diffs)}
    if len(groups) <= 20:
        per = [sum(v) / len(v) for v in groups.values()]
        out["clusters_better"], out["clusters_total"] = sum(v > 0 for v in per), len(per)
    return out


def verdict(c: dict) -> str:
    lo, hi = c["ci_bonferroni"]
    if hi < 0:
        return "contributes (removing it lowers the score)"
    if lo > 0:
        return "removal helps (post-hoc; simplification candidate)"
    if lo > -MARGIN and hi < MARGIN:
        return f"not needed (within +/-{MARGIN})"
    return "inconclusive"


def arm_summary(rows: list[dict], arm: str) -> dict:
    v = [r["systems"][arm] for r in rows]
    ans = [r["systems"][arm] for r in rows if not r["unsupported"]]
    uns = [r["systems"][arm] for r in rows if r["unsupported"]]
    return {"score": sum(x["score"] for x in v) / len(v),
            "answerable": sum(x["score"] for x in ans) / max(len(ans), 1),
            "correct_refusals": sum(x["score"] == 1 for x in uns), "refusal_n": len(uns),
            "refused_answerable": sum(refused(x["answer"]) for x in ans),
            "calls_alone": sum(x["calls"] for x in v),
            "tokens_per_q": sum(x["prompt_tokens"] + x["completion_tokens"] for x in v) / len(v),
            "repairs": sum(bool(x.get("escalated")) for x in v)}


def reproducibility(benchmark: str, rows: list[dict]) -> dict:
    fx = "locomo" if benchmark == "locomo" else "lme"
    paths = sorted(glob.glob(str(ROOT / "kaggle_outputs/care_fx" / f"{fx}-s*" / "**" / "report.json"), recursive=True))
    saved = {r["id"]: r["systems"]["bm25_lex"]["answer"] for p in paths
             for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]}
    same = sum(saved.get(r["id"]) == r["systems"]["care"]["answer"] for r in rows)
    return {"compared": len(rows), "identical_answers": same, "all_identical": same == len(rows) == len(saved)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", choices=("locomo", "lme"), required=True)
    ap.add_argument("--root", type=Path, default=ROOT / "kaggle_outputs/care_ab")
    args = ap.parse_args()
    rows = load(args.benchmark, args.root)
    cluster = (lambda r: r["history"]) if args.benchmark == "locomo" else (lambda r: r["id"])
    repro = reproducibility(args.benchmark, rows)
    summary = {a: arm_summary(rows, a) for a in ARMS}
    contrasts = {a: contrast(rows, a, "care", cluster) for a in ABLATION_ARMS}
    for a, c in contrasts.items():
        c["verdict"] = verdict(c)
    first = summary["first_pass"]["score"]
    gain = summary["care"]["score"] - first
    decomposition = {a: {"score": summary[a]["score"], "share_of_repair_gain":
                         ((summary[a]["score"] - first) / gain if gain else None)}
                     for a in ("repair_retry_only", "repair_expand_only", "care")}
    by_type = {}
    for t in sorted({r["category"] for r in rows}, key=str):
        sub = [r for r in rows if r["category"] == t]
        name = LOCOMO_CATS.get(t, str(t)) if args.benchmark == "locomo" else str(t)
        by_type[name] = {"n": len(sub), **{a: sum(r["systems"][a]["score"] for r in sub) / len(sub) for a in ARMS}}
    physical = sum(r["physical_calls"] for r in rows)
    result = {"protocol": "care-ablation-v1", "benchmark": args.benchmark, "n": len(rows), "arms": summary,
              "contrasts_vs_care": contrasts, "decomposition": decomposition, "by_type": by_type,
              "reproducibility": repro, "physical_calls": physical, "bonferroni_level": BONFERRONI, "margin": MARGIN}
    out = ROOT / "analysis"
    out.mkdir(exist_ok=True)
    stem = f"CARE_ablation_{args.benchmark}_20261121"
    (out / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [f"# CARE component ablation: {args.benchmark} ({len(rows)} questions)", "",
             "LoCoMo is development data." if args.benchmark == "locomo" else
             "LongMemEval cleaned oracle; diagnostic metric; results were seen before design choices.", "",
             f"Reproducibility: `care` arm identical to saved CARE answers on {repro['identical_answers']}/{repro['compared']}"
             f" ({'PASS' if repro['all_identical'] else 'FAIL'}). Physical model calls: {physical}.", "",
             "| Arm | Score | Answerable | Correct refusals | Refused ans. | Repairs | Calls (alone) | Tokens/q |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        s = summary[a]
        lines.append(f"| {a} | {s['score']:.4f} | {s['answerable']:.4f} | {s['correct_refusals']}/{s['refusal_n']} | "
                     f"{s['refused_answerable']} | {s['repairs']} | {s['calls_alone']} | {s['tokens_per_q']:.0f} |")
    lines += ["", f"Contrast with `care` (paired bootstrap; Bonferroni {BONFERRONI * 100:.3f}% over {FAMILY} arms; margin +/-{MARGIN}):", ""]
    for a, c in contrasts.items():
        extra = f", {c['clusters_better']}/{c['clusters_total']} clusters better" if "clusters_better" in c else ""
        lines.append(f"- **{a}**: {c['mean']:+.4f}, 95% [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], adjusted "
                     f"[{c['ci_bonferroni'][0]:+.4f}, {c['ci_bonferroni'][1]:+.4f}] ({c['better']} better / {c['worse']} worse{extra}): "
                     f"{c['verdict']}")
    lines += ["", "Repair decomposition (share of the first-pass -> care gain):", ""]
    for a, d in decomposition.items():
        share = "--" if d["share_of_repair_gain"] is None else f"{d['share_of_repair_gain']:.2f}"
        lines.append(f"- {a}: {d['score']:.4f} (share {share})")
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
