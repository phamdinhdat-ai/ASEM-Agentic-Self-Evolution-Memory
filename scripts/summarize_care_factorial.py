"""Summarize the pre-registered factorial run (docs/CARE_factorial_protocol_20261117.md).

Reads the shard reports, then reports every arm, the three primary hypotheses
(H1 dense_hyb - bm25_lex, H2 dense_hyb - dense_lex, H3 dense_first - bm25_first,
H4 dense_graph - dense_hyb) with 95% and Bonferroni 98.75% intervals, per-type scores, and the declared
reproducibility checks. LongMemEval is the test set; LoCoMo is development.

    python -m scripts.summarize_care_factorial --root kaggle_outputs/care_fx
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
HYPOTHESES = {"H1": ("dense_hyb", "bm25_lex"), "H2": ("dense_hyb", "dense_lex"),
              "H3": ("dense_first", "bm25_first"), "H4": ("dense_graph", "dense_hyb")}
BONFERRONI = 1 - 0.05 / len(HYPOTHESES)          # 98.75% with four primary hypotheses
EXTRA = {"dense_hyb_vs_full_context": ("dense_hyb", "full_context_8k"),
         "dense_hyb_vs_rag": ("dense_hyb", "rag_dense_top18"),
         "bm25_hyb_vs_bm25_lex": ("bm25_hyb", "bm25_lex"),
         "dense_graph_vs_bm25_lex": ("dense_graph", "bm25_lex")}
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
REF = ("no information available", "not mentioned")


def refused(text: str) -> bool:
    return any(k in (text or "").lower() for k in REF) or not (text or "").strip()


def load(root: Path, short: str, expected: int) -> list[dict]:
    paths = sorted(glob.glob(str(root / f"{short}-s*" / "**" / "report.json"), recursive=True))
    if len(paths) != 5:
        raise ValueError(f"{short}: expected 5 shard reports, found {len(paths)}")
    rows = [r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
    if len({r["id"] for r in rows}) != len(rows) or len(rows) != expected:
        raise ValueError(f"{short}: {len(rows)} rows, expected {expected} distinct")
    return rows


def interval(groups: dict[str, list[float]], level: float, seed: int = 20261117) -> list[float]:
    names = sorted(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        pick = [groups[rng.choice(names)] for _ in names]
        draws.append(sum(sum(p) for p in pick) / sum(len(p) for p in pick))
    draws.sort()
    tail = (1 - level) / 2
    return [draws[int(tail * 10000)], draws[min(9999, int((1 - tail) * 10000))]]


def contrast(rows: list[dict], a: str, b: str, cluster) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[cluster(r)].append(r["systems"][a]["score"] - r["systems"][b]["score"])
    diffs = [d for g in groups.values() for d in g]
    out = {"mean": sum(diffs) / len(diffs), "ci95": interval(groups, 0.95),
           "ci_bonferroni": interval(groups, BONFERRONI),
           "better": sum(d > 1e-12 for d in diffs), "worse": sum(d < -1e-12 for d in diffs)}
    if len(groups) <= 20:
        per = {k: sum(v) / len(v) for k, v in groups.items()}
        out["clusters_better"] = sum(v > 0 for v in per.values())
        out["clusters_total"] = len(per)
    return out


def summary(rows: list[dict], arm: str, type_of) -> dict:
    vals = [r["systems"][arm] for r in rows]
    ans = [r["systems"][arm] for r in rows if not r["unsupported"]]
    uns = [r["systems"][arm] for r in rows if r["unsupported"]]
    by_type = defaultdict(list)
    for r in rows:
        by_type[type_of(r)].append(r["systems"][arm]["score"])
    rec = [v["evidence_recall"] for v in ans if v.get("evidence_recall") is not None]
    return {"score": sum(v["score"] for v in vals) / len(vals),
            "answerable": sum(v["score"] for v in ans) / len(ans),
            "refusal_correct": sum(v["score"] == 1 for v in uns), "refusal_n": len(uns),
            "refused_answerable": sum(refused(v["answer"]) for v in ans), "answerable_n": len(ans),
            "calls": sum(v["calls"] for v in vals),
            "tokens": sum(v["prompt_tokens"] + v["completion_tokens"] for v in vals),
            "evidence_recall": sum(rec) / len(rec) if rec else None,
            "by_type": {k: sum(v) / len(v) for k, v in sorted(by_type.items())}}


def reproducibility(locomo: list[dict] | None, lme: list[dict] | None) -> dict:
    out = {}
    if locomo:
        saved = {}
        for p in glob.glob(str(ROOT / "build/care/relation_full/shard-*/**/report.json"), recursive=True):
            for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]:
                saved[r["id"]] = r["systems"]["bridge_v2"]["answer"]
        same = sum(r["systems"]["bm25_lex"]["answer"] == saved.get(r["id"]) for r in locomo)
        out["locomo_bm25_lex_vs_saved_care_answers"] = {"identical": same, "n": len(locomo)}
        frozen_path = ROOT / "build/care/qwen3_0p6b_dense_rankings.json"
        if frozen_path.exists():
            frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["rankings"]
            overlap = [len(set(r["dense_top8"]) & set(frozen[r["id"]][:8])) / 8 for r in locomo]
            out["locomo_dense_top8_overlap_with_frozen"] = sum(overlap) / len(overlap)
    if lme:
        saved = {}
        for p in glob.glob(str(ROOT / "kaggle_outputs/longmemeval_20261116/shard-*/**/report.json"),
                           recursive=True):
            for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]:
                saved[r["id"]] = r["systems"]["conditional"]["answer"]
        same = sum(r["systems"]["bm25_lex"]["answer"] == saved.get(r["id"]) for r in lme)
        out["lme_bm25_lex_vs_saved_care_answers"] = {"identical": same, "n": len(lme)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "kaggle_outputs/care_fx")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/CARE_factorial_results_20261117.json")
    parser.add_argument("--only", choices=("locomo", "lme"), default=None)
    args = parser.parse_args()
    result = {"protocol": "docs/CARE_factorial_protocol_20261117.md"}
    locomo = lme = None
    if args.only in (None, "locomo"):
        locomo = load(args.root, "locomo", 1986)
        arms = list(locomo[0]["systems"])
        conv = lambda r: r["history"]
        result["locomo"] = {
            "role": "development",
            "arms": {a: summary(locomo, a, lambda r: LOCOMO_CATS[int(r["category"])]) for a in arms},
            "hypotheses": {h: contrast(locomo, a, b, conv) for h, (a, b) in HYPOTHESES.items()},
            "extra": {k: contrast(locomo, a, b, conv) for k, (a, b) in EXTRA.items() if a in arms and b in arms}}
    if args.only in (None, "lme"):
        lme = load(args.root, "lme", 500)
        arms = list(lme[0]["systems"])
        q = lambda r: r["id"]
        result["longmemeval"] = {
            "role": "test",
            "arms": {a: summary(lme, a, lambda r: r["category"]) for a in arms},
            "hypotheses": {h: contrast(lme, a, b, q) for h, (a, b) in HYPOTHESES.items()},
            "extra": {k: contrast(lme, a, b, q) for k, (a, b) in EXTRA.items() if a in arms and b in arms}}
    result["reproducibility"] = reproducibility(locomo, lme)
    # Amendment 2 means shards can run different code; record each shard's code hash.
    result["code_hashes"] = {
        str(Path(p).parent.parent.name) + "/" + str(Path(p).parent.name): json.loads(
            Path(p).read_text(encoding="utf-8"))["manifest"].get("code_sha256", "")[:16]
        for p in sorted(glob.glob(str(args.root / "*-s*" / "**" / "report.json"), recursive=True))}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for bench in ("locomo", "longmemeval"):
        if bench not in result:
            continue
        print(f"\n== {bench} ({result[bench]['role']})")
        for a, s in result[bench]["arms"].items():
            print(f"  {a:18s} score={s['score']:.4f} ans={s['answerable']:.4f} "
                  f"refusal={s['refusal_correct']}/{s['refusal_n']} refused_ans={s['refused_answerable']} "
                  f"calls={s['calls']}")
        for h, c in {**result[bench]["hypotheses"], **result[bench]["extra"]}.items():
            print(f"  {h:28s} {c['mean']:+.4f} 95%[{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
                  f"bonf[{c['ci_bonferroni'][0]:+.4f},{c['ci_bonferroni'][1]:+.4f}]")
    print("\nreproducibility", json.dumps(result["reproducibility"]))


if __name__ == "__main__":
    main()
