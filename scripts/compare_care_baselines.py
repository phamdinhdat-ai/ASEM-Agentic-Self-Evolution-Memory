"""Paired comparison of CARE variants and baselines on the same 1,986 LoCoMo questions.

Systems (all Qwen3-8B reader, 8K window; scores from saved per-question runs):
    no_memory, full_context_8k        build/care/context_baselines/shard-*/**/report.json
    sim_retrieval_minilm (top-18)     build/care/sim_retrieval/shard-*/**/report.json
    bm25_8_first, bm25_8_repair       build/care/relation_full/shard-*/**/report.json (CARE)
    minilm_8_first, minilm_8_repair   build/care/matched_reader/shard-*/**/report.json
    qwen3emb_8_first, qwen3emb_8_repair   matched reader with Qwen3-Embedding-0.6B rankings
A system whose five shard reports are missing is skipped. Gold labels are used
only for scoring. Intervals resample the ten conversations (10,000 draws); with
ten clusters they are optimistic, so the per-conversation win count is reported too.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path
import random

from scripts.run_care_locomo import score_locomo

ROOT = Path(__file__).resolve().parents[1]
REF = ("no information available", "not mentioned")
CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
REFERENCE = "bm25_8_repair"


def refused(text: str) -> bool:
    return any(k in (text or "").lower() for k in REF) or not (text or "").strip()


def load(pattern: str) -> dict[str, dict] | None:
    paths = sorted(glob.glob(str(ROOT / pattern), recursive=True))
    if len(paths) != 5:
        return None
    rows = [r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate questions in {pattern}")
    return by_id


def collect(qwen_pattern: str) -> tuple[dict, dict]:
    gold = {f"{c['sample_id']}:{i}": qa
            for c in json.loads((ROOT / "build/care/locomo10.json").read_text(encoding="utf-8"))
            for i, qa in enumerate(c["qa"])}
    systems: dict[str, dict[str, dict]] = {}

    ctx = load("build/care/context_baselines/shard-*/**/report.json")
    if ctx:
        for arm, name in (("no_memory", "no_memory"), ("full_context_8k", "full_context_8k")):
            systems[name] = {q: {"score": r["systems"][arm]["score"], "answer": r["systems"][arm]["answer"],
                                 "calls": 1} for q, r in ctx.items()}
    sim = load("build/care/sim_retrieval/shard-*/**/report.json")
    if sim:
        systems["sim_retrieval_minilm"] = {q: {"score": r["systems"]["sim_retrieval"]["score"],
                                               "answer": r["systems"]["sim_retrieval"]["answer"], "calls": 1}
                                           for q, r in sim.items()}
    care = load("build/care/relation_full/shard-*/**/report.json")
    if care:
        first, final = {}, {}
        for q, r in care.items():
            s = r["systems"]["bridge_v2"]
            fp = s["first_pass"] or s
            category = int(gold[q]["category"])
            first[q] = {"score": score_locomo(fp["answer"], str(gold[q].get("answer", "")), category),
                        "answer": fp["answer"], "calls": 1}
            final[q] = {"score": s["f1"], "answer": s["answer"], "calls": int(s["new_calls"])}
        systems["bm25_8_first"], systems["bm25_8_repair"] = first, final
    for prefix, pattern in (("minilm_8", "build/care/matched_reader/shard-*/**/report.json"),
                            ("qwen3emb_8", qwen_pattern)):
        rows = load(pattern)
        if rows:
            for arm, suffix in (("dense_8_first", "first"), ("dense_8_conditional", "repair")):
                systems[f"{prefix}_{suffix}"] = {q: {"score": r["systems"][arm]["score"],
                                                     "answer": r["systems"][arm]["answer"],
                                                     "calls": r["systems"][arm]["model_calls"]}
                                                 for q, r in rows.items()}
    for name, values in systems.items():
        if set(values) != set(gold):
            raise ValueError(f"{name} does not cover the 1,986 questions")
    return gold, systems


def summarize(gold: dict, values: dict) -> dict:
    ids = sorted(gold)
    answerable = [q for q in ids if int(gold[q]["category"]) != 5]
    unsupported = [q for q in ids if int(gold[q]["category"]) == 5]
    out = {"aggregate": sum(values[q]["score"] for q in ids) / len(ids),
           "answerable_f1": sum(values[q]["score"] for q in answerable) / len(answerable),
           "refusal_correct": sum(values[q]["score"] == 1 for q in unsupported),
           "refused_answerable": sum(refused(values[q]["answer"]) for q in answerable),
           "calls": sum(values[q]["calls"] for q in ids), "by_category": {}}
    for c, name in CATEGORY.items():
        qs = [q for q in ids if int(gold[q]["category"]) == c]
        out["by_category"][name] = sum(values[q]["score"] for q in qs) / len(qs)
    return out


def contrast(gold: dict, a: dict, b: dict, seed: int = 20261117) -> dict:
    by_conv = defaultdict(list)
    for q in gold:
        by_conv[q.split(":")[0]].append(a[q]["score"] - b[q]["score"])
    names = sorted(by_conv)
    diffs = [d for v in by_conv.values() for d in v]
    rng = random.Random(seed)
    draws = []
    for _ in range(10000):
        pick = [by_conv[rng.choice(names)] for _ in names]
        draws.append(sum(sum(p) for p in pick) / sum(len(p) for p in pick))
    draws.sort()
    per_conv = {n: sum(v) / len(v) for n, v in by_conv.items()}
    return {"mean_diff": sum(diffs) / len(diffs), "ci95": [draws[250], draws[9750]],
            "improved": sum(d > 1e-12 for d in diffs), "worsened": sum(d < -1e-12 for d in diffs),
            "conversations_better": sum(v > 0 for v in per_conv.values()),
            "conversations_worse": sum(v < 0 for v in per_conv.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-pattern", default="kaggle_outputs/mr_qwen3emb/shard-*/**/report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/CARE_baselines_comparison_20261117.json")
    args = parser.parse_args()
    gold, systems = collect(args.qwen_pattern)
    result = {"n_questions": len(gold), "systems": {n: summarize(gold, v) for n, v in systems.items()},
              "contrasts_vs_care": {}, "contrasts": {}}
    for name in systems:
        if name != REFERENCE and REFERENCE in systems:
            result["contrasts_vs_care"][name] = contrast(gold, systems[name], systems[REFERENCE])
    for a, b in (("qwen3emb_8_first", "bm25_8_first"), ("qwen3emb_8_first", "minilm_8_first"),
                 ("qwen3emb_8_repair", "minilm_8_repair"), ("qwen3emb_8_repair", "qwen3emb_8_first")):
        if a in systems and b in systems:
            result["contrasts"][f"{a}_minus_{b}"] = contrast(gold, systems[a], systems[b])
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{'system':24s} {'agg':>7s} {'ans F1':>7s} {'refuse ok':>9s} {'refused ans':>11s} {'calls':>6s}")
    for n, s in result["systems"].items():
        print(f"{n:24s} {s['aggregate']:7.4f} {s['answerable_f1']:7.4f} {s['refusal_correct']:5d}/446 "
              f"{s['refused_answerable']:6d}/1540 {s['calls']:6d}")
    for n, c in {**result["contrasts_vs_care"], **result["contrasts"]}.items():
        print(f"  {n:40s} {c['mean_diff']:+.4f} [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}] "
              f"conv {c['conversations_better']}/10 better")


if __name__ == "__main__":
    main()
