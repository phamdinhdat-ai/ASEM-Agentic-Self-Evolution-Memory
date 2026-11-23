"""Headline LoCoMo contrasts for the v7 manuscript, from the saved factorial run (no model calls).

Paired conversation-cluster bootstrap (10,000 draws, seed 20261121), overall and per LoCoMo category,
for CARE (bm25_lex) against the same-reader dense RAG baseline and against its own first pass.

    python -m scripts.compute_care_paper_stats
"""
from __future__ import annotations
from collections import defaultdict
import glob, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
CONTRASTS = [("bm25_lex", "rag_dense_top18"), ("bm25_lex", "bm25_first"), ("dense_first", "bm25_first"), ("dense_lex", "bm25_lex")]


def mean(x): return sum(x) / len(x)


def contrast(rows, a, b, seed=20261121):
    g = defaultdict(list)
    for r in rows:
        g[r["history"]].append(r["systems"][a]["score"] - r["systems"][b]["score"])
    names, rng, draws = sorted(g), random.Random(seed), []
    for _ in range(10000):
        pick = [g[rng.choice(names)] for _ in names]
        draws.append(sum(sum(p) for p in pick) / sum(len(p) for p in pick))
    draws.sort()
    d = [x for v in g.values() for x in v]
    return {"mean": mean(d), "ci95": [draws[250], draws[9749]],
            "conversations_better": sum(mean(v) > 0 for v in g.values()), "conversations": len(g),
            "questions_better": sum(x > 1e-12 for x in d), "questions_worse": sum(x < -1e-12 for x in d)}


def main():
    paths = sorted(glob.glob(str(ROOT / "kaggle_outputs/care_fx/locomo-s*/**/report.json"), recursive=True))
    rows = [r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
    assert len(rows) == 1986 and len({r["id"] for r in rows}) == 1986
    out = {"n": len(rows), "overall": {f"{a}-{b}": contrast(rows, a, b) for a, b in CONTRASTS}, "by_category": {}}
    ans = [r for r in rows if not r["unsupported"]]
    out["answerable_only"] = {"n": len(ans), "bm25_lex-rag_dense_top18": contrast(ans, "bm25_lex", "rag_dense_top18")}
    for c in range(1, 6):
        sub = [r for r in rows if int(r["category"]) == c]
        sc = lambda arm: mean([r["systems"][arm]["score"] for r in sub])
        out["by_category"][CAT[c]] = {"n": len(sub), "rag": sc("rag_dense_top18"), "first_pass": sc("bm25_first"),
                                      "care": sc("bm25_lex"),
                                      "care_minus_rag": contrast(sub, "bm25_lex", "rag_dense_top18")}
    out["tokens_per_question"] = {a: mean([r["systems"][a]["prompt_tokens"] + r["systems"][a]["completion_tokens"]
                                           for r in rows]) for a in ("rag_dense_top18", "bm25_first", "bm25_lex")}
    path = ROOT / "analysis/CARE_paper_v7_stats_20261121.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["overall"], indent=1)[:900])


if __name__ == "__main__":
    main()
