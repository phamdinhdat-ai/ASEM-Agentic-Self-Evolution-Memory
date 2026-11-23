"""Summarize protocol v3 (LongMemEval-S, docs/CARE_protocol_v3_20261118.md).

Primary (fixed in the protocol): H-main = bm25_hyb - bm25_lex on 500 questions, question-level paired
bootstrap, 10,000 draws, 95% two-sided. Confirmed / not confirmed / reversed by where the interval lies.
Every other contrast is exploratory and labelled so. Refuses to run unless all five shards are present.

    python -m scripts.summarize_care_v3
"""

from __future__ import annotations

from collections import defaultdict
import glob
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "kaggle_outputs/care_v3"
OUT = ROOT / "analysis"
ARMS = ["no_memory", "full_context_8k", "rag_dense_top18", "bm25_first", "bm25_lex", "bm25_hyb",
        "dense_first", "dense_lex", "dense_hyb", "dense_graph"]
EXPLORATORY = {"dense_first_vs_bm25_first": ("dense_first", "bm25_first"),
               "dense_hyb_vs_bm25_lex": ("dense_hyb", "bm25_lex"),
               "bm25_lex_vs_rag": ("bm25_lex", "rag_dense_top18"),
               "bm25_hyb_vs_rag": ("bm25_hyb", "rag_dense_top18"),
               "dense_hyb_vs_rag": ("dense_hyb", "rag_dense_top18"),
               "bm25_lex_vs_full_context_8k": ("bm25_lex", "full_context_8k"),
               "dense_graph_vs_dense_hyb": ("dense_graph", "dense_hyb")}
REF = ("no information available", "not mentioned")
SEED, DRAWS = 20261118, 10000


def refused(text: str) -> bool:
    return any(k in (text or "").lower() for k in REF) or not (text or "").strip()


def load() -> list[dict]:
    paths = sorted(glob.glob(str(SRC / "s[0-4]" / "**" / "report.json"), recursive=True))
    if len(paths) != 5:
        raise ValueError(f"expected 5 shard reports, found {len(paths)}")
    rows = [r for p in paths for r in json.loads(Path(p).read_text(encoding="utf-8"))["rows"]]
    if len(rows) != 500 or len({r["id"] for r in rows}) != 500:
        raise ValueError(f"{len(rows)} rows; expected 500 distinct")
    return rows


def paired(rows: list[dict], a: str, b: str) -> dict:
    d = [r["systems"][a]["score"] - r["systems"][b]["score"] for r in rows]
    rng = random.Random(SEED)
    n = len(d)
    draws = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(DRAWS))
    lo, hi = draws[int(0.025 * DRAWS)], draws[int(0.975 * DRAWS) - 1]
    return {"mean": sum(d) / n, "ci95": [lo, hi], "better": sum(x > 1e-12 for x in d),
            "worse": sum(x < -1e-12 for x in d)}


def arm_summary(rows: list[dict], arm: str) -> dict:
    v = [r["systems"][arm] for r in rows]
    ans = [r["systems"][arm] for r in rows if not r["unsupported"]]
    uns = [r["systems"][arm] for r in rows if r["unsupported"]]
    rec = [x["evidence_recall"] for x in ans if x.get("evidence_recall") is not None]
    return {"score": sum(x["score"] for x in v) / len(v),
            "answerable": sum(x["score"] for x in ans) / max(len(ans), 1),
            "correct_refusals": sum(x["score"] == 1 for x in uns), "refusal_n": len(uns),
            "refused_answerable": sum(refused(x["answer"]) for x in ans), "answerable_n": len(ans),
            "calls": sum(x["calls"] for x in v),
            "tokens_per_q": sum(x["prompt_tokens"] + x["completion_tokens"] for x in v) / len(v),
            "evidence_recall": sum(rec) / len(rec) if rec else None}


def main() -> None:
    rows = load()
    summary = {a: arm_summary(rows, a) for a in ARMS}
    by_type: dict[str, dict] = {}
    for t in sorted({str(r["category"]) for r in rows}):
        sub = [r for r in rows if str(r["category"]) == t]
        by_type[t] = {"n": len(sub), **{a: sum(r["systems"][a]["score"] for r in sub) / len(sub) for a in ARMS}}
    main_c = paired(rows, "bm25_hyb", "bm25_lex")
    verdict = ("confirmed" if main_c["ci95"][0] > 0 else "reversed" if main_c["ci95"][1] < 0 else "not confirmed")
    result = {"protocol": "care-v3", "n": len(rows), "arms": summary,
              "H_main": {"contrast": "bm25_hyb - bm25_lex", **main_c, "verdict": verdict},
              "exploratory": {k: paired(rows, *v) for k, v in EXPLORATORY.items()},
              "by_type": by_type}
    OUT.mkdir(exist_ok=True)
    (OUT / "CARE_v3_lmes_s_20261121.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# CARE v3: LongMemEval-S (500 questions, full haystacks)", "",
             "Diagnostic token-F1 metric, not the official judged accuracy. H-main is the only confirmatory test.", "",
             f"**H-main** bm25_hyb - bm25_lex = {main_c['mean']:+.4f}, 95% CI [{main_c['ci95'][0]:+.4f}, "
             f"{main_c['ci95'][1]:+.4f}] ({main_c['better']} better / {main_c['worse']} worse): **{verdict}**.", "",
             "| Arm | Score | Answerable | Correct refusals | Refused ans. | Calls | Tokens/q | Evidence recall |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for a in ARMS:
        s = summary[a]
        rc = "--" if s["evidence_recall"] is None else f"{s['evidence_recall']:.3f}"
        lines.append(f"| {a} | {s['score']:.4f} | {s['answerable']:.4f} | {s['correct_refusals']}/{s['refusal_n']} | "
                     f"{s['refused_answerable']} | {s['calls']} | {s['tokens_per_q']:.0f} | {rc} |")
    lines += ["", "Exploratory contrasts (95% paired bootstrap, not used for claims):", ""]
    for k, c in result["exploratory"].items():
        lines.append(f"- {k}: {c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] "
                     f"({c['better']} better / {c['worse']} worse)")
    (OUT / "CARE_v3_lmes_s_20261121.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
