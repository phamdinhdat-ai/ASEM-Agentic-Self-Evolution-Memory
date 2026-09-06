"""Focused comparison: ASEMv2, FastASEM, SimRetrieval, FullContext on conv-26."""
import json
import re
import collections
from pathlib import Path


def norm(t):
    if t is None:
        return ""
    if not isinstance(t, str):
        t = str(t)
    t = t.lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def em(p, r):
    np_, nr = norm(p), norm(r)
    if not np_ or not nr:
        return 0.0
    return 1.0 if (np_ == nr or nr in np_ or np_ in nr) else 0.0


def rl(p, r):
    pt, rt = norm(p).split(), norm(r).split()
    if not pt or not rt:
        return 0.0
    m, n = len(pt), len(rt)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pt[i - 1] == rt[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    l = dp[m][n]
    if l == 0:
        return 0.0
    pr, rc = l / m, l / n
    return 2 * pr * rc / (pr + rc)


files = {
    "ASEMv2": "locomo10_ASEMv2_conv26.jsonl",
    "FastASEM": "locomo10_FastASEM_conv0_qa.jsonl",
    "SimRetrieval": "locomo10_SimRetrieval_conv26.jsonl",
    "FullContext": "locomo10_FullContext_conv26.jsonl",
}
base = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "results" / "preds"
data = {}
for sysname, f in files.items():
    cats = collections.defaultdict(lambda: [0, 0.0, 0.0])
    tot = [0, 0.0, 0.0]
    nulls = 0
    for line in open(base / f, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        c = r.get("category_name", "?")
        pred, ref = r.get("pred"), r.get("ref")
        if pred is None:
            nulls += 1
        for agg in (cats[c], tot):
            agg[0] += 1
            if pred is None:
                continue
            agg[1] += em(pred, ref)
            agg[2] += rl(pred, ref)
    data[sysname] = (tot, cats, nulls)

print("=== conv-26 (117 questions each) - consistent EM / ROUGE-L ===")
print(f"{'system':<14}{'n':>4}{'EM%':>8}{'ROUGE-L%':>10}{'null':>6}")
for s, (tot, cats, nulls) in data.items():
    n, e, ro = tot
    print(f"{s:<14}{n:>4}{100 * e / n:>8.1f}{100 * ro / n:>10.1f}{nulls:>6}")

print()
print("=== per-category (EM% / ROUGE-L%) ===")
catnames = sorted({c for _, cats, _ in data.values() for c in cats})
for c in catnames:
    print(f"-- {c}")
    for s, (tot, cats, nulls) in data.items():
        if c in cats:
            n, e, ro = cats[c]
            print(f"   {s:<14} n={n:<4} EM={100 * e / n:5.1f}%  RL={100 * ro / n:5.1f}%")
