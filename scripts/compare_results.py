"""Compare all LoCoMo benchmark result files under data/benchmarks/results.

Loads:
  - data/benchmarks/results/*.json   (aggregate result files)
  - data/benchmarks/results/preds/*.jsonl (per-question prediction files)

Computes consistent EM (substring-aware, normalized) and ROUGE-L (LCS F1)
for every per-question file, and prints a comparison table grouped by
benchmark (locomo10 vs full locomo) and by conversation coverage.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "benchmarks" / "results"
PREDS = RESULTS / "preds"


def normalize_text(text) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower().strip()
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def compute_em(pred, ref) -> float:
    norm_p = normalize_text(pred)
    norm_r = normalize_text(ref)
    if not norm_p or not norm_r:
        return 0.0
    if norm_p == norm_r:
        return 1.0
    if norm_r in norm_p or norm_p in norm_r:
        return 1.0
    return 0.0


def compute_rouge_l(pred, ref) -> float:
    p_tokens = normalize_text(pred).split()
    r_tokens = normalize_text(ref).split()
    if not p_tokens or not r_tokens:
        return 0.0
    m, n = len(p_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == r_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return (2 * prec * rec) / (prec + rec)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_parse_error": line[:80]})
    return rows


def summarize_jsonl(rows):
    """Compute consistent metrics over per-question rows."""
    valid = [r for r in rows if "_parse_error" not in r]
    n = len(valid)
    if n == 0:
        return {"n": 0, "em": 0.0, "rouge_l": 0.0, "null_preds": 0, "sessions": set()}
    ems, rls, nulls = [], [], 0
    sessions = set()
    for r in valid:
        pred = r.get("pred")
        ref = r.get("ref")
        sessions.add(r.get("session_id", "?"))
        if pred is None:
            nulls += 1
            ems.append(0.0)
            rls.append(0.0)
            continue
        ems.append(compute_em(pred, ref))
        rls.append(compute_rouge_l(pred, ref))
    return {
        "n": n,
        "em": 100.0 * sum(ems) / n,
        "rouge_l": 100.0 * sum(rls) / n,
        "null_preds": nulls,
        "sessions": sessions,
    }


def main():
    print("=" * 100)
    print("PER-QUESTION PREDICTION FILES (preds/*.jsonl) — consistent metrics")
    print("=" * 100)
    header = f"{'file':<38} {'n':>4} {'EM%':>7} {'ROUGE-L%':>9} {'null':>5}  sessions"
    print(header)
    print("-" * 100)
    for path in sorted(PREDS.glob("*.jsonl")):
        rows = load_jsonl(path)
        s = summarize_jsonl(rows)
        sess = ",".join(sorted(s["sessions"])) if s["sessions"] else "-"
        if len(sess) > 40:
            sess = sess[:37] + "..."
        print(f"{path.name:<38} {s['n']:>4} {s['em']:>7.1f} {s['rouge_l']:>9.2f} {s['null_preds']:>5}  {sess}")

    print()
    print("=" * 100)
    print("AGGREGATE JSON FILES (results/*.json)")
    print("=" * 100)
    for path in sorted(RESULTS.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"{path.name}: PARSE ERROR {e}")
            continue
        print(f"\n--- {path.name} ---")
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict):
                    em = val.get("em", val.get("em_score"))
                    rl = val.get("rougeL", val.get("rouge_l"))
                    extra = ""
                    if "judge_score" in val:
                        extra += f" judge={val['judge_score']}"
                    if "total_questions" in val:
                        extra += f" n={val['total_questions']}"
                    if "bank_size" in val:
                        extra += f" bank={val['bank_size']}"
                    if "__n__" in val:
                        extra += f" partial={val['__n__']}/{val.get('__total__', '?')}"
                    print(f"  {key:<40} em={em}  rougeL={rl}{extra}")
                else:
                    print(f"  {key}: {val}")
        else:
            print(f"  {data}")


if __name__ == "__main__":
    main()
