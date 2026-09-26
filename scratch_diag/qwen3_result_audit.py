"""LLM-free audit of the ds_fixed / Qwen3-4B-Instruct-2507 static-eval results.

Checks, straight from the artifacts on disk:
  1. the real category distribution of locomo10.json + sample questions, to
     verify the CATEGORY_NAMES mapping used by the harness;
  2. how the adversarial gold is built (`adversarial_answer` vs `answer`);
  3. answer-length / abstention / judge stats per system and per category;
  4. paired McNemar + bootstrap on ASEM vs FullContext judge verdicts;
  5. judge calibration: does judge=True imply the gold is actually present?

Usage:  python scratch_diag/qwen3_result_audit.py
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import statistics
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "datasets", "locomo", "locomo10.json")
RES = os.path.join(ROOT, "data", "benchmarks", "results", "static", "locomo10")
TAG = "ds_fixed__qwen_qwen3_4b_instruct_2507"
SYSTEMS = ["ASEM", "FastASEM", "FullContext", "NoMemory"]

CATEGORY_NAMES = {1: "single_hop", 2: "temporal", 3: "commonsense", 4: "conversational", 5: "adversarial"}
ORDER = ["adversarial", "temporal", "commonsense", "single_hop", "conversational"]


def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------- abstention
try:  # use the production heuristic when importable
    from asem.answer_agent import is_abstention  # type: ignore
except Exception:  # pragma: no cover - fallback mirrors the real one
    _REFUSAL_START_RE = re.compile(
        r"^\s*(i\s*(don'?t|do not)\s*know|i\s*(cannot|can'?t|am unable|'m unable)|"
        r"not\s+(mentioned|specified|stated|enough|clear)|unable to (determine|answer)|"
        r"no\s+information|insufficient\s+information|unknown)\b",
        re.I,
    )
    _NOTE_GAP_RE = re.compile(
        r"(notes?|memory|memories|context|information)\s+(do(es)?\s+not|don'?t|doesn'?t)|"
        r"no\s+(note|memory|mention|record|information)|not\s+(mentioned|specified|stated|provided)",
        re.I,
    )

    def is_abstention(answer: str, *, max_chars: int = 240) -> bool:
        text = (answer or "").strip()
        if not text:
            return True
        if _REFUSAL_START_RE.match(text):
            return True
        return bool(_NOTE_GAP_RE.search(text)) and len(text) <= max_chars


def norm(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def gold_in_pred(ref: str, pred: str) -> bool:
    r, p = norm(ref), norm(pred)
    if not r:
        return False
    if r in p:
        return True
    # all numeric tokens of the ref present (dates / counts)
    nums = re.findall(r"\d+", r)
    if nums and all(n in p for n in nums) and len(r) <= 40:
        return True
    return False


def mcnemar(a: int, b: int, c: int, d: int):
    """Exact binomial two-sided p for discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def bootstrap_ci(deltas, iters=2000, seed=0):
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


# ---------------------------------------------------------------- 1. dataset
def audit_dataset():
    print("=" * 78)
    print("1) locomo10.json — category distribution and label sanity check")
    print("=" * 78)
    data = json.load(open(DATASET, "r", encoding="utf-8"))
    counts = Counter()
    samples = defaultdict(list)
    evidence_counts = defaultdict(list)
    adv_same = adv_diff = 0
    adv_examples = []
    for rec in data:
        for qa in rec.get("qa", []):
            c = qa.get("category", 1)
            counts[c] += 1
            ev = qa.get("evidence", []) or []
            evidence_counts[c].append(len(ev))
            if len(samples[c]) < 4:
                samples[c].append(qa)
            if c == 5:
                if str(qa.get("adversarial_answer", "")).strip() == str(qa.get("answer", "")).strip():
                    adv_same += 1
                else:
                    adv_diff += 1
                if len(adv_examples) < 6:
                    adv_examples.append(qa)
    total = sum(counts.values())
    print(f"total QA = {total}")
    for c in sorted(counts):
        ev = evidence_counts[c]
        multi = sum(1 for e in ev if e > 1)
        print(f"  category {c} -> harness label {CATEGORY_NAMES.get(c)!r:16s} n={counts[c]:4d}  "
              f"evidence turns: mean {statistics.mean(ev):.2f}  >1 on {100*multi/len(ev):.0f}%")
    print("\n  samples per category:")
    for c in sorted(counts):
        print(f"  --- category {c} ({CATEGORY_NAMES.get(c)}) ---")
        for qa in samples[c]:
            print(f"      Q: {qa.get('question')}")
            print(f"        answer={qa.get('answer')!r}  evidence={qa.get('evidence')}")
    print(f"\n  adversarial (cat 5): adversarial_answer == answer on {adv_same}/{adv_same + adv_diff} rows")
    print("  adversarial samples:")
    for qa in adv_examples:
        print(f"      Q: {qa.get('question')}")
        print(f"        answer={qa.get('answer')!r}  adversarial_answer={qa.get('adversarial_answer')!r}")
    return counts


# ---------------------------------------------------------------- 2. systems
def load_all():
    preds, scores = {}, {}
    for s in SYSTEMS:
        preds[s] = {r["idx"]: r for r in load_jsonl(os.path.join(RES, "preds", f"{TAG}__{s}.jsonl"))}
        scores[s] = {r["idx"]: r for r in load_jsonl(os.path.join(RES, "scores", f"{TAG}__{s}.jsonl"))}
    return preds, scores


def audit_lengths(preds, scores):
    print()
    print("=" * 78)
    print("2) answer style: length / abstention / judge, overall and per category")
    print("=" * 78)
    hdr = f"{'system':<12}{'n':>5}{'chars p50':>11}{'words p50':>11}{'abst%':>8}{'judge%':>8}{'F1':>8}{'EM':>7}"
    print(hdr)
    for s in SYSTEMS:
        rows = list(preds[s].values())
        chars = [len(r["pred"]) for r in rows]
        words = [len(r["pred"].split()) for r in rows]
        abst = sum(1 for r in rows if is_abstention(r["pred"]))
        jc = [scores[s][r["idx"]]["judge_correct"] for r in rows if scores[s][r["idx"]].get("judge_correct") is not None]
        f1 = statistics.mean(r.get("f1", 0.0) for r in rows)
        em = statistics.mean(r.get("em", 0.0) for r in rows)
        print(f"{s:<12}{len(rows):>5}{statistics.median(chars):>11.0f}{statistics.median(words):>11.0f}"
              f"{100*abst/len(rows):>7.1f}%{100*sum(jc)/len(jc):>7.1f}%{f1:>8.3f}{em:>7.4f}")

    print()
    print(f"{'category':<15}{'n':>5}  " + "  ".join(f"{s[:9]:>9}" for s in SYSTEMS))
    print("  judge% per category")
    order = ["adversarial", "temporal", "commonsense", "single_hop", "conversational"]
    for cat in order:
        idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == cat]
        cells = []
        for s in SYSTEMS:
            jc = [scores[s][i]["judge_correct"] for i in idxs if scores[s][i].get("judge_correct") is not None]
            cells.append(f"{100*sum(jc)/len(jc):>8.1f}%")
        print(f"{cat:<15}{len(idxs):>5}  " + "  ".join(cells))
    print("  median chars per category")
    for cat in order:
        idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == cat]
        cells = [f"{statistics.median([len(preds[s][i]['pred']) for i in idxs]):>9.0f}" for s in SYSTEMS]
        print(f"{cat:<15}{len(idxs):>5}  " + "  ".join(cells))
    print("  abstention% per category")
    for cat in order:
        idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == cat]
        cells = [f"{100*sum(1 for i in idxs if is_abstention(preds[s][i]['pred']))/len(idxs):>8.1f}%" for s in SYSTEMS]
        print(f"{cat:<15}{len(idxs):>5}  " + "  ".join(cells))

    print()
    print("  per category: abstained n / judge% | answered n / judge%")
    for cat in order:
        idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == cat]
        cells = []
        for s in SYSTEMS:
            ab = [i for i in idxs if is_abstention(preds[s][i]["pred"])]
            an = [i for i in idxs if not is_abstention(preds[s][i]["pred"])]
            abj = [scores[s][i]["judge_correct"] for i in ab if scores[s][i].get("judge_correct") is not None]
            anj = [scores[s][i]["judge_correct"] for i in an if scores[s][i].get("judge_correct") is not None]
            cells.append(f"{len(ab):>3}/{100*sum(abj)/max(1,len(abj)):>3.0f}%|{len(an):>4}/{100*sum(anj)/max(1,len(anj)):>3.0f}%")
        print(f"{cat:<15}{len(idxs):>5}  " + "  ".join(f"{c:>15}" for c in cells))


# ---------------------------------------------------------------- 3. paired
def audit_paired(preds, scores):
    print()
    print("=" * 78)
    print("3) paired judge comparison ASEM vs FullContext (McNemar + bootstrap)")
    print("=" * 78)
    print(f"{'subset':<18}{'n':>6}{'ASEM':>8}{'FC':>8}{'diff':>8}{'b:ASEMonly':>13}{'c:Conly':>9}{'p':>9}{'95% CI':>18}")
    subsets = {"ALL": None, "adversarial": "adversarial", "temporal": "temporal",
               "commonsense": "commonsense", "single_hop": "single_hop",
               "conversational": "conversational", "no adversarial": "!adversarial"}
    for label, cat in subsets.items():
        idxs = []
        for i, r in preds["ASEM"].items():
            if cat is None:
                idxs.append(i)
            elif cat == "!adversarial":
                if r.get("category_name") != "adversarial":
                    idxs.append(i)
            elif r.get("category_name") == cat:
                idxs.append(i)
        a = b = c = d = 0
        deltas = []
        for i in idxs:
            x = scores["ASEM"][i].get("judge_correct")
            y = scores["FullContext"][i].get("judge_correct")
            if x is None or y is None:
                continue
            deltas.append(int(x) - int(y))
            if x and y:
                a += 1
            elif x and not y:
                b += 1
            elif not x and y:
                c += 1
            else:
                d += 1
        n = a + b + c + d
        asem = (a + b) / n
        fc = (a + c) / n
        lo, hi = bootstrap_ci(deltas)
        p = mcnemar(a, b, c, d)
        print(f"{label:<18}{n:>6}{asem:>8.3f}{fc:>8.3f}{asem-fc:>+8.3f}{b:>13}{c:>9}{p:>9.4f}"
              f"{f'[{lo:+.3f},{hi:+.3f}]':>18}")


# ---------------------------------------------------------------- 4. judge QA
def audit_judge(preds, scores):
    print()
    print("=" * 78)
    print("4) judge calibration — is judge=True supported by the text?")
    print("=" * 78)
    for s in SYSTEMS:
        rows = list(preds[s].values())
        tp = fp = tn = fn = 0
        for r in rows:
            j = scores[s][r["idx"]].get("judge_correct")
            if j is None:
                continue
            g = gold_in_pred(r["ref"], r["pred"])
            if j and g:
                tp += 1
            elif j and not g:
                fp += 1
            elif not j and g:
                fn += 1
            else:
                tn += 1
        tot = tp + fp + tn + fn
        print(f"{s:<12} judge=True&gold_present {tp:>5} ({100*tp/tot:>5.1f}%) | "
              f"judge=True&gold_ABSENT {fp:>4} ({100*fp/tot:>4.1f}%) | "
              f"judge=False&gold_present {fn:>4} ({100*fn/tot:>4.1f}%) | "
              f"judge=False&absent {tn:>5} ({100*tn/tot:>5.1f}%)")

    print()
    print("abstention vs answering, per category (abstained n/judge% | answered n/judge%)")
    for cat in ORDER:
        idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == cat]
        cells = []
        for s in SYSTEMS:
            ab = [i for i in idxs if is_abstention(preds[s][i]["pred"])]
            an = [i for i in idxs if not is_abstention(preds[s][i]["pred"])]
            abj = [scores[s][i]["judge_correct"] for i in ab if scores[s][i].get("judge_correct") is not None]
            anj = [scores[s][i]["judge_correct"] for i in an if scores[s][i].get("judge_correct") is not None]
            cells.append(f"{len(ab)}/{100*sum(abj)/max(1,len(abj)):.0f}%|{len(an)}/{100*sum(anj)/max(1,len(anj)):.0f}%")
        print(f"{cat:<15}{len(idxs):>5}  " + "  ".join(f"{c:>13}" for c in cells))

    print()
    print("abstention handling (per system): judge accuracy on abstained answers vs answered")
    for s in SYSTEMS:
        rows = list(preds[s].values())
        ab = [r for r in rows if is_abstention(r["pred"])]
        an = [r for r in rows if not is_abstention(r["pred"])]
        abj = [scores[s][r["idx"]]["judge_correct"] for r in ab if scores[s][r["idx"]].get("judge_correct") is not None]
        anj = [scores[s][r["idx"]]["judge_correct"] for r in an if scores[s][r["idx"]].get("judge_correct") is not None]
        print(f"  {s:<12} abstained n={len(ab):>4} judge_correct={100*sum(abj)/max(1,len(abj)):>5.1f}%   "
              f"answered n={len(an):>4} judge_correct={100*sum(anj)/max(1,len(anj)):>5.1f}%")

    print()
    print("adversarial detail — ASEM vs FastASEM answer behaviour (cat 5)")
    idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == "adversarial"]
    for s in ["ASEM", "FastASEM"]:
        rows = [preds[s][i] for i in idxs]
        starts_no = sum(1 for r in rows if re.match(r"^\s*no\b", r["pred"], re.I))
        starts_idk = sum(1 for r in rows if re.match(r"^\s*i\s*(don'?t|do not)\s*know", r["pred"], re.I))
        jc = [scores[s][i]["judge_correct"] for i in idxs if scores[s][i].get("judge_correct") is not None]
        print(f"  {s:<10} judge={100*sum(jc)/len(jc):>5.1f}%   starts with 'No'={starts_no:>3}   "
              f"starts with 'I don't know'={starts_idk:>3}")

    print()
    print("sample adversarial rows where ASEM is judged correct but FullContext is not")
    shown = 0
    for i in idxs:
        if scores["ASEM"][i].get("judge_correct") and not scores["FullContext"][i].get("judge_correct"):
            r = preds["ASEM"][i]
            print(f"  #{i} Q: {r['question']}")
            print(f"       gold={r['ref']!r}")
            print(f"       ASEM={r['pred'][:150]!r}")
            print(f"       FC  ={preds['FullContext'][i]['pred'][:150]!r}")
            print(f"       judge: {scores['ASEM'][i]['judge_reasoning'][:160]}")
            shown += 1
            if shown >= 7:
                break

    print()
    print("sample rows where the judge accepts a verbose answer whose F1 is ~0")
    shown = 0
    for i, r in preds["ASEM"].items():
        if scores["ASEM"][i].get("judge_correct") and r.get("f1", 0) < 0.05:
            print(f"  #{i} [{r.get('category_name')}] Q: {r['question'][:90]}")
            print(f"       gold={r['ref']!r} f1={r['f1']:.3f}")
            print(f"       pred={r['pred'][:150]!r}")
            shown += 1
            if shown >= 7:
                break


def audit_failure_modes(preds, scores):
    print()
    print("=" * 78)
    print("5) judge_error mode per system / category (why the judge said no)")
    print("=" * 78)
    for s in SYSTEMS:
        modes = Counter()
        for r in preds[s].values():
            e = scores[s][r["idx"]].get("judge_error")
            if e:
                modes[e] += 1
        tot = sum(modes.values()) or 1
        print(f"  {s:<12} " + "  ".join(f"{k}={v} ({100*v/tot:.0f}%)" for k, v in modes.most_common()))

    print()
    print("  per category (ASEM vs FullContext), judge_error mode shares")
    for cat in ORDER:
        idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == cat]
        line = f"  {cat:<15}"
        for s in ["ASEM", "FastASEM", "FullContext"]:
            modes = Counter()
            for i in idxs:
                e = scores[s][i].get("judge_error")
                if e:
                    modes[e] += 1
            tot = sum(modes.values()) or 1
            share = ", ".join(f"{k[:8]}:{100*v/tot:.0f}%" for k, v in modes.most_common())
            line += f"  {s[:4]}={share}"
        print(line)

    print()
    print("  near-miss / dilution: judge=False rows by token-F1 band")
    print(f"  {'system':<12}{'F1<0.2':>9}{'0.2-0.5':>9}{'>0.5':>8}   (judge=False only)")
    for s in SYSTEMS:
        bands = [0, 0, 0]
        for r in preds[s].values():
            if scores[s][r["idx"]].get("judge_correct"):
                continue
            f = r.get("f1", 0.0)
            bands[0 if f < 0.2 else (1 if f < 0.5 else 2)] += 1
        print(f"  {s:<12}{bands[0]:>9}{bands[1]:>9}{bands[2]:>8}")


def audit_targeted(preds, scores):
    print()
    print("=" * 78)
    print("6) targeted checks")
    print("=" * 78)

    print("  (a) temporal rows with judge=True: is the gold's number actually in the answer?")
    for s in SYSTEMS:
        idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == "temporal"]
        hit = miss = 0
        for i in idxs:
            if not scores[s][i].get("judge_correct"):
                continue
            r = preds[s][i]
            nums = re.findall(r"\d+", r["ref"])
            if not nums:
                continue
            if all(n in r["pred"] for n in nums):
                hit += 1
            else:
                miss += 1
        print(f"    {s:<12} number present in pred={hit:>3}  NUMBER ABSENT but judged correct={miss:>3}")

    print()
    print("  (b) suspicious temporal acceptances (gold year absent from answer), ASEM")
    shown = 0
    for i, r in preds["ASEM"].items():
        if r.get("category_name") != "temporal" or not scores["ASEM"][i].get("judge_correct"):
            continue
        nums = re.findall(r"\d{4}", r["ref"])
        if nums and any(n not in r["pred"] for n in nums):
            print(f"    #{i} Q: {r['question'][:80]}")
            print(f"         gold={r['ref']!r}  pred={r['pred'][:110]!r}")
            print(f"         judge: {scores['ASEM'][i]['judge_reasoning'][:140]}")
            shown += 1
            if shown >= 6:
                break

    print()
    print("  (c) FastASEM adversarial abstentions (29.6% of cat 5) — what do they look like?")
    idxs = [i for i, r in preds["ASEM"].items() if r.get("category_name") == "adversarial"]
    shown = 0
    for i in idxs:
        r = preds["FastASEM"][i]
        if not is_abstention(r["pred"]):
            continue
        print(f"    #{i} Q: {r['question'][:85]}")
        print(f"         gold={r['ref']!r}")
        print(f"         pred={r['pred'][:130]!r}")
        shown += 1
        if shown >= 6:
            break

    print()
    print("  (d) single_hop (cat 4) rows: ASEM wrong / FullContext right — does ASEM even answer?")
    shown = 0
    for i, r in preds["ASEM"].items():
        if r.get("category_name") != "conversational":
            continue
        if scores["ASEM"][i].get("judge_correct") or not scores["FullContext"][i].get("judge_correct"):
            continue
        print(f"    #{i} Q: {r['question'][:85]}")
        print(f"         gold={r['ref']!r}")
        print(f"         ASEM({len(r['pred'])}c)={r['pred'][:110]!r}")
        print(f"         FC  ({len(preds['FullContext'][i]['pred'])}c)={preds['FullContext'][i]['pred'][:110]!r}")
        shown += 1
        if shown >= 8:
            break


def main():
    audit_dataset()
    preds, scores = load_all()
    audit_lengths(preds, scores)
    audit_paired(preds, scores)
    audit_judge(preds, scores)
    audit_failure_modes(preds, scores)
    audit_targeted(preds, scores)


if __name__ == "__main__":
    main()
