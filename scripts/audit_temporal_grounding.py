"""LLM-free audit for the temporal-grounding pipeline and run results.

Why this exists
---------------
Two classes of evidence had to be re-derived by hand during the last
investigation, which made them slow and easy to get wrong:

1. **Does every session date in a corpus actually parse?** LongMemEval's
   ``2023/02/20 (Mon) 12:15`` format failed 100% of the time and fell through to
   ``datetime.now()``, so every note in the bank carried the benchmark run date.
2. **How much of a result file is the "date arithmetic" slice, and how did it
   score?** The failing cases are a specific, measurable subset, not the whole
   temporal category.

This script answers both without an LLM, an API key, or the dataset being
mounted for the result-file mode.

Usage
-----
    # 1. Corpus date coverage (before/after the parser fix)
    python scripts/audit_temporal_grounding.py --corpus locomo longmemeval

    # 2. Score a finished run, split by arithmetic vs. ordering questions
    python scripts/audit_temporal_grounding.py --preds \
        data/benchmarks/results/longmemeval/check_thg_judge2/preds/*.jsonl

    # 3. Both, with the arithmetic slice written out for inspection
    python scripts/audit_temporal_grounding.py --corpus longmemeval --preds <file> --dump-slice

Outputs a human-readable report; exits non-zero only on an internal error, so it
is safe to run in a pipeline.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from asem.temporal import parse_session_datetime  # noqa: E402

# ---------------------------------------------------------------------------
# Question classification
# ---------------------------------------------------------------------------
# "How many days between X and Y", "how many days before/after ..." — these need
# ARITHMETIC on two resolved dates, which is precisely what a wrong session date
# destroys. They are the slice that failed on LongMemEval.
_ARITHMETIC_RE = re.compile(
    r"how\s+(?:many|much|long)"
    r"[^?]{0,60}?"
    r"\b(?:days?|weeks?|months?|years?|hours?|minutes?)\b"
    r"|how\s+(?:many|much|long)\s+(?:days?|weeks?|months?|years?|hours?|minutes?)\b"
    r"|\b(?:days?|weeks?|months?|years?)\s+(?:before|after|between)\b"
    r"|\bhow\s+long\s+(?:before|after|between|had|did|was|were)\b",
    re.IGNORECASE,
)
# "Which came first, X or Y?" — needs ORDERING only. These survived on
# LongMemEval because a comparison of two equally-corrupted dates is still
# correct, which is exactly why they are reported as a separate control group.
_ORDERING_RE = re.compile(
    r"which\s+(?:one\s+)?(?:did|do|was|were|event|vehicle|device|seed|item)"
    r"|which\s+came\s+first"
    r"|\b(?:first|earlier|earliest|before|prior|order)\b"
    r"|\bwhich\b[^?]{0,40}\b(?:first|before)\b",
    re.IGNORECASE,
)


def classify_question(question: str) -> str:
    """Return ``arithmetic`` | ``ordering`` | ``temporal-other`` | ``non-temporal``."""
    text = question or ""
    if _ARITHMETIC_RE.search(text):
        return "arithmetic"
    if _ORDERING_RE.search(text):
        return "ordering"
    return "temporal-other" if re.search(r"\b(when|date|time)\b", text, re.IGNORECASE) else "non-temporal"


def _numbers(text: str) -> List[float]:
    out = []
    for match in re.finditer(r"\d+(?:[.,]\d+)?", text or ""):
        try:
            out.append(float(match.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def _gold_numbers(ref: str) -> List[float]:
    """Numbers in a gold answer, ignoring out-of-range tokens like years."""
    return [n for n in _numbers(ref) if 0 <= n <= 4000]


def numeric_hit(pred: str, ref: str) -> Optional[bool]:
    """Does the prediction contain any number the gold accepts? None if no numbers."""
    gold = _gold_numbers(ref)
    if not gold:
        return None
    pred_nums = _numbers(pred)
    if not pred_nums:
        return False
    # LongMemEval lists alternatives, e.g. "7 days. 8 days (including the last
    # day) is also acceptable." — a match against ANY alternative is correct.
    return any(any(abs(p - g) < 1e-6 for g in gold) for p in pred_nums)


# ---------------------------------------------------------------------------
# Mode 1 — corpus date coverage
# ---------------------------------------------------------------------------

_CORPORA = {
    "longmemeval": [
        "datasets/longmemeval/longmemeval_oracle.json",
        "datasets/longmemeval/longmemeval_s_cleaned.json",
    ],
    "locomo": ["datasets/locomo/locomo10.json"],
}


def audit_corpus(name: str, path: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Report how many session dates in a corpus parse, grouped by source field."""
    if not os.path.exists(path):
        return {"corpus": name, "path": path, "error": "missing"}

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    checked = 0
    failures: List[str] = []
    by_field: Dict[str, List[int]] = defaultdict(lambda: [0, 0])  # field -> [ok, bad]

    def check(value: str, field: str) -> None:
        nonlocal checked
        if not value or not str(value).strip():
            return
        checked += 1
        by_field[field][0] += 1
        _dt, iso = parse_session_datetime(value)
        if iso is None:
            by_field[field][1] += 1
            if len(failures) < 10:
                failures.append(str(value)[:60])

    records = data if isinstance(data, list) else [data]
    if limit:
        records = records[:limit]

    if name == "longmemeval":
        for record in records:
            for raw in record.get("haystack_dates") or []:
                check(raw, "haystack_dates")
            check(record.get("question_date", ""), "question_date")
    else:  # locomo
        for record in records:
            conv = record.get("conversation", {}) or {}
            for key, value in conv.items():
                if key.endswith("_date_time"):
                    check(value, "session_*_date_time")

    return {
        "corpus": name,
        "path": path,
        "records": len(records),
        "checked": checked,
        "by_field": {k: {"ok": v[0], "bad": v[1]} for k, v in sorted(by_field.items())},
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Mode 2 — audit a finished run's predictions
# ---------------------------------------------------------------------------

def _load_sibling_scores(path: str) -> Dict[int, Dict[str, Any]]:
    """Load judge verdicts from the run's ``scores/`` file, keyed by ``idx``.

    The runner writes predictions to ``<root>/preds/<name>.jsonl`` and judge
    verdicts to ``<root>/scores/<name>.jsonl``. Joining them here means the audit
    can report judge accuracy from the same run without the caller having to
    pass both paths (the earlier version silently reported ``n/a`` for judge).
    """
    directory, filename = os.path.split(os.path.abspath(path))
    root = os.path.dirname(directory)  # .../<tag>/preds -> .../<tag>
    candidate = os.path.join(root, "scores", filename)
    if not os.path.exists(candidate):
        return {}

    verdicts: Dict[int, Dict[str, Any]] = {}
    try:
        with open(candidate, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                idx = record.get("idx")
                if isinstance(idx, int):
                    verdicts[idx] = record
    except (OSError, json.JSONDecodeError):
        return {}
    return verdicts


def audit_preds(path: str, dump_slice: bool = False) -> Dict[str, Any]:
    """Split a prediction file by question kind and report numeric accuracy."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {"path": path, "error": "empty"}

    # Judge verdicts live in a sibling file; merge them so judge accuracy and
    # self-contradiction counts are available for real runs.
    verdicts_by_idx = _load_sibling_scores(path)
    for row in rows:
        verdict = verdicts_by_idx.get(row.get("idx"))
        if not verdict:
            continue
        for key in ("judge_correct", "judge_error", "judge_reasoning"):
            if key in verdict and row.get(key) is None:
                row[key] = verdict[key]

    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "numeric_ok": 0, "numeric_n": 0, "judge_ok": 0, "judge_n": 0}
    )
    slice_rows: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []

    for row in rows:
        question = row.get("question") or row.get("query") or ""
        pred = str(row.get("pred") or row.get("ai_response") or "")
        ref = str(row.get("ref") or row.get("expected_answer") or "")
        # LongMemEval's question_type is the more reliable signal when present.
        kind = classify_question(question)
        qtype = str(row.get("question_type") or row.get("category_name") or "")
        if kind == "non-temporal" and qtype:
            kind = "temporal-other" if "temporal" in qtype.lower() else "non-temporal"

        bucket = buckets[kind]
        bucket["n"] += 1

        hit = numeric_hit(pred, ref)
        if hit is not None:
            bucket["numeric_n"] += 1
            bucket["numeric_ok"] += int(hit)

        verdict = None
        for key in ("judge_correct", "judge"):
            if key in row and row[key] is not None:
                verdict = bool(row[key])
                break
        if verdict is not None:
            bucket["judge_n"] += 1
            bucket["judge_ok"] += int(verdict)

        if kind == "arithmetic":
            slice_rows.append(
                {"idx": row.get("idx"), "question": question, "gold": ref, "pred": pred,
                 "numeric_ok": hit, "judge": verdict}
            )

        # Judge self-contradiction: correct=True while an error mode is named.
        error = row.get("judge_error")
        if verdict is True and error:
            disagreements.append({"idx": row.get("idx"), "error": error, "gold": ref, "pred": pred})

    report: Dict[str, Any] = {"path": path, "rows": len(rows), "buckets": {}}
    for kind, bucket in sorted(buckets.items()):
        report["buckets"][kind] = {
            **{k: v for k, v in bucket.items() if k != "n"},
            "n": bucket["n"],
            "numeric_acc": (bucket["numeric_ok"] / bucket["numeric_n"]) if bucket["numeric_n"] else None,
            "judge_acc": (bucket["judge_ok"] / bucket["judge_n"]) if bucket["judge_n"] else None,
        }
    report["judge_self_contradictions"] = len(disagreements)
    if dump_slice:
        report["arithmetic_slice"] = slice_rows
    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(corpus_reports: Sequence[Dict[str, Any]], pred_reports: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Temporal grounding audit", ""]

    for report in corpus_reports:
        lines.append(f"## Corpus `{report['corpus']}`")
        if report.get("error"):
            lines.append(f"- **{report['error']}**: `{report['path']}`\n")
            continue
        bad = sum(v["bad"] for v in report["by_field"].values())
        lines.append(f"- `{report['path']}` — {report['records']} records")
        lines.append(f"- dates checked: **{report['checked']}**, unparseable: **{bad}**")
        for field, counts in report["by_field"].items():
            status = "OK" if counts["bad"] == 0 else "FAIL"
            lines.append(f"  - `{field}`: {counts['ok']} ok / {counts['bad']} unparseable — {status}")
        if report["failures"]:
            lines.append(f"  - examples: {report['failures'][:5]}")
        lines.append("")

    for report in pred_reports:
        lines.append(f"## Predictions `{os.path.basename(report['path'])}`")
        if report.get("error"):
            lines.append(f"- **{report['error']}**\n")
            continue
        lines.append(f"- rows: {report['rows']}")
        lines.append("")
        lines.append("| kind | n | numeric acc | judge acc |")
        lines.append("|---|---|---|---|")
        for kind, bucket in report["buckets"].items():
            num = bucket["numeric_acc"]
            jud = bucket["judge_acc"]
            num_cell = (
                "n/a"
                if num is None
                else f"{num:.3f} ({bucket['numeric_ok']}/{bucket['numeric_n']})"
            )
            jud_cell = (
                "n/a"
                if jud is None
                else f"{jud:.3f} ({bucket['judge_ok']}/{bucket['judge_n']})"
            )
            lines.append(f"| {kind} | {bucket['n']} | {num_cell} | {jud_cell} |")
        lines.append("")
        lines.append(
            f"- judge self-contradictions (correct=true with an error mode): "
            f"**{report['judge_self_contradictions']}**"
        )
        if report.get("arithmetic_slice"):
            lines.append("")
            lines.append("### Arithmetic slice (the date-difference questions)")
            lines.append("")
            lines.append("| idx | gold | pred | numeric | judge |")
            lines.append("|---|---|---|---|---|")
            for item in report["arithmetic_slice"]:
                gold = str(item["gold"])[:44].replace("|", "/")
                pred = str(item["pred"])[:44].replace("|", "/")
                lines.append(f"| {item['idx']} | {gold} | {pred} | {item['numeric_ok']} | {item['judge']} |")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", nargs="*", choices=sorted(_CORPORA), default=[],
                        help="Audit session-date coverage for these corpora")
    parser.add_argument("--preds", nargs="*", default=[],
                        help="Prediction .jsonl files (globs allowed) to audit")
    parser.add_argument("--limit", type=int, default=None, help="Cap corpus records")
    parser.add_argument("--dump-slice", action="store_true", help="Include the arithmetic Q&A slice")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown")
    parser.add_argument("--out", default=None, help="Write the report to this file")
    args = parser.parse_args()

    corpus_reports = []
    for name in args.corpus:
        for path in _CORPORA[name]:
            corpus_reports.append(audit_corpus(name, path, limit=args.limit))

    pred_reports = []
    for pattern in args.preds:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            if os.path.exists(path):
                pred_reports.append(audit_preds(path, dump_slice=args.dump_slice))

    if not corpus_reports and not pred_reports:
        parser.error("nothing to do — pass --corpus and/or --preds")

    text = (
        json.dumps({"corpus": corpus_reports, "preds": pred_reports}, indent=2)
        if args.json
        else render(corpus_reports, pred_reports)
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
