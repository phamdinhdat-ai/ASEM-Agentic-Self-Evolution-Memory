"""Run every ASEM system + baseline on the LongMemEval benchmark.

One-pass, end-to-end runner:

    for each system:
        for each question (each owns its own haystack):
            reset()  -> ingest the haystack -> answer -> score

This mirrors ``scripts/run_locomo10_experiments.py`` but is **additive**: it
reuses ``eval.phase_runner`` (ingestion dispatch, session conversion) and
``eval.systems`` (builders) verbatim, and adds LongMemEval-specific scoring on
top of ``eval.metrics`` (QA metrics + session recall + abstention accuracy).

Usage
-----
    python scripts/run_longmemeval_eval.py \
        --input datasets/longmemeval/longmemeval_oracle.json \
        --tag oracle \
        --config configs/models/deepseek_api.yaml \
        --judge-config configs/models/judge_api.yaml \
        --systems ASEM-THG ASEM ASEMv2 FastASEM NoMemory FullContext \
                  SimRetrieval AtomicLinking RLManagerOnly ValueRetrievalOnly \
        --metrics em f1 rougeL bertscore_f1 judge --per-category --recall

    # Quick smoke test
    python scripts/run_longmemeval_eval.py --limit 5 \
        --systems NoMemory FullContext SimRetrieval

Outputs (under ``--results``)::

    <results>/<tag>__<model>.json          aggregate results
    <results>/<tag>__<model>.md            rendered report table
    <results>/preds/<tag>__<model>__<System>.jsonl   resumable predictions
    <results>/scores/<tag>__<model>__<System>.jsonl  per-row judge verdicts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Bootstrap (project root on sys.path + .env), matching the other runners.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_dotenv_path = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_dotenv_path):
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv_path, override=False)
    except ImportError:
        with open(_dotenv_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from asem.logging_utils import setup_logging  # noqa: E402
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))

from eval.longmemeval import (  # noqa: E402
    LongMemEvalItem,
    build_sessions,
    group_by_session,
    has_answer_turns,
    is_abstention_answer,
    load_longmemeval,
    normalize_session_id,
    note_ids_from_telemetry,
    session_ids_from_telemetry,
    session_recall,
    sessions_for_group,
    turn_recall,
)
from eval.metrics import (  # noqa: E402
    canonical_metrics,
    compute_metrics,
    metric_needs_bertscore,
    metric_needs_judge,
    per_group_metrics,
)
from eval.phase_runner import (  # noqa: E402
    build_backend_from_config,
    close_system,
    ingest_system,
    model_tag_from_config,
)
from eval.systems import (  # noqa: E402
    ALL_SYSTEMS,
    NO_BANK_SYSTEMS,
    build_system,
)

DEFAULT_SYSTEMS: List[str] = list(ALL_SYSTEMS)


# ---------------------------------------------------------------------------
# Ingestion dispatch for one LongMemEval question
# ---------------------------------------------------------------------------

def _ingest_question(system: Any, name: str, sessions: Sequence[Any]) -> None:
    """Ingest one question's haystack via the shared dispatch.

    ``ingest_system`` already routes ASEM-THG/FastASEM -> fast dicts, ASEMv2 ->
    per session with a header, and ASEM v1 + memory baselines -> ``(label,
    turns)`` batches.  We only skip it for the bank-less baselines.
    """
    if name in NO_BANK_SYSTEMS or not sessions:
        return
    ingest_system(system, name, list(sessions))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _system_bank(system: Any) -> Any:
    """Reach the system's ``MemoryBank`` (ASEMPipeline lives behind a wrapper)."""
    bank = getattr(system, "memory_bank", None)
    if bank is None:
        bank = getattr(getattr(system, "pipeline", None), "memory_bank", None)
    return bank


def _retrieved_session_ids(
    item: LongMemEvalItem, system: Any, name: str
) -> Optional[List[str]]:
    """Resolve the answered system's retrieved notes onto haystack session ids.

    ``last_used_notes`` is a list of note **ids**; the session attribution comes
    from the bank.  Following the repo convention (see ``eval/static_eval.py``),
    retrieval telemetry is only exposed by systems that populate it (currently
    ``ASEMTHGSystem``).  For everyone else this returns ``None`` — "not
    measured" — so recall is reported as ``null`` rather than a fake 0.0.
    """
    if name in NO_BANK_SYSTEMS:
        return []
    used = getattr(system, "last_used_notes", None)
    if used is None:
        return None
    raw = session_ids_from_telemetry(used, _system_bank(system))
    return [normalize_session_id(sid, item) for sid in raw]


def _retrieved_texts(system: Any, name: str) -> List[str]:
    """Text of the notes the system actually answered from.

    Feeds ``turn_recall``. This used to be hardcoded to ``[]`` at the call site,
    which forced ``_recall_stats`` down its ``else: 0.0`` branch — so the report
    printed a confident ``turn_recall = 0.0000`` for every system that DOES emit
    telemetry. A metric that can only ever read zero is worse than no metric: it
    looks like a retrieval failure and invites a wrong conclusion.
    """
    if name in NO_BANK_SYSTEMS:
        return []
    used = getattr(system, "last_used_notes", None)
    if not used:
        return []

    texts: List[str] = []
    bank = _system_bank(system)
    for entry in used:
        # ``last_used_notes`` is either note ids or Note objects depending on the
        # system; handle both rather than assuming.
        note = entry
        if isinstance(entry, str):
            note = bank.get_note(entry) if bank is not None else None
        if note is None:
            continue
        # ONLY the raw turn ``c``. Gold is verbatim turn text, so ``X`` (the
        # LLM's paraphrase) and ``session_date`` (metadata) can never match it —
        # including them only inflates the haystack and makes the substring test
        # look better than the retrieval actually is.
        raw = getattr(note, "c", None)
        if raw:
            texts.append(str(raw))
    return texts


def _recall_stats(
    item: LongMemEvalItem,
    name: str,
    retrieved_ids: Optional[Sequence[str]],
    retrieved_texts: Sequence[str],
) -> Dict[str, Optional[float]]:
    """Session-level and (optional) turn-level recall for one answered item.

    Returns ``None`` for every metric the system does not expose, so the report
    distinguishes "system does not emit retrieval telemetry" from a genuine
    retrieval miss (0.0).
    """
    stats: Dict[str, Optional[float]] = {
        "session_recall": None,
        "session_hit": None,
        "turn_recall": None,
    }

    if name in NO_BANK_SYSTEMS:
        # Bank-less baselines "retrieve" their whole transcript by construction.
        if item.gold_session_ids:
            stats["session_recall"] = 1.0
            stats["session_hit"] = 1.0
        return stats

    if retrieved_ids is None:
        return stats

    recall, hit = session_recall(retrieved_ids, item.gold_session_ids)
    stats["session_recall"] = recall
    stats["session_hit"] = hit

    gold_turns = has_answer_turns(item)
    if not gold_turns:
        # Nothing to recall against — report "not measured", not a failure.
        stats["turn_recall"] = None
    elif not retrieved_texts:
        # The system answered but emitted no note text. This is UNMEASURED, and
        # reporting 0.0 here is what produced the misleading all-zero
        # `turn_recall` column in the previous LongMemEval report.
        stats["turn_recall"] = None
    else:
        stats["turn_recall"] = turn_recall(retrieved_texts, gold_turns)
    return stats


def _judge_rows(
    questions: Sequence[str],
    preds: Sequence[str],
    refs: Sequence[str],
    categories: Sequence[str],
    judge_config: str,
) -> Tuple[List[Optional[bool]], List[Dict[str, Any]]]:
    """Run the LLM judge over every prediction; returns (flags, per-row records).

    ``LLMJudge`` is constructed from a backend built off ``judge_config`` (the
    same ``build_backend_from_config`` path the rest of the codebase uses).
    """
    from eval.llm_as_a_judge import LLMJudge

    judge_backend = build_backend_from_config(judge_config)
    judge = LLMJudge(backend=judge_backend)
    flags: List[Optional[bool]] = []
    rows: List[Dict[str, Any]] = []
    for i, (question, pred, ref) in enumerate(zip(questions, preds, refs)):
        row: Dict[str, Any] = {"idx": i, "pred": pred, "ref": ref}
        try:
            verdict = judge.judge(question, ref, pred, question_type=categories[i])
            row["judge_correct"] = bool(verdict.is_correct)
            row["judge_reasoning"] = verdict.reasoning
            row["judge_error"] = verdict.error
            flags.append(row["judge_correct"])
        except Exception as exc:  # noqa: BLE001
            row["judge_correct"] = None
            row["judge_error"] = str(exc)
            flags.append(None)
        rows.append(row)
    return flags, rows


def _score_system(
    name: str,
    preds: List[str],
    refs: List[str],
    enriched: List[Dict[str, Any]],
    metric_names: Sequence[str],
    judge_config: Optional[str],
    judge_enabled: bool,
) -> Dict[str, Any]:
    """Aggregate QA metrics + LongMemEval-specific recall/abstention stats."""
    questions = [rec.get("question", "") for rec in enriched]
    categories = [rec.get("category_name", "") for rec in enriched]
    question_types = categories
    abst_flags = [bool(rec.get("is_abstention")) for rec in enriched]

    bert_scores = None
    if metric_needs_bertscore(metric_names):
        from eval.metrics import bertscore_f1_batch

        bert_scores = bertscore_f1_batch(preds, refs)

    judge_flags = None
    judge_rows: List[Dict[str, Any]] = []
    if judge_enabled and metric_needs_judge(metric_names) and judge_config:
        judge_flags, judge_rows = _judge_rows(questions, preds, refs, categories, judge_config)

    # LongMemEval has no adversarial-refusal category, so no adversarial flags.
    overall = compute_metrics(
        preds,
        refs,
        metric_names,
        bertscore_scores=bert_scores,
        judge_flags=judge_flags,
    )
    per_category = per_group_metrics(
        preds,
        refs,
        question_types,
        metric_names,
        bertscore_scores=bert_scores,
        judge_flags=judge_flags,
    ) if enriched else {}

    # Session / turn recall — ``None`` entries mean "not measured" and are
    # dropped; if nothing was measured the whole metric is ``None``.
    session_recall_mean = _mean_optional([rec.get("session_recall") for rec in enriched])
    session_hit_mean = _mean_optional([rec.get("session_hit") for rec in enriched])
    turn_recall_mean = _mean_optional([rec.get("turn_recall") for rec in enriched])

    # Abstention accuracy: a refusal is correct on an ``_abs`` question and
    # wrong on a normal one.
    abst_total = sum(1 for f in abst_flags if f)
    refused = [is_abstention_answer(p) for p in preds]
    abst_correct = sum(
        1 for refused_i, is_abs in zip(refused, abst_flags) if refused_i == is_abs
    )
    abstention_accuracy = (abst_correct / len(preds)) if preds else 0.0

    # Per-question-type recall + abstention breakdown
    recall_by_cat: Dict[str, Dict[str, Optional[float]]] = {}
    for rec in enriched:
        cat = rec.get("category_name", "") or "unknown"
        bucket = recall_by_cat.setdefault(
            cat,
            {
                "n": 0.0,
                "session_recall": [],
                "turn_recall": [],
                "abstain_rate": 0.0,
            },
        )
        bucket["n"] += 1
        if rec.get("session_recall") is not None:
            bucket["session_recall"].append(float(rec["session_recall"]))
        if rec.get("turn_recall") is not None:
            bucket["turn_recall"].append(float(rec["turn_recall"]))
        bucket["abstain_rate"] += 1.0 if is_abstention_answer(rec.get("pred", "")) else 0.0
    for cat, bucket in recall_by_cat.items():
        n = bucket["n"] or 1.0
        bucket["session_recall"] = _mean_optional(bucket["session_recall"])
        bucket["turn_recall"] = _mean_optional(bucket["turn_recall"])
        bucket["abstain_rate"] /= n

    return {
        "overall": overall,
        "per_category": per_category,
        "recall": {
            "session_recall": session_recall_mean,
            "session_hit": session_hit_mean,
            "turn_recall": turn_recall_mean,
        },
        "recall_by_category": recall_by_cat,
        "abstention": {
            "n_abstention_questions": float(abst_total),
            "refusal_rate": _mean([1.0 if r else 0.0 for r in refused]),
            "abstention_accuracy": abstention_accuracy,
        },
        "n": float(len(preds)),
        "judge_rows": judge_rows,
    }


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return (sum(values) / len(values)) if values else 0.0


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    """Mean of the values that were actually measured.

    ``None`` means "this system did not emit the telemetry"; it is dropped from
    the mean, and an all-``None`` input yields ``None`` (rendered as ``n/a``)
    rather than a misleading ``0.0``.
    """
    vals = [float(v) for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


# ---------------------------------------------------------------------------
# Per-system evaluation loop (resumable)
# ---------------------------------------------------------------------------

def evaluate_system(
    name: str,
    items: List[LongMemEvalItem],
    groups: List[List[Dict[str, Any]]],
    config_path: str,
    db_dir: str,
    preds_dir: str,
    scores_dir: str,
    metric_names: Sequence[str],
    backend: Any,
    tag: str,
    model_tag: str,
    judge_config: Optional[str],
    judge_enabled: bool,
    recall_enabled: bool,
) -> Dict[str, Any]:
    """Answer every LongMemEval question with one system, resumably."""
    preds_path = os.path.join(preds_dir, f"{tag}__{model_tag}__{name}.jsonl")
    enriched: List[Dict[str, Any]] = []
    done: Set[int] = set()

    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                done.add(rec["idx"])
                enriched.append(rec)
        if done:
            print(f"    [{name}] resuming from {len(done)} saved predictions", flush=True)

    # Build the system once; reset() between questions keeps banks isolated.
    system = build_system(name, config_path, db_dir, backend=backend)

    try:
        with open(preds_path, "a", encoding="utf-8") as fh:
            for idx, group in enumerate(groups):
                if idx in done:
                    continue
                item: LongMemEvalItem = group[0]["_item"]
                query = group[0]["query"]
                ref = group[0]["answer"]
                history = group[0]["history"]
                sessions = sessions_for_group(group)

                if hasattr(system, "reset"):
                    system.reset()

                _ingest_question(system, name, sessions)

                try:
                    pred = system.answer(query, history) or ""
                except Exception as exc:  # noqa: BLE001
                    print(f"\n    [{name}] ERROR on {item.question_id}: {exc}", flush=True)
                    traceback.print_exc()
                    pred = ""

                retrieved_ids = (
                    _retrieved_session_ids(item, system, name) if recall_enabled else None
                )
                # Note text actually answered from, so `turn_recall` measures
                # something. Previously hardcoded to [] (see `_retrieved_texts`).
                retrieved_texts: List[str] = (
                    _retrieved_texts(system, name) if recall_enabled else []
                )

                stats = (
                    _recall_stats(
                        item,
                        name,
                        retrieved_ids=retrieved_ids,
                        retrieved_texts=retrieved_texts,
                    )
                    if recall_enabled
                    else {"session_recall": None, "session_hit": None, "turn_recall": None}
                )

                rec = {
                    "idx": idx,
                    "question_id": item.question_id,
                    "question_type": item.question_type,
                    "question": item.question,
                    "expected_answer": ref,
                    "ai_response": pred,
                    "pred": pred,
                    "ref": ref,
                    "category_name": item.question_type,
                    "is_abstention": item.is_abstention,
                    "gold_session_ids": item.gold_session_ids,
                    "retrieved_session_ids": list(retrieved_ids or []),
                    "session_recall": stats["session_recall"],
                    "session_hit": stats["session_hit"],
                    "turn_recall": stats["turn_recall"],
                }
                enriched.append(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

                n_done = len(enriched)
                if n_done % 10 == 0 or n_done == len(groups):
                    print(
                        f"    [{name}] {n_done}/{len(groups)} "
                        f"({n_done / len(groups) * 100:.0f}%)",
                        flush=True,
                    )
    finally:
        close_system(system)

    enhanced = sorted(enriched, key=lambda r: r["idx"])
    preds = [rec["pred"] for rec in enhanced]
    refs = [rec["ref"] for rec in enhanced]

    scored = _score_system(
        name, preds, refs, enhanced, metric_names, judge_config, judge_enabled
    )

    judge_rows = scored.pop("judge_rows", [])
    if judge_rows:
        os.makedirs(scores_dir, exist_ok=True)
        scores_path = os.path.join(scores_dir, f"{tag}__{model_tag}__{name}.jsonl")
        with open(scores_path, "w", encoding="utf-8") as fh:
            for row in judge_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return scored


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_opt(value: Optional[float]) -> str:
    """Format a possibly-unmeasured metric: ``None`` -> ``n/a``."""
    return "n/a" if value is None else f"{value:.4f}"


def render_report(
    tag: str,
    model_tag: str,
    config_path: str,
    items: List[LongMemEvalItem],
    results: Dict[str, Any],
    metric_names: Sequence[str],
) -> str:
    wanted = canonical_metrics(metric_names)
    lines: List[str] = []
    lines.append(f"# LongMemEval evaluation — tag `{tag}`, model `{model_tag}`\n")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Questions: {len(items)}")
    lines.append(f"- Metrics: {', '.join(wanted)}")
    lines.append("")
    lines.append(
        "> Recall columns are `n/a` for systems that do not emit retrieval "
        "telemetry (only `ASEM-THG` does); `n/a` means *not measured*, not 0.\n"
    )

    # Overall table
    header = ["System", "n"] + list(wanted) + ["sess_recall", "sess_hit", "turn_recall", "abst_acc"]
    lines.append("## Overall\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for name in results:
        entry = results[name]
        overall = entry.get("overall", {})
        recall = entry.get("recall", {})
        abst = entry.get("abstention", {})
        row = [
            name,
            f"{entry.get('n', 0):.0f}",
            *[f"{overall.get(m, 0.0):.4f}" for m in wanted],
            _fmt_opt(recall.get("session_recall")),
            _fmt_opt(recall.get("session_hit")),
            _fmt_opt(recall.get("turn_recall")),
            f"{abst.get('abstention_accuracy', 0.0):.4f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Per question type
    lines.append("## Per question type\n")
    for name, entry in results.items():
        per_cat = entry.get("per_category", {})
        if not per_cat:
            continue
        lines.append(f"### {name}\n")
        head = ["question_type", "n"] + list(wanted) + ["sess_recall", "turn_recall"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        recall_by_cat = entry.get("recall_by_category", {})
        for cat, m in per_cat.items():
            rcat = recall_by_cat.get(cat, {})
            row = [
                cat,
                f"{m.get('n', 0):.0f}",
                *[f"{m.get(k, 0.0):.4f}" for k in wanted],
                _fmt_opt(rcat.get("session_recall")),
                _fmt_opt(rcat.get("turn_recall")),
            ]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # Abstention summary
    lines.append("## Abstention\n")
    lines.append("| System | abstention questions | refusal rate | abstention accuracy |")
    lines.append("|---|---|---|---|")
    for name, entry in results.items():
        abst = entry.get("abstention", {})
        lines.append(
            f"| {name} | {abst.get('n_abstention_questions', 0):.0f} | "
            f"{abst.get('refusal_rate', 0.0):.4f} | {abst.get('abstention_accuracy', 0.0):.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_items(path: str, limit: Optional[int]) -> List[LongMemEvalItem]:
    items = load_longmemeval(path)
    if limit is not None:
        items = items[:limit]
    return items


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all ASEM systems + baselines on LongMemEval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="datasets/longmemeval/longmemeval_oracle.json",
        help="LongMemEval .json (oracle / s_cleaned)",
    )
    parser.add_argument("--tag", default="oracle", help="Run tag (names the artifacts)")
    parser.add_argument(
        "--config",
        default="configs/models/deepseek_api.yaml",
        help="YAML config with the inference backend + hyperparameters",
    )
    parser.add_argument(
        "--judge-config",
        default="configs/models/judge_api.yaml",
        help="YAML config for the LLM judge (only used with the 'judge' metric)",
    )
    parser.add_argument(
        "--results",
        default="data/benchmarks/results/longmemeval",
        help="Results directory (JSON/MD + preds/ + scores/)",
    )
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/longmemeval_banks",
        help="Directory for SQLite memory banks",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=f"Systems to run. Choices: {' '.join(ALL_SYSTEMS)}. Default: all.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["em", "f1", "rougeL"],
        help="Metrics: em em_loose f1 rougeL bertscore_f1 judge",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--per-category", action="store_true", help="Include the per-question-type breakdown")
    parser.add_argument("--recall", action="store_true", help="Compute session/turn recall stats")
    parser.add_argument("--judge", action="store_true", help="Run the LLM judge")
    parser.add_argument("--clean", action="store_true", help="Delete previous preds/scores before running")
    args = parser.parse_args()

    systems = args.systems or list(DEFAULT_SYSTEMS)
    unknown = [s for s in systems if s not in ALL_SYSTEMS]
    if unknown:
        parser.error(f"unknown system(s): {unknown}. Choices: {ALL_SYSTEMS}")

    metric_names = canonical_metrics(args.metrics)
    if args.judge:
        metric_names = canonical_metrics(list(metric_names) + ["judge"])

    items = _load_items(args.input, args.limit)
    groups = group_by_session(items)
    print("=" * 64)
    print(f"LongMemEval run: tag={args.tag} questions={len(items)} systems={len(systems)}")
    print(f"  input : {args.input}")
    print(f"  config: {args.config}")
    print("=" * 64)

    model_tag = model_tag_from_config(args.config)
    out_root = os.path.join(args.results, args.tag)
    preds_dir = os.path.join(out_root, "preds")
    scores_dir = os.path.join(out_root, "scores")
    os.makedirs(preds_dir, exist_ok=True)
    os.makedirs(scores_dir, exist_ok=True)
    json_path = os.path.join(out_root, f"{args.tag}__{model_tag}.json")
    md_path = os.path.join(out_root, f"{args.tag}__{model_tag}.md")

    if args.clean:
        for name in systems:
            for d in (preds_dir, scores_dir):
                stale = os.path.join(d, f"{args.tag}__{model_tag}__{name}.jsonl")
                if os.path.exists(stale):
                    os.remove(stale)

    backend = build_backend_from_config(args.config)

    results: Dict[str, Any] = {}
    for i, name in enumerate(systems, 1):
        print(f"\n[{i}/{len(systems)}] {name}", flush=True)
        try:
            results[name] = evaluate_system(
                name=name,
                items=items,
                groups=groups,
                config_path=args.config,
                db_dir=args.db_dir,
                preds_dir=preds_dir,
                scores_dir=scores_dir,
                metric_names=metric_names,
                backend=backend,
                tag=args.tag,
                model_tag=model_tag,
                judge_config=args.judge_config,
                judge_enabled=args.judge,
                recall_enabled=args.recall,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  !! {name} failed: {exc}", flush=True)
            traceback.print_exc()
            results[name] = {"error": str(exc)}

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "meta": {
                        "tag": args.tag,
                        "model": model_tag,
                        "config": args.config,
                        "input": args.input,
                        "questions": len(items),
                        "metrics": list(metric_names),
                        "created": datetime.now().isoformat(timespec="seconds"),
                    },
                    "systems": results,
                },
                fh,
                indent=2,
            )

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_report(args.tag, model_tag, args.config, items, results, metric_names))

    print("\n" + "=" * 64)
    print(f"Results -> {json_path}")
    print(f"Report  -> {md_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
