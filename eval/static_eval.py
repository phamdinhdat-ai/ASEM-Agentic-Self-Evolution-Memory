"""Static-bank evaluation: answer full QA from frozen (ingest-once) banks.

Phase B of the static-bank workflow. It reads the frozen banks produced by
``scripts/build_static_banks.py`` and answers every QA pair of the dataset,
scoring each answer with EM / token-F1 / ROUGE-L / BERTScore-F1 / LLM-as-a-judge
while comparing the memory systems against the ``FullContext`` and ``NoMemory``
baselines on exactly the same questions.

Design goals (crash / interrupt safety)
--------------------------------------
* **Resume, never restart.** One JSON line is appended and flushed per answered
  QA pair (``preds/<tag>__<model>__<System>.jsonl``). On start-up the file is
  reloaded to learn which indices are done; re-running the identical command
  continues where it stopped.
* **Failures are data.** An exception while answering records ``pred=""`` plus
  the error string and moves on. A judge or BERTScore failure degrades to
  ``None``/``0.0`` and is recorded — it can never lose an answer. A run of
  consecutive answer failures aborts that system (not the whole run) so a dead
  endpoint does not burn the remaining budget.
* **Always a valid result file.** The aggregate JSON is rewritten atomically
  after every conversation, so a kill at any moment leaves usable metrics.

Frozen banks are never mutated: each bank is copied into a per-run working
directory (SQLite online backup) before it is opened, so systems that write
while answering (``ValueRetrievalOnly``'s q-updates) cannot contaminate the
ingest-once artifact or another model's run.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from asem.logging_utils import get_logger
from eval.metrics import (
    bertscore_f1_batch,
    canonical_metrics,
    compute_metrics,
    em_loose,
    metric_needs_bertscore,
    metric_needs_judge,
    per_category_metrics,
    rouge_l,
    token_f1,
)
from eval.phase_runner import (
    CATEGORY_NAMES,
    bank_dir,
    bank_file,
    build_eval_system,
    close_system,
    system_bank_size,
    working_copy,
    write_results,
)
from eval.systems import NO_BANK_SYSTEMS

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retry(fn, attempts: int = 3, base_delay: float = 1.0, label: str = "call"):
    """Call ``fn`` with exponential backoff. Re-raises the last error."""
    last: Optional[BaseException] = None
    for attempt in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < attempts - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "{} failed ({}/{}): {} — retrying in {:.1f}s",
                    label, attempt + 1, attempts, exc, delay,
                )
                time.sleep(delay)
    assert last is not None
    raise last


def _append_jsonl(handle: Any, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    handle.flush()


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line from a hard kill — ignore it.
                logger.warning("Ignoring malformed line in {}", path)
    return out


def _log_line(log_path: Optional[str], message: str) -> None:
    if not log_path:
        return
    try:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}\n")
    except OSError:
        pass


def build_context_baseline(
    name: str,
    backend: Any,
    full_context_dates: bool = False,
    max_full_context_turns: int = 0,
):
    """Build NoMemory / FullContext (bankless) with a configurable context budget."""
    from eval.baselines import FullContext, NoMemory
    from eval.systems import (
        _FULL_CONTEXT_PROMPT,
        _FULL_CONTEXT_PROMPT_DATED,
        _NO_MEMORY_PROMPT,
    )

    if name == "NoMemory":
        return NoMemory(backend=backend, prompt_template=_NO_MEMORY_PROMPT)
    if name == "FullContext":
        prompt = _FULL_CONTEXT_PROMPT_DATED if full_context_dates else _FULL_CONTEXT_PROMPT
        return FullContext(
            backend=backend,
            prompt_template=prompt,
            max_history_turns=max_full_context_turns,
        )
    raise ValueError(f"Not a context baseline: {name}")


# ---------------------------------------------------------------------------
# Per-system accumulating state
# ---------------------------------------------------------------------------

@dataclass
class SystemState:
    """Everything accumulated for one system across a (possibly resumed) run."""

    name: str
    order: List[int] = field(default_factory=list)
    records: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    scores: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    missing_conversations: int = 0
    n_errors: int = 0
    consecutive_errors: int = 0
    aborted: Optional[str] = None

    def is_done(self, idx: int) -> bool:
        return idx in self.records

    def add(self, idx: int, record: Dict[str, Any]) -> None:
        if idx not in self.records:
            self.order.append(idx)
        self.records[idx] = record
        if record.get("error"):
            self.n_errors += 1
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

    def metric_arrays(self) -> Tuple[List[str], List[str], List[str]]:
        preds = [str(self.records[i].get("pred", "")) for i in self.order]
        refs = [str(self.records[i].get("ref", "")) for i in self.order]
        cats = [str(self.records[i].get("category_name", "unknown")) for i in self.order]
        return preds, refs, cats


def _aggregate_state(
    state: SystemState,
    metric_names: Sequence[str],
    per_category: bool,
) -> Dict[str, Any]:
    """Compute overall + per-category metrics from a system's accumulated state."""
    preds, refs, cats = state.metric_arrays()
    loose = [float(state.records[i].get("em_loose") or 0.0) for i in state.order]

    need_bert = metric_needs_bertscore(metric_names)
    need_judge = metric_needs_judge(metric_names)

    bert: Optional[List[float]] = None
    if need_bert:
        if all("bertscore_f1" in state.scores.get(i, {}) for i in state.order):
            bert = [float(state.scores[i]["bertscore_f1"]) for i in state.order]
        # else: leave None -> compute_metrics recomputes the batch

    judge: Optional[List[Optional[bool]]] = None
    if need_judge:
        if all("judge_correct" in state.scores.get(i, {}) for i in state.order):
            judge = [state.scores[i].get("judge_correct") for i in state.order]

    entry: Dict[str, Any] = {
        "n": len(state.order),
        "missing_conversations": state.missing_conversations,
        "n_answer_errors": state.n_errors,
        "aborted": state.aborted,
        "overall": compute_metrics(
            preds, refs, metric_names,
            em_loose_flags=loose, bertscore_scores=bert, judge_flags=judge,
        ),
    }
    if per_category:
        entry["per_category"] = per_category_metrics(
            preds, refs, cats, metric_names,
            em_loose_flags=loose, bertscore_scores=bert, judge_flags=judge,
        )
    return entry


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_METRIC_HEADERS = {
    "em": "EM",
    "em_loose": "EM(loose)",
    "f1": "F1",
    "rougeL": "ROUGE-L",
    "bertscore_f1": "BERTScore-F1",
    "judge": "Judge",
}


def render_report_table(results: Dict[str, Any], metric_names: Sequence[str]) -> str:
    """Markdown table: rows = systems, columns = metrics, plus per-category tables."""
    metrics = canonical_metrics(metric_names)
    cols = [m for m in metrics if m in _METRIC_HEADERS]

    lines: List[str] = []
    lines.append("| System | n | " + " | ".join(_METRIC_HEADERS[m] for m in cols) +
                 " | Errors |")
    lines.append("|" + "---|" * (len(cols) + 3))
    for name, entry in results.get("systems", {}).items():
        overall = entry.get("overall", {})
        cells = [f"{overall.get(m, 0.0):.4f}" for m in cols]
        lines.append(
            f"| {name} | {entry.get('n', 0)} | " + " | ".join(cells) +
            f" | {entry.get('n_answer_errors', 0)} |"
        )

    if any(entry.get("per_category") for entry in results.get("systems", {}).values()):
        lines.append("")
        first = next(iter(results["systems"].values()))
        cats = sorted(first.get("per_category", {}).keys())
        for cat in cats:
            lines.append(f"\n**{cat}**")
            lines.append("| System | n | " + " | ".join(_METRIC_HEADERS[m] for m in cols) + " |")
            lines.append("|" + "---|" * (len(cols) + 2))
            for name, entry in results.get("systems", {}).items():
                bucket = (entry.get("per_category") or {}).get(cat)
                if not bucket:
                    continue
                cells = [f"{bucket.get(m, 0.0):.4f}" for m in cols]
                lines.append(f"| {name} | {bucket.get('n', 0)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_static_eval(
    groups: Sequence[Sequence[Dict[str, Any]]],
    systems: Sequence[str],
    config_path: str,
    bank_root: str,
    tag: str,
    dataset: str,
    backend: Any = None,
    metric_names: Sequence[str] = ("em", "rougeL", "f1", "bertscore_f1", "judge"),
    judge_config: Optional[str] = None,
    judge_backend: Any = None,
    judge_max_retries: int = 3,
    preds_dir: Optional[str] = None,
    scores_dir: Optional[str] = None,
    out_path: Optional[str] = None,
    log_path: Optional[str] = None,
    per_category: bool = True,
    require_banks: bool = True,
    work_root: Optional[str] = None,
    model_tag: Optional[str] = None,
    limit: Optional[int] = None,
    full_context_dates: bool = False,
    max_full_context_turns: int = 0,
    resume: bool = True,
    abort_after_consecutive_errors: int = 8,
) -> Dict[str, Any]:
    """Answer every QA pair from the frozen banks and score it.

    Args:
        groups: One list of QA items per conversation (see ``group_by_conversation``).
        systems: System names; bankless names (``FullContext``/``NoMemory``) are
            answered from ``item["history"]`` instead of a bank.
        bank_root: Directory holding ``<tag>/<System>/<conv>/<bank>.sqlite``.
        work_root: When set, each bank is copied here before opening so the
            frozen static banks are never written.
        resume: Reload previously written predictions/scores and skip done items.
        abort_after_consecutive_errors: Abort the current system after this many
            consecutive answer failures (0 disables the guard).

    Returns:
        Result dict with ``systems[name]["overall"|"per_category"|"n"]``.
    """
    from eval.phase_runner import build_backend_from_config, model_tag_from_config

    metrics = canonical_metrics(metric_names)
    need_bert = metric_needs_bertscore(metrics)
    need_judge = metric_needs_judge(metrics)

    backend = backend if backend is not None else build_backend_from_config(config_path)
    model_tag = model_tag or model_tag_from_config(config_path)

    if need_judge and judge_backend is None:
        judge_backend = (
            build_backend_from_config(judge_config) if judge_config else backend
        )

    judge = None
    if need_judge:
        from eval.llm_as_a_judge import LLMJudge

        judge = LLMJudge(backend=judge_backend, max_retries=judge_max_retries)

    selected = list(groups)[:limit] if limit else list(groups)
    states: Dict[str, SystemState] = {name: SystemState(name=name) for name in systems}

    if preds_dir:
        os.makedirs(preds_dir, exist_ok=True)
    if scores_dir:
        os.makedirs(scores_dir, exist_ok=True)

    def _preds_path(name: str) -> Optional[str]:
        return os.path.join(preds_dir, f"{tag}__{model_tag}__{name}.jsonl") if preds_dir else None

    def _scores_path(name: str) -> Optional[str]:
        return os.path.join(scores_dir, f"{tag}__{model_tag}__{name}.jsonl") if scores_dir else None

    # ---- Resume ---------------------------------------------------------
    if resume:
        for name in systems:
            state = states[name]
            preds_file = _preds_path(name)
            for rec in _read_jsonl(preds_file) if preds_file else []:
                idx = int(rec.get("idx", -1))
                if idx < 0:
                    continue
                state.add(idx, rec)
            scores_file = _scores_path(name)
            for rec in _read_jsonl(scores_file) if scores_file else []:
                idx = int(rec.get("idx", -1))
                if idx >= 0:
                    state.scores[idx] = rec
            if state.records:
                logger.info(
                    "Resuming {}: {} answered, {} scored", name, len(state.records),
                    len(state.scores),
                )

    handles: Dict[str, Any] = {}
    score_handles: Dict[str, Any] = {}
    try:
        for name in systems:
            pf = _preds_path(name)
            if pf:
                handles[name] = open(pf, "a", encoding="utf-8")
            sf = _scores_path(name)
            if sf:
                score_handles[name] = open(sf, "a", encoding="utf-8")

        t0 = time.perf_counter()
        total_pairs = sum(len(g) for g in selected)
        done_before = sum(len(states[n].records) for n in systems)

        def _snapshot(final: bool = False) -> Dict[str, Any]:
            return {
                "phase": "eval",
                "dataset": dataset,
                "tag": tag,
                "model_tag": model_tag,
                "config": config_path,
                "judge_config": judge_config or config_path,
                "metrics": metrics,
                "bank_root": bank_root,
                "work_copy": bool(work_root),
                "n_qa": total_pairs,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
                "systems": {
                    n: _aggregate_state(states[n], metrics, per_category) for n in systems
                },
            }

        # ---- Conversations ---------------------------------------------
        for group in selected:
            conversation_id = str(group[0].get("session_id", ""))
            built: Dict[str, Any] = {}

            for name in systems:
                if name in NO_BANK_SYSTEMS:
                    built[name] = build_context_baseline(
                        name, backend, full_context_dates, max_full_context_turns
                    )
                    continue

                src_file = bank_file(bank_root, tag, name, conversation_id)
                if not os.path.exists(src_file):
                    states[name].missing_conversations += 1
                    msg = (
                        f"No bank for system={name} conversation={conversation_id} "
                        f"(expected {src_file})"
                    )
                    if require_banks:
                        raise FileNotFoundError(msg + " — run build_static_banks.py first")
                    logger.warning(msg + " — skipping")
                    _log_line(log_path, f"[{name}] SKIP {conversation_id}: missing bank")
                    continue

                bdir = bank_dir(bank_root, tag, name, conversation_id)
                if work_root:
                    bdir = working_copy(src_file, work_root, tag, model_tag, name, conversation_id)
                built[name] = build_eval_system(name, config_path, bdir, backend)
                logger.info(
                    "[{}] bank loaded | conv={} | {} notes",
                    name, conversation_id, system_bank_size(built[name]),
                )

            # ---- Answer the QA pairs -------------------------------------
            for item in group:
                idx = int(item.get("_idx", -1))
                if idx < 0:
                    continue
                query = str(item.get("query", ""))
                ref = str(item.get("answer", ""))
                history = [str(h) for h in item.get("history", [])]
                raw_question = str(item.get("raw_question") or query)
                category_name = item.get("category_name") or CATEGORY_NAMES.get(
                    int(item.get("category", 0)), "unknown"
                )

                for name, system in built.items():
                    state = states[name]
                    if state.aborted or state.is_done(idx):
                        continue

                    hist = history if name in NO_BANK_SYSTEMS else []
                    error: Optional[str] = None
                    try:
                        pred = str(system.answer(query, hist) or "")
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        logger.opt(exception=exc).error(
                            "[{}] answer failed | idx={}", name, idx
                        )
                        pred = ""
                        error = f"{type(exc).__name__}: {exc}"

                    # Cheap per-item metrics are stored inline so a resumed run
                    # never has to re-derive them.
                    record = {
                        "idx": idx,
                        "conversation_id": conversation_id,
                        "category": item.get("category", 0),
                        "category_name": category_name,
                        "question": raw_question,
                        "query": query,
                        "pred": pred,
                        "ref": ref,
                        "error": error,
                        "em": 1.0 if _norm_eq(pred, ref) else 0.0,
                        "em_loose": em_loose(pred, ref),
                        "f1": token_f1(pred, ref),
                        "rougeL": rouge_l(pred, ref),
                    }
                    state.add(idx, record)
                    if handles.get(name):
                        _append_jsonl(handles[name], record)

                    if (
                        error
                        and abort_after_consecutive_errors
                        and state.consecutive_errors >= abort_after_consecutive_errors
                    ):
                        state.aborted = (
                            f"{state.consecutive_errors} consecutive answer errors "
                            f"(>= {abort_after_consecutive_errors}); last: {error}"
                        )
                        logger.error("[{}] aborting system: {}", name, state.aborted)
                        _log_line(log_path, f"[{name}] ABORTED {state.aborted}")

            # ---- Score the conversation (BERTScore + judge, batched) -----
            for name, system in built.items():
                state = states[name]
                pending = [
                    int(item.get("_idx", -1))
                    for item in group
                    if int(item.get("_idx", -1)) in state.records
                    and int(item.get("_idx", -1)) not in state.scores
                ]
                if not pending:
                    continue
                pending = [i for i in pending if i in state.records]

                if need_bert:
                    preds = [state.records[i]["pred"] for i in pending]
                    refs = [state.records[i]["ref"] for i in pending]
                    bs = bertscore_f1_batch(preds, refs)
                else:
                    bs = [None] * len(pending)

                for i, b in zip(pending, bs):
                    rec: Dict[str, Any] = {"idx": i, "conversation_id": conversation_id}
                    if need_bert:
                        rec["bertscore_f1"] = float(b or 0.0)
                    if need_judge and judge is not None:
                        item = state.records[i]
                        try:
                            verdict = _retry(
                                lambda item=item: judge.judge(
                                    question=item.get("question", ""),
                                    expected_answer=item.get("ref", ""),
                                    ai_response=item.get("pred", ""),
                                    conversation_id=item.get("conversation_id", ""),
                                    question_type=item.get("category_name", ""),
                                    category=int(item.get("category") or 0),
                                ),
                                attempts=judge_max_retries,
                                label=f"judge[{name}]",
                            )
                            rec["judge_correct"] = bool(verdict.is_correct)
                            rec["judge_reasoning"] = verdict.reasoning
                            rec["judge_error"] = verdict.error
                        except Exception as exc:  # noqa: BLE001
                            logger.opt(exception=exc).error(
                                "[{}] judge failed permanently | idx={}", name, i
                            )
                            rec["judge_correct"] = None
                            rec["judge_reasoning"] = ""
                            rec["judge_error"] = f"{type(exc).__name__}: {exc}"
                    state.scores[i] = rec
                    if score_handles.get(name):
                        _append_jsonl(score_handles[name], rec)

            for system in built.values():
                close_system(system)

            # ---- Crash-safe checkpoint after every conversation ----------
            if out_path:
                write_results(out_path, _snapshot())
            _log_line(
                log_path,
                f"[{conversation_id}] " + " | ".join(
                    f"{n}: n={len(states[n].records)} err={states[n].n_errors}"
                    for n in systems
                ),
            )
            logger.info(
                "Conversation {} done | answered={} | {:.1f}s",
                conversation_id,
                {n: len(states[n].records) for n in systems},
                time.perf_counter() - t0,
            )

        # ---- Final backfill (safety net for a resumed/interrupted run) ----
        for name in systems:
            state = states[name]
            if not state.order:
                continue
            if need_bert:
                missing = [i for i in state.order if "bertscore_f1" not in state.scores.get(i, {})]
                if missing:
                    logger.info("{}: backfilling BERTScore for {} items", name, len(missing))
                    bs = bertscore_f1_batch(
                        [state.records[i]["pred"] for i in missing],
                        [state.records[i]["ref"] for i in missing],
                    )
                    for i, b in zip(missing, bs):
                        state.scores.setdefault(i, {"idx": i})["bertscore_f1"] = float(b or 0.0)
            if need_judge and judge is not None:
                missing = [i for i in state.order if "judge_correct" not in state.scores.get(i, {})]
                if missing:
                    logger.info("{}: backfilling judge for {} items", name, len(missing))
                    for i in missing:
                        item = state.records[i]
                        try:
                            verdict = _retry(
                                lambda item=item: judge.judge(
                                    question=item.get("question", ""),
                                    expected_answer=item.get("ref", ""),
                                    ai_response=item.get("pred", ""),
                                    conversation_id=item.get("conversation_id", ""),
                                    question_type=item.get("category_name", ""),
                                    category=int(item.get("category") or 0),
                                ),
                                attempts=judge_max_retries,
                                label=f"judge[{name}]",
                            )
                            state.scores.setdefault(i, {"idx": i}).update(
                                {"judge_correct": bool(verdict.is_correct),
                                 "judge_reasoning": verdict.reasoning,
                                 "judge_error": verdict.error}
                            )
                        except Exception as exc:  # noqa: BLE001
                            state.scores.setdefault(i, {"idx": i}).update(
                                {"judge_correct": None, "judge_reasoning": "",
                                 "judge_error": f"{type(exc).__name__}: {exc}"}
                            )

        results = _snapshot(final=True)
        results["resumed_from"] = done_before
        if out_path:
            write_results(out_path, results)
        return results

    finally:
        for fh in list(handles.values()) + list(score_handles.values()):
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass


def _norm_eq(pred: Any, ref: Any) -> bool:
    from eval.metrics import exact_match

    return exact_match(pred, ref) == 1.0


__all__ = [
    "SystemState",
    "build_context_baseline",
    "render_report_table",
    "run_static_eval",
]
