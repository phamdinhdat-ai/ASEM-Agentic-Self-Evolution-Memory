"""
Run ASEM experiments on the locomo10.json sample dataset.

Converts raw LoCoMo conversations + QA pairs into sequential-memory evaluation
format, then runs all 7 systems (NoMemory, FullContext, SimRetrieval,
AtomicLinking, RLManagerOnly, ValueRetrievalOnly, ASEM) with conversation-aware
reset, incremental save/resume, and per-category breakdown.

Usage
-----
    # Quick smoke test (10 examples, fast)
    python scripts/run_locomo10_experiments.py --limit 10

    # Full run on all ~1990 QA pairs
    python scripts/run_locomo10_experiments.py

    # Specific systems only, with BERTScore
    python scripts/run_locomo10_experiments.py \\
        --systems NoMemory FullContext ASEM \\
        --metrics em rougeL bertscore_f1

    # With per-category breakdown
    python scripts/run_locomo10_experiments.py --per-category

    # Using a different backend config
    python scripts/run_locomo10_experiments.py \\
        --config configs/langchain_ollama.yaml

    # Ablation: sweep hyperparameters
    python scripts/run_locomo10_experiments.py --ablate lambda \\
        --lambda-values 0.0 0.2 0.4 0.6 0.8 1.0 \\
        --systems ASEM --limit 100

    # Ablation: disable components
    python scripts/run_locomo10_experiments.py --ablate components \\
        --disable-link-evolver --disable-zscore \\
        --systems ASEM --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple


os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "sk-datpd5"
# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env
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
                    _v = _v.strip().strip('"').strip("'")
                    os.environ.setdefault(_k.strip(), _v)

# Configure logging early
from asem.logging_utils import setup_logging  # noqa: E402
setup_logging(level=os.environ.get("LOG_LEVEL", "INFO"))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "commonsense",
    4: "conversational",
    5: "adversarial",
}

ALL_SYSTEMS = [
    "NoMemory", "FullContext", "SimRetrieval",
    "AtomicLinking", "RLManagerOnly", "ValueRetrievalOnly", "ASEM",
]


# ---------------------------------------------------------------------------
# Data conversion: locomo10.json → eval format
# ---------------------------------------------------------------------------

def _parse_dia_id(dia_id: str) -> Tuple[int, int]:
    """Parse 'D3:7' → (session=3, turn=7)."""
    m = re.match(r"D(\d+):(\d+)", dia_id)
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1


def _build_turn_index(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build a flat dia_id → turn dict from all sessions in a conversation."""
    index: Dict[str, Dict[str, Any]] = {}
    for key, value in conversation.items():
        if not key.startswith("session_") or not isinstance(value, list):
            continue
        for turn in value:
            dia_id = turn.get("dia_id")
            if dia_id:
                index[dia_id] = turn
    return index


def _turn_to_text(turn: Dict[str, Any]) -> str:
    """Format a dialogue turn as a readable string."""
    speaker = turn.get("speaker", "Unknown")
    text = turn.get("text", "")
    blip = turn.get("blip_caption", "")
    content = f"[{speaker}] {text}"
    if blip:
        content += f" (image: {blip})"
    return content


def _build_history_for_qa(
    qa: Dict[str, Any],
    turn_index: Dict[str, Dict[str, Any]],
    conversation: Dict[str, Any],
) -> List[str]:
    """
    Build sequential history for a QA pair.

    Collects ALL dialogue turns from session 1 up to the latest session
    referenced in evidence, with session date markers, sorted chronologically.
    Evidence turns are INCLUDED — the task is retrieval, not clairvoyance.
    """
    evidence_keys: Set[str] = set()
    raw_evidence = qa.get("evidence", [])
    for eid in raw_evidence:
        for part in re.split(r"[;,]", str(eid)):
            part = part.strip()
            if part:
                evidence_keys.add(part)

    if not evidence_keys:
        return []

    # Find the latest session referenced in evidence
    max_session = 0
    for eid in evidence_keys:
        sess, _ = _parse_dia_id(eid)
        if sess > max_session:
            max_session = sess

    # Collect all turns grouped by session, with date markers
    session_turns: Dict[int, List[Tuple[int, str]]] = {}
    for dia_id, turn in turn_index.items():
        sess, turn_num = _parse_dia_id(dia_id)
        if sess < 1 or sess > max_session:
            continue
        if sess not in session_turns:
            session_turns[sess] = []
        session_turns[sess].append((turn_num, _turn_to_text(turn)))

    # Build history with session date headers
    history: List[str] = []
    for sess in sorted(session_turns.keys()):
        date_key = f"session_{sess}_date_time"
        date_str = conversation.get(date_key, "")
        header = f"[Session {sess}"
        if date_str:
            header += f" — {date_str}"
        header += "]"
        history.append(header)
        for _, text in sorted(session_turns[sess]):
            history.append(text)

    return history


def convert_locomo10_to_eval(
    dataset_path: str,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convert locomo10.json to the eval format expected by the evaluation harness.

    Each output item has:
        query, answer, history, category, category_name, session_id, evidence
    """
    print(f"Loading {dataset_path} ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    eval_items: List[Dict[str, Any]] = []
    skipped = 0

    for idx, record in enumerate(dataset):
        conversation = record.get("conversation", {})
        qa_list = record.get("qa", [])
        session_id = f"locomo_{idx:04d}"
        speaker_a = conversation.get("speaker_a", "Speaker A")
        speaker_b = conversation.get("speaker_b", "Speaker B")

        turn_index = _build_turn_index(conversation)

        for qa in qa_list:
            question = str(qa.get("question", "")).strip()
            category = qa.get("category", 1)

            # Determine gold answer
            if category == 5:
                gold_answer = str(qa.get("adversarial_answer", "")).strip()
            else:
                gold_answer = str(qa.get("answer", "")).strip()

            if not question or not gold_answer:
                skipped += 1
                continue

            evidence = []
            raw_evidence = qa.get("evidence", [])
            for eid in raw_evidence:
                for part in re.split(r"[;,]", str(eid)):
                    part = part.strip()
                    if part:
                        evidence.append(part)

            # Build sequential history up to the evidence session
            history = _build_history_for_qa(qa, turn_index, conversation)

            # Enrich query with speaker context
            enriched_query = (
                f"Conversation between {speaker_a} and {speaker_b}. "
                f"Question: {question}"
            )

            eval_items.append({
                "query": enriched_query,
                "answer": gold_answer,
                "history": history,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, f"cat{category}"),
                "session_id": session_id,
                "evidence": evidence,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            })

    print(f"  {len(eval_items)} QA pairs converted  (skipped {skipped} empty)")

    if limit is not None and limit < len(eval_items):
        eval_items = eval_items[:limit]
        print(f"  Limited to first {limit} examples (--limit)")

    # Annotate with indices for O(1) lookup during evaluation
    for i, item in enumerate(eval_items):
        item["_idx"] = i

    return eval_items


def group_by_conversation(
    eval_items: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
    """Group eval items by session_id while preserving order."""
    groups: List[List[Dict[str, Any]]] = []
    current_group: List[Dict[str, Any]] = []
    current_sid = None

    for item in eval_items:
        sid = item.get("session_id", "")
        if sid != current_sid:
            if current_group:
                groups.append(current_group)
            current_group = [item]
            current_sid = sid
        else:
            current_group.append(item)

    if current_group:
        groups.append(current_group)

    return groups


def split_by_category(
    examples: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Split examples by category name."""
    by_cat: Dict[str, List] = defaultdict(list)
    for ex in examples:
        key = ex.get("category_name") or f"cat{ex.get('category', 0)}"
        by_cat[key].append(ex)
    return dict(by_cat)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match(preds: List[str], refs: List[str]) -> float:
    matches = [
        1.0 if _normalize(p) == _normalize(r) else 0.0
        for p, r in zip(preds, refs)
    ]
    if not matches:
        return 0.0
    return sum(matches) / len(matches)


def compute_metrics(
    preds: List[str],
    refs: List[str],
    metric_names: List[str],
) -> Dict[str, float]:
    """Compute requested metrics. Uses HuggingFace evaluate for ROUGE/BERTScore."""
    results: Dict[str, float] = {}

    if "em" in metric_names:
        results["em"] = exact_match(preds, refs)

    if "rougeL" in metric_names:
        import evaluate as hf_evaluate
        rouge = hf_evaluate.load("rouge")
        scores = rouge.compute(predictions=preds, references=refs)
        results["rougeL"] = float(scores.get("rougeL", 0.0))

    if "bertscore_f1" in metric_names:
        import evaluate as hf_evaluate
        bert = hf_evaluate.load("bertscore")
        scores = bert.compute(
            predictions=preds, references=refs, lang="en"
        )
        results["bertscore_f1"] = float(sum(scores["f1"]) / len(scores["f1"]))

    return results


# ---------------------------------------------------------------------------
# System factory
# ---------------------------------------------------------------------------

def build_runners_from_systems_module(
    config_path: str,
    db_dir: str,
    systems: Optional[List[str]] = None,
    max_history_turns: int = 0,
) -> Dict[str, object]:
    """Build system runners using the eval/systems.py builder.

    Each system gets its own isolated MemoryBank and properly handles
    incremental history with deduplication via the fixed baselines module.
    """
    from eval.systems import build_baselines, build_asem_system

    os.makedirs(db_dir, exist_ok=True)

    runners: Dict[str, object] = {}

    # Build baselines (each gets its own bank)
    baseline_runners = build_baselines(
        config_path, db_dir, max_history_turns=max_history_turns,
    )

    # Build ASEM
    asem_runner = build_asem_system(config_path, db_dir)

    all_available = dict(baseline_runners)
    all_available["ASEM"] = asem_runner

    for name in (systems or ALL_SYSTEMS):
        if name in all_available:
            runners[name] = all_available[name]
        else:
            print(f"  WARNING: Unknown system '{name}' — skipping")

    return runners


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_system(
    runner: object,
    eval_items: List[Dict[str, Any]],
    conversation_groups: List[List[Dict[str, Any]]],
    metric_names: List[str],
    preds_dir: str,
    sys_name: str,
    results: Dict[str, Any],
    results_path: str,
    flush_fn: callable,
) -> Tuple[List[str], List[str]]:
    """Evaluate a single system on all eval items, conversation by conversation.

    Returns (predictions, references).
    """
    preds_path = os.path.join(preds_dir, f"locomo10_{sys_name}.jsonl")
    done_ids: Set[int] = set()
    preds_so_far: List[str] = []
    refs_so_far: List[str] = []

    # Resume from partial predictions
    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done_ids.add(rec["idx"])
                    preds_so_far.append(rec["pred"])
                    refs_so_far.append(rec["ref"])
        if done_ids:
            print(f"    Resuming from {len(done_ids)} saved predictions")
            partial_key = f"locomo10/{sys_name}/__partial__"
            partial_metrics = compute_metrics(preds_so_far, refs_so_far, metric_names)
            partial_metrics["__n__"] = len(done_ids)
            partial_metrics["__total__"] = len(eval_items)
            results[partial_key] = partial_metrics
            flush_fn(results)

    # Evaluate conversation by conversation
    with open(preds_path, "a", encoding="utf-8") as fp:
        for group in conversation_groups:
            # Reset runner at each conversation boundary
            if hasattr(runner, 'reset'):
                runner.reset()

            for item in group:
                idx = item.get("_idx", -1)
                if idx in done_ids:
                    continue

                query = str(item.get("query", ""))
                ref = str(item.get("answer", ""))
                history = [str(h) for h in item.get("history", [])]

                try:
                    pred = runner.answer(query, history)
                except Exception as exc:
                    print(f"\n    ERROR on example {idx} (session {item.get('session_id')}): {exc}")
                    traceback.print_exc()
                    pred = ""

                preds_so_far.append(pred)
                refs_so_far.append(ref)

                fp.write(json.dumps({
                    "idx": idx,
                    "session_id": item.get("session_id", ""),
                    "category": item.get("category", 0),
                    "category_name": item.get("category_name", ""),
                    "query": query,
                    "pred": pred,
                    "ref": ref,
                }) + "\n")
                fp.flush()

                n_done = len(done_ids) + len(preds_so_far)
                if n_done % 5 == 0 or n_done == len(eval_items):
                    pct = n_done / len(eval_items) * 100
                    print(f"    [{sys_name}] {n_done}/{len(eval_items)} "
                          f"({pct:.0f}%)  latest pred: {pred[:80]!r}", flush=True)

                # Save partial metrics every 25 examples
                if n_done % 25 == 0 or n_done == len(eval_items):
                    partial_key = f"locomo10/{sys_name}/__partial__"
                    partial_metrics = compute_metrics(
                        preds_so_far, refs_so_far, metric_names,
                    )
                    partial_metrics["__n__"] = n_done
                    partial_metrics["__total__"] = len(eval_items)
                    results[partial_key] = partial_metrics
                    flush_fn(results)

    return preds_so_far, refs_so_far


# ---------------------------------------------------------------------------
# Ablation support
# ---------------------------------------------------------------------------

def run_lambda_ablation(
    config_path: str,
    db_base_dir: str,
    eval_items: List[Dict[str, Any]],
    conversation_groups: List[List[Dict[str, Any]]],
    metric_names: List[str],
    lambda_values: List[float],
    results: Dict[str, Any],
    results_path: str,
    preds_dir: str,
):
    """Sweep lambda values for ASEM only."""
    import yaml
    from eval.systems import build_asem_system

    def _flush(res):
        tmp = results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        os.replace(tmp, results_path)

    for lam in lambda_values:
        sys_name = f"ASEM_lambda={lam:.2f}"
        key = f"locomo10/{sys_name}"

        if key in results:
            print(f"\n  [{sys_name}] already completed — skipping")
            continue

        print(f"\n  [{sys_name}] lambda={lam:.2f} ...")
        db_dir = os.path.join(db_base_dir, f"lambda_{lam:.2f}")
        os.makedirs(db_dir, exist_ok=True)

        # Patch config to use this lambda
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cfg["hyperparameters"]["lambda"] = lam
        tmp_config = os.path.join(db_dir, "_config.yaml")
        with open(tmp_config, "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh)

        runner = build_asem_system(tmp_config, db_dir)
        preds, refs = evaluate_system(
            runner, eval_items, conversation_groups, metric_names,
            preds_dir, sys_name, results, results_path, _flush,
        )
        results[key] = compute_metrics(preds, refs, metric_names)
        results.pop(f"locomo10/{sys_name}/__partial__", None)
        _flush(results)
        print(f"    [{sys_name}] FINAL: {results[key]}")


def run_component_ablation(
    config_path: str,
    db_base_dir: str,
    eval_items: List[Dict[str, Any]],
    conversation_groups: List[List[Dict[str, Any]]],
    metric_names: List[str],
    disable_link_evolver: bool,
    disable_zscore: bool,
    results: Dict[str, Any],
    results_path: str,
    preds_dir: str,
):
    """Run ASEM with individual components disabled."""
    import yaml
    from asem.backends import build_backend
    from asem.answer_agent import AnswerAgent
    from asem.link_evolver import LinkEvolver
    from asem.memory_bank import MemoryBank
    from asem.memory_manager import MemoryManager
    from asem.note import NoteConstructor
    from asem.pipeline import ASEMPipeline
    from asem.retriever import HybridRetriever
    from asem.utility_updater import UtilityUpdater
    from eval.systems import ASEMSystem, _load_text, _make_bank

    def _flush(res):
        tmp = results_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        os.replace(tmp, results_path)

    variants = []
    if disable_link_evolver:
        variants.append(("ASEM_noLinks", "link_evolver disabled"))
    if disable_zscore:
        variants.append(("ASEM_noZScore", "z-score normalization disabled"))
    if disable_link_evolver and disable_zscore:
        variants.append(("ASEM_noLinks_noZScore", "links + zscore disabled"))

    for sys_name, desc in variants:
        key = f"locomo10/{sys_name}"
        if key in results:
            print(f"\n  [{sys_name}] already completed — skipping")
            continue

        print(f"\n  [{sys_name}] {desc} ...")
        db_dir = os.path.join(db_base_dir, sys_name.lower())
        os.makedirs(db_dir, exist_ok=True)

        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        backend = build_backend(cfg["inference"])
        hp = cfg["hyperparameters"]

        note_prompt = _load_text("data/prompts/P1_note_construction.txt")
        link_prompt = _load_text("data/prompts/P2_link_generation.txt")
        evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

        nc = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
        mm = MemoryManager(backend=backend, prompt_template=(
            "Decide memory write operation. Output JSON:\n"
            '{{"op": "ADD|UPDATE|DELETE|NOOP", "target_id": "<note_id or null>"}}\n'
            "Rules: ADD if new info. UPDATE if similar note exists. "
            "DELETE if contradicted. NOOP if irrelevant.\n"
            "Content: {content}\n"
            "Existing notes: {memory}"
        ))
        le = LinkEvolver(
            backend=backend,
            link_prompt_template=link_prompt,
            evolve_prompt_template=evolve_prompt,
            k=hp["k"],
        )
        ret = HybridRetriever(
            backend=backend,
            k1=hp["k1"], k2=hp["k2"],
            delta=hp["delta"], lambda_weight=hp["lambda"],
            use_zscore=not disable_zscore,
        )
        aa = AnswerAgent(
            backend=backend,
            prompt_template=(
                "Select the memory notes needed to answer and provide the answer. "
                "Output JSON:\n"
                '{{"selected_ids": ["id1", ...], "answer": "concise answer"}}\n'
                "Query: {query}\n"
                "Memory notes: {candidates}"
            ),
            baseline_prompt_template=(
                "Answer using the memory notes below. Reply with ONLY the answer.\n"
                "Memory:\n{context}\n"
                "Question: {query}\n"
                "Answer:"
            ),
        )
        uu = UtilityUpdater(
            backend=backend, alpha=hp["alpha"], q0=hp["q0"],
            summary_prompt_template=(
                "Summarize this interaction as a memory note. "
                "Output 1-2 factual sentences capturing what was learned.\n"
                "Query: {query}\n"
                "Answer: {answer}\n"
                "Reward: {reward}"
            ),
            note_constructor=nc,
        )

        pipeline = ASEMPipeline(
            memory_bank=_make_bank(db_dir, sys_name.lower()),
            note_constructor=nc,
            memory_manager=mm,
            link_evolver=le,
            retriever=ret,
            answer_agent=aa,
            utility_updater=uu,
        )

        # If link evolver is disabled, patch it out
        if disable_link_evolver:
            pipeline.link_evolver = _NoOpLinkEvolver()

        runner = ASEMSystem(pipeline=pipeline)
        preds, refs = evaluate_system(
            runner, eval_items, conversation_groups, metric_names,
            preds_dir, sys_name, results, results_path, _flush,
        )
        results[key] = compute_metrics(preds, refs, metric_names)
        results.pop(f"locomo10/{sys_name}/__partial__", None)
        _flush(results)
        print(f"    [{sys_name}] FINAL: {results[key]}")


class _NoOpLinkEvolver:
    """Drop-in replacement that skips all linking."""
    def link_and_evolve(self, note, bank):
        pass


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_summary_table(
    results: Dict[str, Any],
    metric_names: List[str],
    system_names: List[str],
    by_cat: Optional[Dict[str, List]] = None,
    per_category: bool = False,
):
    """Print a formatted summary table of results."""
    col_w = 16
    header = f"{'System':<30}" + "".join(f"{m:>{col_w}}" for m in metric_names)
    print(header)
    print("-" * len(header))

    for sys_name in system_names:
        key = f"locomo10/{sys_name}"
        metrics = results.get(key, {})
        row = f"{sys_name:<30}" + "".join(
            f"{metrics.get(m, 0.0):>{col_w}.4f}" for m in metric_names
        )
        print(row)

    # Also print any ablation keys
    ablation_keys = sorted(
        [k for k in results if k.startswith("locomo10/ASEM_")],
    )
    if ablation_keys:
        print()
        print("Ablation results:")
        print(header)
        print("-" * len(header))
        for key in ablation_keys:
            name = key.split("/", 1)[1]
            metrics = results.get(key, {})
            row = f"{name:<30}" + "".join(
                f"{metrics.get(m, 0.0):>{col_w}.4f}" for m in metric_names
            )
            print(row)

    if per_category and by_cat:
        print(f"\n{'='*60}")
        print("PER-CATEGORY BREAKDOWN")
        print(f"{'='*60}")
        for cat_name in sorted(by_cat.keys()):
            print(f"\n  [{cat_name}]")
            cat_header = f"  {'System':<28}" + "".join(
                f"{m:>{col_w}}" for m in metric_names
            )
            print(cat_header)
            print("  " + "-" * (len(cat_header) - 2))
            for sys_name in system_names:
                cat_key = f"locomo10_cat_{cat_name}/{sys_name}"
                if cat_key in results:
                    m = results[cat_key]
                    row = f"  {sys_name:<28}" + "".join(
                        f"{m.get(met, 0.0):>{col_w}.4f}" for met in metric_names
                    )
                    print(row)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASEM experiments on locomo10.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python scripts/run_locomo10_experiments.py --limit 10

  # Full run with all systems
  python scripts/run_locomo10_experiments.py --per-category --metrics em rougeL bertscore_f1

  # Lambda sweep
  python scripts/run_locomo10_experiments.py --ablate lambda --lambda-values 0.0 0.2 0.4 0.6 0.8 1.0 --systems ASEM --limit 200

  # Component ablation
  python scripts/run_locomo10_experiments.py --ablate components --disable-link-evolver --disable-zscore --systems ASEM --limit 200
""",
    )
    parser.add_argument(
        "--input",
        default="datasets/locomo/locomo10.json",
        help="Path to locomo10.json",
    )
    parser.add_argument(
        "--config",
        default="configs/locomo_openai.yaml",
        help="YAML config for inference backend + hyperparameters",
    )
    parser.add_argument(
        "--results",
        default="data/benchmarks/results/locomo10_experiments.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--db-dir",
        default="data/benchmarks/eval_banks_locomo10",
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
        default=["em", "rougeL"],
        help="Metrics: em rougeL bertscore_f1 (default: em rougeL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N QA pairs (smoke-test mode)",
    )
    parser.add_argument(
        "--per-category",
        action="store_true",
        help="Also report metrics broken down by QA category",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=0,
        help="Truncate history for FullContext to this many turns (0=no truncation). "
             "Recommended: 150 for LoCoMo full runs.",
    )
    parser.add_argument(
        "--ablate",
        choices=["lambda", "components"],
        default=None,
        help="Run ablation study instead of standard evaluation",
    )
    parser.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Lambda values to sweep (for --ablate lambda)",
    )
    parser.add_argument(
        "--disable-link-evolver",
        action="store_true",
        help="Disable link evolver (for --ablate components)",
    )
    parser.add_argument(
        "--disable-zscore",
        action="store_true",
        help="Disable z-score normalization (for --ablate components)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Force-clean all previous results, predictions, and databases before running.",
    )
    args = parser.parse_args()

    # Load config early and apply logging settings from it
    import yaml as _yaml  # noqa: E402
    from asem.logging_utils import setup_logging_from_config  # noqa: E402
    try:
        with open(args.config, "r", encoding="utf-8") as _fh:
            _cfg = _yaml.safe_load(_fh)
        setup_logging_from_config(_cfg)
    except Exception:
        pass  # fall back to LOG_LEVEL env var / defaults

    # Use timestamped db-dir to avoid stale SQLite file locks
    db_dir = os.path.join(args.db_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(db_dir, exist_ok=True)
    print(f"DB dir: {db_dir}")

    # ------------------------------------------------------------------
    # Step 1: Convert locomo10.json → eval format
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Convert locomo10.json to eval format")
    print("=" * 60)
    eval_data = convert_locomo10_to_eval(args.input, limit=args.limit)

    # Group by conversation (needed for conversation-aware reset)
    conversation_groups = group_by_conversation(eval_data)
    print(f"  {len(conversation_groups)} conversations")
    print(f"  avg {len(eval_data) / len(conversation_groups):.1f} QA pairs per conversation")

    # Category distribution
    by_cat = split_by_category(eval_data)
    print("\nCategory distribution:")
    for cat_name, items in sorted(by_cat.items()):
        print(f"  {cat_name}: {len(items)}")

    # ------------------------------------------------------------------
    # Step 2: Build system runners
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 2: Build system runners  (config: {args.config})")
    print(f"{'='*60}")

    if args.ablate:
        # Ablation mode — build only the systems we need
        runners = {}
    else:
        runners = build_runners_from_systems_module(
            config_path=args.config,
            db_dir=db_dir,
            systems=args.systems,
            max_history_turns=args.max_history_turns,
        )
    print(f"  Systems: {list(runners.keys()) if runners else '(ablation mode)'}")

    # ------------------------------------------------------------------
    # Step 3: Run evaluation
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 3: Run evaluation  (metrics: {args.metrics})")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    preds_dir = os.path.join(os.path.dirname(args.results), "preds")
    os.makedirs(preds_dir, exist_ok=True)

    # Clean mode
    if args.clean:
        import shutil
        for path in [args.results, preds_dir, db_dir]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        os.makedirs(preds_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)
        print("  Cleaned all previous results, predictions, and databases.")

    # Load previous results for incremental resume
    results: Dict[str, Any] = {}
    if os.path.exists(args.results):
        with open(args.results, "r", encoding="utf-8") as fh:
            try:
                results = json.load(fh)
            except json.JSONDecodeError:
                results = {}

    def _flush_results(res: Dict[str, Any]) -> None:
        tmp = args.results + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        os.replace(tmp, args.results)

    # --- Ablation path ---
    if args.ablate == "lambda":
        run_lambda_ablation(
            config_path=args.config,
            db_base_dir=db_dir,
            eval_items=eval_data,
            conversation_groups=conversation_groups,
            metric_names=args.metrics,
            lambda_values=args.lambda_values,
            results=results,
            results_path=args.results,
            preds_dir=preds_dir,
        )

    elif args.ablate == "components":
        run_component_ablation(
            config_path=args.config,
            db_base_dir=db_dir,
            eval_items=eval_data,
            conversation_groups=conversation_groups,
            metric_names=args.metrics,
            disable_link_evolver=args.disable_link_evolver,
            disable_zscore=args.disable_zscore,
            results=results,
            results_path=args.results,
            preds_dir=preds_dir,
        )

    else:
        # --- Standard evaluation path ---
        for sys_name, runner in runners.items():
            key = f"locomo10/{sys_name}"

            if key in results:
                print(f"\n  [{sys_name}] already completed — skipping")
                continue

            print(f"\n  [{sys_name}] running on {len(eval_data)} examples ...")

            preds, refs = evaluate_system(
                runner=runner,
                eval_items=eval_data,
                conversation_groups=conversation_groups,
                metric_names=args.metrics,
                preds_dir=preds_dir,
                sys_name=sys_name,
                results=results,
                results_path=args.results,
                flush_fn=_flush_results,
            )

            # Final metrics
            final_metrics = compute_metrics(preds, refs, args.metrics)
            results[key] = final_metrics
            results.pop(f"locomo10/{sys_name}/__partial__", None)
            _flush_results(results)
            print(f"    [{sys_name}] FINAL: {final_metrics}")

    # ------------------------------------------------------------------
    # Step 4: Per-category breakdown (optional)
    # ------------------------------------------------------------------
    if args.per_category and not args.ablate:
        print(f"\n{'='*60}")
        print("STEP 4: Per-category breakdown")
        print(f"{'='*60}")

        preds_dir_cat = os.path.join(preds_dir, "by_category")
        os.makedirs(preds_dir_cat, exist_ok=True)

        for cat_name, cat_examples in sorted(by_cat.items()):
            cat_groups = group_by_conversation(cat_examples)
            cat_refs = [ex["answer"] for ex in cat_examples]

            for sys_name, runner in runners.items():
                cat_key = f"locomo10_cat_{cat_name}/{sys_name}"
                if cat_key in results:
                    print(f"  [{cat_name}/{sys_name}] cached — skipping")
                    continue

                # Use a fresh runner for per-category evaluation
                fresh_runners = build_runners_from_systems_module(
                    config_path=args.config,
                    db_dir=os.path.join(db_dir, f"cat_{cat_name}"),
                    systems=[sys_name],
                    max_history_turns=args.max_history_turns,
                )
                if sys_name not in fresh_runners:
                    continue
                cat_runner = fresh_runners[sys_name]

                cat_preds, _ = evaluate_system(
                    runner=cat_runner,
                    eval_items=cat_examples,
                    conversation_groups=cat_groups,
                    metric_names=args.metrics,
                    preds_dir=preds_dir_cat,
                    sys_name=f"{cat_name}_{sys_name}",
                    results=results,
                    results_path=args.results,
                    flush_fn=_flush_results,
                )

                metrics = compute_metrics(cat_preds, cat_refs, args.metrics)
                results[cat_key] = metrics
                print(f"  [{cat_name}/{sys_name}]: {metrics}")

        _flush_results(results)

    # ------------------------------------------------------------------
    # Step 5: Print summary table
    # ------------------------------------------------------------------
    system_names = list(runners.keys()) if runners else []
    if args.ablate:
        # Collect all system-like keys from results
        system_names = sorted(set(
            k.split("/", 1)[1]
            for k in results
            if k.startswith("locomo10/") and "__partial__" not in k and "cat_" not in k
        ))

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    print_summary_table(
        results=results,
        metric_names=args.metrics,
        system_names=system_names,
        by_cat=by_cat,
        per_category=args.per_category and not args.ablate,
    )

    print(f"\nResults saved to: {args.results}")
    print(f"Predictions saved to: {preds_dir}/")
    print("\nTo generate a Markdown table:")
    print(f"  python eval/results_table.py --results {args.results} "
          f"--output data/benchmarks/results/locomo10_table.md")


if __name__ == "__main__":
    main()
