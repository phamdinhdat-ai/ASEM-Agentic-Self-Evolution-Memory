"""
Run ASEM experiments on the locomo10.json sample dataset.

Converts raw LoCoMo conversations + QA pairs into sequential-memory evaluation
format, then runs all systems (6 baselines + ASEM) with per-category breakdown.

Usage
-----
    # Quick smoke test (10 examples, fast)
    python scripts/run_locomo10_experiments.py --limit 10

    # Full run on all ~1990 QA pairs
    python scripts/run_locomo10_experiments.py

    # Specific systems only, with BERTScore
    python scripts/run_locomo10_experiments.py \
        --systems NoMemory FullContext ASEM \
        --metrics em rougeL bertscore_f1

    # With per-category breakdown
    python scripts/run_locomo10_experiments.py --per-category

    # Using a different backend
    python scripts/run_locomo10_experiments.py \
        --config configs/langchain_ollama.yaml
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
from typing import Any, Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# LoCoMo category metadata
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "commonsense",
    4: "conversational",
    5: "adversarial",
}


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
    evidence_keys = set()
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
        - query:    enriched question with speaker context
        - answer:   gold answer string
        - history:  list of dialogue turn strings (chronological)
        - category: int (1-5)
        - category_name: str
        - session_id: str
        - evidence: list of evidence dia_ids
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

    return eval_items


# ---------------------------------------------------------------------------
# Metrics (mirrors eval/evaluate.py but self-contained)
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
# System runners — thin wrappers that process full sequential history
# ---------------------------------------------------------------------------

class SystemRunner:
    """Base class for system wrappers that handle full sequential history."""

    def __init__(self, name: str):
        self.name = name

    def answer(self, query: str, history: List[str]) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset per-conversation state (e.g. memory bank). Called between conversations."""
        pass


class NoMemoryRunner(SystemRunner):
    """Backbone-only — ignores all history."""

    def __init__(self, backend, prompt_template: str):
        super().__init__("NoMemory")
        self._backend = backend
        self._prompt = prompt_template

    def answer(self, query: str, history: List[str]) -> str:
        return self._backend.generate(self._prompt.format(query=query))


class FullContextRunner(SystemRunner):
    """All history concatenated into the context window."""

    def __init__(self, backend, prompt_template: str, max_history_turns: int = 0):
        super().__init__("FullContext")
        self._backend = backend
        self._prompt = prompt_template
        self._max_history = max_history_turns

    def answer(self, query: str, history: List[str]) -> str:
        h = history
        if self._max_history > 0 and len(h) > self._max_history:
            # Keep first few (setup) and last N-5 (most recent) turns
            keep_first = 5
            keep_last = self._max_history - keep_first
            h = h[:keep_first] + h[-keep_last:]
        context = "\n".join(h) if h else "(no prior conversation)"
        return self._backend.generate(
            self._prompt.format(query=query, context=context)
        )


class SimRetrievalRunner(SystemRunner):
    """Flat ANN retrieval — writes all history as atomic notes, then retrieves."""

    def __init__(self, backend, memory_bank, note_constructor, top_k: int, prompt_template: str):
        super().__init__("SimRetrieval")
        self._backend = backend
        self._bank = memory_bank
        self._note_constructor = note_constructor
        self._top_k = top_k
        self._prompt = prompt_template

    def answer(self, query: str, history: List[str]) -> str:
        for item in history:
            note = self._note_constructor.build(item, datetime.utcnow())
            self._bank.add(note)
        e_q = self._backend.embed(query)
        notes = self._bank.ann_search(e_q, k=self._top_k)
        context = "\n".join([n.c for n in notes]) if notes else "(no relevant memory)"
        return self._backend.generate(
            self._prompt.format(query=query, context=context)
        )


class AtomicLinkingRunner(SystemRunner):
    """Notes + bidirectional linking — writes all history with full Stage 1 + 3."""

    def __init__(
        self, backend, memory_bank, note_constructor, link_evolver,
        top_k: int, prompt_template: str,
    ):
        super().__init__("AtomicLinking")
        self._backend = backend
        self._bank = memory_bank
        self._note_constructor = note_constructor
        self._link_evolver = link_evolver
        self._top_k = top_k
        self._prompt = prompt_template

    def answer(self, query: str, history: List[str]) -> str:
        for item in history:
            note = self._note_constructor.build(item, datetime.utcnow())
            self._bank.add(note)
            self._link_evolver.link_and_evolve(note, self._bank)
        e_q = self._backend.embed(query)
        notes = self._bank.ann_search(e_q, k=self._top_k)
        context = "\n".join([n.c for n in notes]) if notes else "(no relevant memory)"
        return self._backend.generate(
            self._prompt.format(query=query, context=context)
        )


class RLManagerOnlyRunner(SystemRunner):
    """RL write ops + similarity retrieval — writes all history through MM."""

    def __init__(
        self, backend, memory_bank, note_constructor, memory_manager,
        top_k: int, prompt_template: str,
    ):
        super().__init__("RLManagerOnly")
        self._backend = backend
        self._bank = memory_bank
        self._note_constructor = note_constructor
        self._memory_manager = memory_manager
        self._top_k = top_k
        self._prompt = prompt_template

    def answer(self, query: str, history: List[str]) -> str:
        for item in history:
            note = self._note_constructor.build(item, datetime.utcnow())
            existing = self._bank.list_notes()
            op, target = self._memory_manager.select_op(item, existing)
            if op.value == "ADD":
                self._bank.add(note)
            elif op.value == "UPDATE":
                updated = self._merge_update(target, note)
                self._bank.add(updated)
            elif op.value == "DELETE" and target is not None:
                self._bank.delete(target.id)
        e_q = self._backend.embed(query)
        notes = self._bank.ann_search(e_q, k=self._top_k)
        context = "\n".join([n.c for n in notes]) if notes else "(no relevant memory)"
        return self._backend.generate(
            self._prompt.format(query=query, context=context)
        )

    @staticmethod
    def _merge_update(target, note):
        if target is None:
            return note
        from asem.note import Note
        return Note(
            id=target.id, c=note.c, t=note.t,
            K=note.K, G=note.G, X=note.X,
            e=note.e, L=target.L, z=note.z, q=target.q,
        )


class ValueRetrievalOnlyRunner(SystemRunner):
    """Value-aware retrieval + utility updates — writes all history."""

    def __init__(
        self, backend, memory_bank, note_constructor,
        retriever, utility_updater, answer_agent,
    ):
        super().__init__("ValueRetrievalOnly")
        self._backend = backend
        self._bank = memory_bank
        self._note_constructor = note_constructor
        self._retriever = retriever
        self._utility_updater = utility_updater
        self._answer_agent = answer_agent

    def answer(self, query: str, history: List[str]) -> str:
        for item in history:
            note = self._note_constructor.build(item, datetime.utcnow())
            self._bank.add(note)
        used_notes, answer = self._answer_agent.distil_and_answer(
            query,
            self._retriever.retrieve(query, self._bank),
        )
        self._utility_updater.update(
            reward=1.0,
            used_notes=used_notes,
            memory_bank=self._bank,
        )
        return answer


class ASEMRunner(SystemRunner):
    """Full ASEM pipeline — write path + read path + update path per turn."""

    def __init__(self, pipeline):
        super().__init__("ASEM")
        self._pipeline = pipeline

    def answer(self, query: str, history: List[str]) -> str:
        for item in history:
            self._pipeline.write_path(item, datetime.utcnow())
        _, answer = self._pipeline.read_path(query)
        return answer


# ---------------------------------------------------------------------------
# System factory — builds all runners from a YAML config
# ---------------------------------------------------------------------------

def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def build_runners(
    config_path: str = "configs/locomo_openai.yaml",
    db_dir: str = "data/benchmarks/eval_banks_locomo10",
    systems: Optional[List[str]] = None,
    max_history_turns: int = 0,
) -> Dict[str, SystemRunner]:
    """Build all system runners from a YAML config.

    Args:
        max_history_turns: If > 0, truncate history for FullContextRunner
            to this many turns (keeps first 5 + last N-5).
    """

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

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    note_prompt = _load_text("data/prompts/P1_note_construction.txt")
    link_prompt = _load_text("data/prompts/P2_link_generation.txt")
    evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

    os.makedirs(db_dir, exist_ok=True)

    # Shared components (each runner gets its own memory bank for isolation)
    def _make_bank(name: str) -> MemoryBank:
        path = os.path.join(db_dir, f"{name}.sqlite")
        # Remove stale files from previous crashed runs
        for suffix in ["", "-wal", "-shm", "-journal"]:
            p = path + suffix
            if os.path.exists(p):
                try:
                    os.remove(p)
                except PermissionError:
                    import time
                    time.sleep(0.5)
                    try:
                        os.remove(p)
                    except PermissionError:
                        pass  # will fail below with clear error
        return MemoryBank(path)

    # QA prompts — direct, task-focused, minimal tokens
    no_memory_prompt = (
        "Answer concisely (a few words or one sentence). "
        "Give your best guess if unsure.\n"
        "Question: {query}\n"
        "Answer:"
    )

    retrieval_prompt = (
        "Answer the question using the conversation below. "
        "Reply with ONLY the answer, no explanation.\n"
        "Conversation:\n{context}\n"
        "Question: {query}\n"
        "Answer:"
    )

    all_runners: Dict[str, SystemRunner] = {}

    # --- NoMemory ---
    if systems is None or "NoMemory" in systems:
        all_runners["NoMemory"] = NoMemoryRunner(backend, no_memory_prompt)

    # --- FullContext ---
    if systems is None or "FullContext" in systems:
        all_runners["FullContext"] = FullContextRunner(
            backend, retrieval_prompt, max_history_turns=max_history_turns,
        )

    # --- SimRetrieval ---
    if systems is None or "SimRetrieval" in systems:
        nc_sr = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
        all_runners["SimRetrieval"] = SimRetrievalRunner(
            backend=backend,
            memory_bank=_make_bank("simretrieval"),
            note_constructor=nc_sr,
            top_k=hp["k2"],
            prompt_template=retrieval_prompt,
        )

    # --- AtomicLinking ---
    if systems is None or "AtomicLinking" in systems:
        bank_al = _make_bank("atomiclinking")
        nc_al = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
        le_al = LinkEvolver(
            backend=backend,
            link_prompt_template=link_prompt,
            evolve_prompt_template=evolve_prompt,
            k=hp["k"],
        )
        all_runners["AtomicLinking"] = AtomicLinkingRunner(
            backend=backend,
            memory_bank=bank_al,
            note_constructor=nc_al,
            link_evolver=le_al,
            top_k=hp["k2"],
            prompt_template=retrieval_prompt,
        )

    # --- RLManagerOnly ---
    if systems is None or "RLManagerOnly" in systems:
        bank_rl = _make_bank("rlmanageronly")
        nc_rl = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
        mm_rl = MemoryManager(backend=backend, prompt_template=(
            "Decide memory write operation. Output JSON:\n"
            '{{"op": "ADD|UPDATE|DELETE|NOOP", "target_id": "<note_id or null>"}}\n'
            "Rules: ADD if new info. UPDATE if similar note exists. "
            "DELETE if contradicted. NOOP if irrelevant.\n"
            "Content: {content}\n"
            "Existing notes: {memory}"
        ))
        all_runners["RLManagerOnly"] = RLManagerOnlyRunner(
            backend=backend,
            memory_bank=bank_rl,
            note_constructor=nc_rl,
            memory_manager=mm_rl,
            top_k=hp["k2"],
            prompt_template=retrieval_prompt,
        )

    # --- ValueRetrievalOnly ---
    if systems is None or "ValueRetrievalOnly" in systems:
        bank_vr = _make_bank("valueretrievalonly")
        nc_vr = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
        retriever_vr = HybridRetriever(
            backend=backend,
            k1=hp["k1"], k2=hp["k2"],
            delta=hp["delta"], lambda_weight=hp["lambda"],
        )
        aa_vr = AnswerAgent(
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
        uu_vr = UtilityUpdater(
            backend=backend, alpha=hp["alpha"], q0=hp["q0"],
            summary_prompt_template=(
            "Summarize this interaction as a memory note. "
            "Output 1-2 factual sentences capturing what was learned.\n"
            "Query: {query}\n"
            "Answer: {answer}\n"
            "Reward: {reward}"
        ),
            note_constructor=nc_vr,
        )
        all_runners["ValueRetrievalOnly"] = ValueRetrievalOnlyRunner(
            backend=backend,
            memory_bank=bank_vr,
            note_constructor=nc_vr,
            retriever=retriever_vr,
            utility_updater=uu_vr,
            answer_agent=aa_vr,
        )

    # --- ASEM (full pipeline) ---
    if systems is None or "ASEM" in systems:
        bank_asem = _make_bank("asem")
        nc_asem = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
        mm_asem = MemoryManager(backend=backend, prompt_template=(
            "Decide memory write operation. Output JSON:\n"
            '{{"op": "ADD|UPDATE|DELETE|NOOP", "target_id": "<note_id or null>"}}\n'
            "Rules: ADD if new info. UPDATE if similar note exists. "
            "DELETE if contradicted. NOOP if irrelevant.\n"
            "Content: {content}\n"
            "Existing notes: {memory}"
        ))
        le_asem = LinkEvolver(
            backend=backend,
            link_prompt_template=link_prompt,
            evolve_prompt_template=evolve_prompt,
            k=hp["k"],
        )
        retriever_asem = HybridRetriever(
            backend=backend,
            k1=hp["k1"], k2=hp["k2"],
            delta=hp["delta"], lambda_weight=hp["lambda"],
        )
        aa_asem = AnswerAgent(
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
        uu_asem = UtilityUpdater(
            backend=backend, alpha=hp["alpha"], q0=hp["q0"],
            summary_prompt_template=(
            "Summarize this interaction as a memory note. "
            "Output 1-2 factual sentences capturing what was learned.\n"
            "Query: {query}\n"
            "Answer: {answer}\n"
            "Reward: {reward}"
        ),
            note_constructor=nc_asem,
        )
        pipeline = ASEMPipeline(
            memory_bank=bank_asem,
            note_constructor=nc_asem,
            memory_manager=mm_asem,
            link_evolver=le_asem,
            retriever=retriever_asem,
            answer_agent=aa_asem,
            utility_updater=uu_asem,
        )
        all_runners["ASEM"] = ASEMRunner(pipeline=pipeline)

    return all_runners


# ---------------------------------------------------------------------------
# Per-category split
# ---------------------------------------------------------------------------

def split_by_category(
    examples: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    by_cat: Dict[str, List] = defaultdict(list)
    for ex in examples:
        key = ex.get("category_name") or f"cat{ex.get('category', 0)}"
        by_cat[key].append(ex)
    return dict(by_cat)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ASEM experiments on locomo10.json"
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
        help=(
            "Systems to run. Choices: NoMemory FullContext SimRetrieval "
            "AtomicLinking RLManagerOnly ValueRetrievalOnly ASEM. "
            "Default: all."
        ),
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
        help=(
            "Truncate history for FullContext to this many turns "
            "(0 = no truncation). History is very long in LoCoMo (~688 turns) "
            "so a value of 100-200 is recommended for FullContext."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Force-clean all previous results, predictions, and databases before running.",
    )
    args = parser.parse_args()

    # Use timestamped db-dir to avoid Windows SQLite file locks from previous runs
    from datetime import datetime as _dt
    db_dir = os.path.join(args.db_dir, _dt.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(db_dir, exist_ok=True)
    print(f"DB dir: {db_dir}")

    # ------------------------------------------------------------------
    # Step 1: Convert locomo10.json → eval format
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Convert locomo10.json to eval format")
    print("=" * 60)
    eval_data = convert_locomo10_to_eval(args.input, limit=args.limit)

    # Print category distribution
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
    runners = build_runners(
        config_path=args.config,
        db_dir=db_dir,
        systems=args.systems,
        max_history_turns=args.max_history_turns,
    )
    print(f"  Systems: {list(runners.keys())}")

    # ------------------------------------------------------------------
    # Step 3: Run evaluation
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STEP 3: Run evaluation  (metrics: {args.metrics})")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    preds_dir = os.path.join(os.path.dirname(args.results), "preds")
    os.makedirs(preds_dir, exist_ok=True)

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

    for sys_name, runner in runners.items():
        key = f"locomo10/{sys_name}"
        partial_key = f"locomo10/{sys_name}/__partial__"

        if key in results:
            print(f"\n  [{sys_name}] already completed — skipping")
            continue

        print(f"\n  [{sys_name}] running on {len(eval_data)} examples ...")

        preds_path = os.path.join(preds_dir, f"locomo10_{sys_name}.jsonl")
        done_ids: set = set()
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
                partial_metrics = compute_metrics(preds_so_far, refs_so_far, args.metrics)
                partial_metrics["__n__"] = len(done_ids)
                partial_metrics["__total__"] = len(eval_data)
                results[partial_key] = partial_metrics
                _flush_results(results)

        runner.reset()

        with open(preds_path, "a", encoding="utf-8") as fp:
            for idx, item in enumerate(eval_data):
                if idx in done_ids:
                    continue

                query = str(item.get("query", ""))
                ref = str(item.get("answer", ""))
                history = [str(h) for h in item.get("history", [])]

                try:
                    pred = runner.answer(query, history)
                except Exception as exc:
                    print(f"\n    ERROR on example {idx}: {exc}")
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

                if (idx + 1) % 5 == 0 or (idx + 1) == len(eval_data):
                    pct = (idx + 1) / len(eval_data) * 100
                    print(f"    [{sys_name}] {idx + 1}/{len(eval_data)} "
                          f"({pct:.0f}%)  latest pred: {pred[:80]!r}", flush=True)

                # Save partial metrics every 25 examples
                if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_data):
                    partial_metrics = compute_metrics(
                        preds_so_far, refs_so_far, args.metrics
                    )
                    partial_metrics["__n__"] = len(preds_so_far)
                    partial_metrics["__total__"] = len(eval_data)
                    results[partial_key] = partial_metrics
                    _flush_results(results)

        # Final metrics
        final_metrics = compute_metrics(preds_so_far, refs_so_far, args.metrics)
        results[key] = final_metrics
        results.pop(partial_key, None)
        _flush_results(results)
        print(f"    [{sys_name}] FINAL: {final_metrics}")

    # ------------------------------------------------------------------
    # Step 4: Per-category breakdown (optional)
    # ------------------------------------------------------------------
    if args.per_category:
        print(f"\n{'='*60}")
        print("STEP 4: Per-category breakdown")
        print(f"{'='*60}")

        for cat_name, cat_examples in sorted(by_cat.items()):
            cat_preds: Dict[str, List[str]] = defaultdict(list)
            cat_refs: List[str] = [ex["answer"] for ex in cat_examples]

            preds_dir_cat = os.path.join(preds_dir, "by_category")
            os.makedirs(preds_dir_cat, exist_ok=True)

            for sys_name, runner in runners.items():
                cat_preds_path = os.path.join(
                    preds_dir_cat, f"locomo10_{cat_name}_{sys_name}.jsonl"
                )
                cat_preds[sys_name] = []

                if os.path.exists(cat_preds_path):
                    with open(cat_preds_path, "r", encoding="utf-8") as fp:
                        for line in fp:
                            line = line.strip()
                            if line:
                                cat_preds[sys_name].append(
                                    json.loads(line)["pred"]
                                )
                    if len(cat_preds[sys_name]) == len(cat_examples):
                        print(f"  [{cat_name}/{sys_name}] cached — skipping")
                        continue

                runner.reset()
                cat_preds[sys_name] = []
                with open(cat_preds_path, "w", encoding="utf-8") as fp:
                    for idx, ex in enumerate(cat_examples):
                        try:
                            pred = runner.answer(ex["query"], ex["history"])
                        except Exception:
                            pred = ""
                        cat_preds[sys_name].append(pred)
                        fp.write(json.dumps({
                            "idx": idx, "pred": pred, "ref": ex["answer"],
                        }) + "\n")

                metrics = compute_metrics(
                    cat_preds[sys_name], cat_refs, args.metrics
                )
                key = f"locomo10_cat_{cat_name}/{sys_name}"
                results[key] = metrics
                print(f"  [{cat_name}/{sys_name}]: {metrics}")

        _flush_results(results)

    # ------------------------------------------------------------------
    # Step 5: Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    header_metrics = args.metrics
    col_w = 16
    header = f"{'System':<30}" + "".join(f"{m:>{col_w}}" for m in header_metrics)
    print(header)
    print("-" * len(header))

    for key, metrics in sorted(results.items()):
        if "/" not in key or "__partial__" in key:
            continue
        dataset, sys_name = key.split("/", 1)
        if dataset != "locomo10":
            continue
        row = f"{sys_name:<30}" + "".join(
            f"{metrics.get(m, 0.0):>{col_w}.4f}" for m in header_metrics
        )
        print(row)

    if args.per_category:
        print(f"\n{'='*60}")
        print("PER-CATEGORY BREAKDOWN")
        print(f"{'='*60}")
        for cat_name in sorted(by_cat.keys()):
            print(f"\n  [{cat_name}]")
            cat_header = f"  {'System':<28}" + "".join(
                f"{m:>{col_w}}" for m in header_metrics
            )
            print(cat_header)
            print("  " + "-" * (len(cat_header) - 2))
            for sys_name in runners.keys():
                cat_key = f"locomo10_cat_{cat_name}/{sys_name}"
                if cat_key in results:
                    m = results[cat_key]
                    row = f"  {sys_name:<28}" + "".join(
                        f"{m.get(met, 0.0):>{col_w}.4f}" for met in header_metrics
                    )
                    print(row)

    print(f"\nResults saved to: {args.results}")
    print(f"Predictions saved to: {preds_dir}/")
    print("\nTo generate a Markdown table:")
    print(f"  python eval/results_table.py --results {args.results} "
          f"--output data/benchmarks/results/locomo10_table.md")


if __name__ == "__main__":
    main()
