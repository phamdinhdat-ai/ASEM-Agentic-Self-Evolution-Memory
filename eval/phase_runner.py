"""Phase-separated benchmark runner.

Implements the "ingest once, retrieve with many backbones" workflow:

  Phase A — ingest
      For every conversation, build each system's memory bank once and persist
      it to ``<bank_root>/<tag>/<system>/<conversation_id>/<bank>.sqlite``.

  Phase B — retrieve
      Load those persisted banks and answer every QA pair. Because the notes
      (and their vectors) are fixed after ingestion, only the *backbone LLM*
      differs between retrieval runs — which is exactly what a model-size sweep
      wants to measure.

The two phases communicate purely through the bank directory and the dataset,
so they can run in separate processes / days, and one ingestion result can be
reused by any number of retrieval configs (1B, 1.5B, 4B, ...).

NOTE: every config used for ingestion and retrieval must share the same
``embedder_name`` — the embedder produces the vectors stored in the bank.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from asem.logging_utils import get_logger
from eval.systems import (
    ALL_SYSTEMS,
    BANK_FILE_NAMES,
    NO_BANK_SYSTEMS,
    build_system,
)

logger = get_logger(__name__)


CATEGORY_NAMES: Dict[int, str] = {
    1: "single_hop",
    2: "temporal",
    3: "commonsense",
    4: "conversational",
    5: "adversarial",
}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """One LoCoMo session: number, absolute date string and formatted turns."""
    num: int
    date: str
    turns: List[str]


def load_raw_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_sessions(conversation: Dict[str, Any]) -> List[Session]:
    """Extract sessions from a LoCoMo ``conversation`` object, sorted by number."""
    numbers: Set[int] = set()
    for key in conversation:
        m = re.match(r"session_(\d+)$", str(key))
        if m:
            numbers.add(int(m.group(1)))

    sessions: List[Session] = []
    for num in sorted(numbers):
        turns_data = conversation.get(f"session_{num}", [])
        if not isinstance(turns_data, list):
            continue
        turns: List[str] = []
        for turn in turns_data:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            content = f"[{speaker}] {text}"
            blip = turn.get("blip_caption", "")
            if blip:
                content += f" (image: {blip})"
            turns.append(content)
        if turns:
            sessions.append(
                Session(num=num, date=str(conversation.get(f"session_{num}_date_time", "")), turns=turns)
            )
    return sessions


def _session_label(session: Session) -> str:
    label = f"session_{session.num}"
    if session.date:
        label += f" — {session.date}"
    return label


def sessions_to_batches(sessions: Sequence[Session]) -> List[Tuple[str, List[str]]]:
    """``(label, turns)`` batches consumed by ASEM v1 and the baselines."""
    return [(_session_label(s), list(s.turns)) for s in sessions]


def sessions_to_fast(sessions: Sequence[Session]) -> List[Dict[str, Any]]:
    """Session dicts consumed by ``FastASEMSystem.ingest_conversation``."""
    return [
        {"turns": list(s.turns), "date": s.date, "session_id": f"s{s.num}"}
        for s in sessions
    ]


def conversation_index(session_id: str) -> int:
    """``'locomo_0007'`` -> ``7``."""
    m = re.search(r"(\d+)$", str(session_id))
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Bank paths
# ---------------------------------------------------------------------------

def bank_dir(bank_root: str, tag: str, system: str, conversation_id: str) -> str:
    """Deterministic per-conversation bank directory for one system."""
    return os.path.join(bank_root, tag, system, conversation_id)


def bank_file(bank_root: str, tag: str, system: str, conversation_id: str) -> str:
    """Full path of the SQLite bank file for one system/conversation."""
    name = BANK_FILE_NAMES.get(system, system.lower())
    return os.path.join(bank_dir(bank_root, tag, system, conversation_id), f"{name}.sqlite")


def bank_exists(bank_root: str, tag: str, system: str, conversation_id: str) -> bool:
    return os.path.exists(bank_file(bank_root, tag, system, conversation_id))


def working_copy(
    src_file: str,
    work_root: str,
    tag: str,
    model_tag: str,
    system: str,
    conversation_id: str,
) -> str:
    """Copy a canonical bank into a per-run working dir and return that dir.

    Some systems (e.g. ``ValueRetrievalOnly``) legitimately **write** to their
    bank while answering (q-value updates + experience notes). Copying first
    keeps the ingested bank pristine, so every retrieval backbone starts from
    the exact same memory state and results stay reproducible.

    Uses SQLite's online backup API (not a raw file copy) so any WAL/journal
    content is included in the snapshot.
    """
    dest_dir = os.path.join(work_root, tag, model_tag, system, conversation_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, os.path.basename(src_file))
    # Drop any stale copy + sidecars so each run starts clean.
    for suffix in ("", "-wal", "-shm", "-journal"):
        stale = dest_file + suffix
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass
    src = sqlite3.connect(src_file)
    dst = sqlite3.connect(dest_file)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest_dir


# ---------------------------------------------------------------------------
# Backend / system construction
# ---------------------------------------------------------------------------

def build_backend_from_config(config_path: str) -> Any:
    """Instantiate the inference backend declared in a YAML config."""
    import yaml
    from asem.backends import build_backend

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "inference" not in cfg:
        raise ValueError(f"Config {config_path!r} has no 'inference' block")
    return build_backend(cfg["inference"])


def build_context_baseline(name: str, backend: Any):
    """Build NoMemory / FullContext directly (they hold no memory bank)."""
    from eval.baselines import FullContext, NoMemory
    from eval.systems import _FULL_CONTEXT_PROMPT, _NO_MEMORY_PROMPT

    if name == "NoMemory":
        return NoMemory(backend=backend, prompt_template=_NO_MEMORY_PROMPT)
    if name == "FullContext":
        return FullContext(
            backend=backend,
            prompt_template=_FULL_CONTEXT_PROMPT,
            max_history_turns=0,
        )
    raise ValueError(f"Not a context baseline: {name}")


def build_eval_system(
    name: str,
    config_path: str,
    db_dir: str,
    backend: Any = None,
):
    """Build any eval system; context baselines ignore ``db_dir``."""
    if name in NO_BANK_SYSTEMS:
        return build_context_baseline(name, backend)
    return build_system(name, config_path, db_dir, backend=backend)


def close_system(system: Any) -> None:
    """Close the SQLite connection so Windows releases the file lock."""
    bank = getattr(system, "memory_bank", None)
    if bank is None:
        pipeline = getattr(system, "pipeline", None)
        bank = getattr(pipeline, "memory_bank", None)
    if bank is not None:
        try:
            bank.close()
        except Exception:  # noqa: BLE001
            pass


def system_bank_size(system: Any) -> int:
    bank = getattr(system, "memory_bank", None)
    if bank is None:
        bank = getattr(getattr(system, "pipeline", None), "memory_bank", None)
    if bank is None:
        return 0
    try:
        return bank.size()
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_system(system: Any, name: str, sessions: Sequence[Session]) -> None:
    """Ingest all sessions of one conversation into ``system``."""
    if name == "FastASEM":
        system.ingest_conversation(sessions_to_fast(sessions))
        return

    if name == "ASEMv2":
        # ASEMv2 ingests a single session's dialogue per call (see run_asem_v2).
        for session in sessions:
            header = f"[Session {session.num}"
            if session.date:
                header += f" — {session.date}"
            header += "]"
            system.ingest_conversation([header] + list(session.turns))
        return

    # ASEM v1 + memory baselines accept (label, turns) batches.
    system.ingest_conversation(sessions_to_batches(sessions))


def finalize_system(system: Any) -> int:
    """Run optional post-ingestion linking (ASEM v1); 0 for others."""
    finalize = getattr(system, "finalize_conversation", None)
    if finalize is None:
        return 0
    try:
        return int(finalize() or 0)
    except Exception as exc:  # noqa: BLE001
        logger.opt(exception=exc).warning("finalize_conversation failed")
        return 0


def run_ingest_phase(
    raw_dataset: Sequence[Dict[str, Any]],
    groups: Sequence[Sequence[Dict[str, Any]]],
    systems: Sequence[str],
    ingest_config: str,
    bank_root: str,
    tag: str,
    backend: Any = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Phase A — build and persist one bank per (system, conversation).

    Returns a manifest describing what was written.
    """
    backend = backend if backend is not None else build_backend_from_config(ingest_config)
    wanted = [s for s in systems if s not in NO_BANK_SYSTEMS]

    manifest: Dict[str, Any] = {
        "phase": "ingest",
        "tag": tag,
        "ingest_config": ingest_config,
        "bank_root": bank_root,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "systems": {},
        "conversations": [],
    }

    os.makedirs(bank_root, exist_ok=True)
    t0 = time.perf_counter()
    selected = list(groups)[:limit] if limit else list(groups)

    for group in selected:
        conversation_id = str(group[0].get("session_id", ""))
        conv_idx = conversation_index(conversation_id)
        if conv_idx < 0 or conv_idx >= len(raw_dataset):
            logger.warning("Skipping conversation with unknown id {!r}", conversation_id)
            continue

        record = raw_dataset[conv_idx]
        sessions = extract_sessions(record.get("conversation", {}))
        if not sessions:
            logger.warning("Conversation {} has no sessions — skipping", conversation_id)
            continue

        total_turns = sum(len(s.turns) for s in sessions)
        logger.info(
            "Ingest conversation {} | {} sessions | {} turns",
            conversation_id, len(sessions), total_turns,
        )

        entry: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "sessions": len(sessions),
            "turns": total_turns,
            "systems": {},
        }

        for name in wanted:
            bdir = bank_dir(bank_root, tag, name, conversation_id)
            os.makedirs(bdir, exist_ok=True)
            system = build_eval_system(name, ingest_config, bdir, backend=backend)
            try:
                ingest_system(system, name, sessions)
                new_edges = finalize_system(system)
                size = system_bank_size(system)
            finally:
                close_system(system)

            manifest["systems"].setdefault(name, {"banks": {}, "notes": 0})
            manifest["systems"][name]["banks"][conversation_id] = bank_file(
                bank_root, tag, name, conversation_id
            )
            manifest["systems"][name]["notes"] += size
            entry["systems"][name] = {"notes": size, "link_edges": new_edges}
            logger.info("  [{}] {} notes | {} new edges", name, size, new_edges)

        manifest["conversations"].append(entry)

    elapsed = time.perf_counter() - t0
    manifest["elapsed_sec"] = round(elapsed, 2)
    manifest["n_conversations"] = len(manifest["conversations"])

    manifest_path = os.path.join(bank_root, tag, "manifest.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.info(
        "Ingest phase complete | convs={} | systems={} | {:.1f}s | manifest={}",
        manifest["n_conversations"], list(manifest["systems"]), elapsed, manifest_path,
    )
    return manifest


# ---------------------------------------------------------------------------
# Retrieval / evaluation
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def exact_match(preds: Sequence[str], refs: Sequence[str]) -> float:
    matches = [1.0 if _normalize(p) == _normalize(r) else 0.0 for p, r in zip(preds, refs)]
    return sum(matches) / len(matches) if matches else 0.0


def compute_metrics(preds: List[str], refs: List[str], metric_names: Sequence[str]) -> Dict[str, float]:
    """EM / ROUGE-L / BERTScore-F1 using the HuggingFace ``evaluate`` package."""
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
        scores = bert.compute(predictions=preds, references=refs, lang="en")
        results["bertscore_f1"] = float(sum(scores["f1"]) / len(scores["f1"]))
    return results


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def model_tag_from_config(config_path: str) -> str:
    """Derive a filesystem-safe model tag from a config's inference block."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    inf = cfg.get("inference", {}) or {}
    if inf.get("backend") == "huggingface":
        model_id = (inf.get("huggingface") or {}).get("model_name_or_path", "model")
    else:
        model_id = (inf.get("langchain") or {}).get("model", "model")
    return _slug(model_id)


def run_retrieve_phase(
    groups: Sequence[Sequence[Dict[str, Any]]],
    systems: Sequence[str],
    config_path: str,
    bank_root: str,
    tag: str,
    backend: Any = None,
    metric_names: Sequence[str] = ("em", "rougeL"),
    preds_dir: Optional[str] = None,
    model_tag: Optional[str] = None,
    per_category: bool = False,
    limit: Optional[int] = None,
    require_banks: bool = True,
    work_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Phase B — answer every QA pair from the persisted banks.

    Returns a results dict with ``systems[name]["overall"|"per_category"|"n"]``.

    Args:
        work_root: When set, each bank is copied into this directory before it
            is opened, so answering never mutates the canonical ingested bank.
    """
    backend = backend if backend is not None else build_backend_from_config(config_path)
    model_tag = model_tag or model_tag_from_config(config_path)
    selected = list(groups)[:limit] if limit else list(groups)

    per_system: Dict[str, Dict[str, Any]] = {
        name: {"preds": [], "refs": [], "by_cat": {}, "done": set(), "missing": 0}
        for name in systems
    }

    # Resume: reload any predictions already written.
    if preds_dir:
        os.makedirs(preds_dir, exist_ok=True)
    preds_path = (
        os.path.join(preds_dir, f"{tag}__{model_tag}__{{system}}.jsonl") if preds_dir else None
    )
    if preds_path:
        for name in systems:
            path = preds_path.format(system=name)
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    state = per_system[name]
                    state["done"].add(rec["idx"])
                    state["preds"].append(rec["pred"])
                    state["refs"].append(rec["ref"])
                    cat = rec.get("category_name", "")
                    bucket = state["by_cat"].setdefault(cat, {"preds": [], "refs": []})
                    bucket["preds"].append(rec["pred"])
                    bucket["refs"].append(rec["ref"])
            if per_system[name]["done"]:
                logger.info("Resuming {}: {} saved predictions", name, len(per_system[name]["done"]))

    # Keep file handles open per system for incremental append.
    handles: Dict[str, Any] = {}
    if preds_path:
        for name in systems:
            handles[name] = open(preds_path.format(system=name), "a", encoding="utf-8")

    t0 = time.perf_counter()
    total_pairs = sum(len(g) for g in selected)

    try:
        for group in selected:
            conversation_id = str(group[0].get("session_id", ""))
            built: Dict[str, Any] = {}

            for name in systems:
                if name in NO_BANK_SYSTEMS:
                    built[name] = build_eval_system(name, config_path, "", backend)
                    continue
                bdir = bank_dir(bank_root, tag, name, conversation_id)
                src_file = bank_file(bank_root, tag, name, conversation_id)
                if not os.path.exists(src_file):
                    per_system[name]["missing"] += 1
                    msg = (
                        f"No bank for system={name} conversation={conversation_id} "
                        f"(expected {src_file})"
                    )
                    if require_banks:
                        raise FileNotFoundError(msg + " — run --phase ingest first")
                    logger.warning(msg + " — skipping")
                    continue
                if work_root:
                    bdir = working_copy(
                        src_file, work_root, tag, model_tag, name, conversation_id
                    )
                built[name] = build_eval_system(name, config_path, bdir, backend)
                logger.info(
                    "[{}] loaded bank | conv={} | {} notes",
                    name, conversation_id, system_bank_size(built[name]),
                )

            for item in group:
                idx = int(item.get("_idx", -1))
                query = str(item.get("query", ""))
                ref = str(item.get("answer", ""))
                history = [str(h) for h in item.get("history", [])]
                category_name = item.get("category_name") or CATEGORY_NAMES.get(
                    int(item.get("category", 0)), "unknown"
                )

                for name, system in built.items():
                    state = per_system[name]
                    if idx in state["done"]:
                        continue
                    hist = history if name in NO_BANK_SYSTEMS else []
                    try:
                        pred = system.answer(query, hist)
                    except Exception as exc:  # noqa: BLE001
                        logger.opt(exception=exc).error(
                            "[{}] answer failed | idx={}", name, idx
                        )
                        pred = ""
                    state["preds"].append(pred)
                    state["refs"].append(ref)
                    state["done"].add(idx)
                    bucket = state["by_cat"].setdefault(category_name, {"preds": [], "refs": []})
                    bucket["preds"].append(pred)
                    bucket["refs"].append(ref)

                    if handles:
                        handles[name].write(json.dumps({
                            "idx": idx,
                            "conversation_id": conversation_id,
                            "category": item.get("category", 0),
                            "category_name": category_name,
                            "question": item.get("raw_question", query),
                            "query": query,
                            "pred": pred,
                            "ref": ref,
                        }) + "\n")
                        handles[name].flush()

            for system in built.values():
                close_system(system)
    finally:
        for fh in handles.values():
            try:
                fh.close()
            except Exception:  # noqa: BLE001
                pass

    elapsed = time.perf_counter() - t0
    results: Dict[str, Any] = {
        "phase": "retrieve",
        "tag": tag,
        "model_tag": model_tag,
        "config": config_path,
        "metrics": list(metric_names),
        "elapsed_sec": round(elapsed, 2),
        "n_qa": total_pairs,
        "systems": {},
    }

    for name, state in per_system.items():
        entry: Dict[str, Any] = {
            "n": len(state["preds"]),
            "missing_conversations": state["missing"],
            "overall": compute_metrics(state["preds"], state["refs"], metric_names),
        }
        if per_category:
            entry["per_category"] = {
                cat: {
                    "n": len(bucket["preds"]),
                    **compute_metrics(bucket["preds"], bucket["refs"], metric_names),
                }
                for cat, bucket in sorted(state["by_cat"].items())
            }
        results["systems"][name] = entry

    logger.info(
        "Retrieve phase complete | model={} | {} QA | {:.1f}s",
        model_tag, total_pairs, elapsed,
    )
    for name, entry in results["systems"].items():
        logger.info("  [{}] n={} {}", name, entry["n"], entry["overall"])

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_results(path: str, results: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    os.replace(tmp, path)


def render_sweep_table(sweep: Dict[str, Dict[str, Any]], metric_names: Sequence[str]) -> str:
    """Markdown table: rows = systems, columns = model tags."""
    systems: List[str] = []
    for res in sweep.values():
        for name in res.get("systems", {}):
            if name not in systems:
                systems.append(name)

    model_tags = list(sweep.keys())
    metric = metric_names[0] if metric_names else "em"

    header = "| System | " + " | ".join(f"{tag} ({metric})" for tag in model_tags) + " |"
    sep = "|" + "---|" * (len(model_tags) + 1)
    lines = [header, sep]
    for system in systems:
        cells = []
        for tag in model_tags:
            entry = sweep[tag].get("systems", {}).get(system)
            value = entry.get("overall", {}).get(metric, 0.0) if entry else 0.0
            cells.append(f"{value:.4f}")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


__all__ = [
    "ALL_SYSTEMS",
    "CATEGORY_NAMES",
    "Session",
    "bank_dir",
    "bank_exists",
    "bank_file",
    "build_backend_from_config",
    "build_eval_system",
    "close_system",
    "compute_metrics",
    "conversation_index",
    "extract_sessions",
    "load_raw_dataset",
    "model_tag_from_config",
    "render_sweep_table",
    "run_ingest_phase",
    "run_retrieve_phase",
    "sessions_to_batches",
    "sessions_to_fast",
    "working_copy",
    "write_results",
]
