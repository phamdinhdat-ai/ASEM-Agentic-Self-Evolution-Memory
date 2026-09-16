"""Static (ingest-once) memory-bank builder.

Phase A of the static-bank workflow: for every ingestion-based system, build one
frozen memory bank per conversation and persist it under

    <bank_root>/<tag>/<System>/<conversation_id>/<bank>.sqlite

plus a ``manifest.json`` (what was built, note counts, errors) and a
``banks.json`` index. Evaluation runs then read — never write — these banks, so a
single ingestion result can back any number of retrieval backbones.

Guarantees
----------
* **Idempotent / resumable**: a ``(system, conversation)`` pair is skipped only
  when the manifest records it as finished (``status: "ok"``/``"skipped"``) AND
  its bank file is non-empty. A bank left half-written by an interruption is
  detected and rebuilt from scratch, so an incomplete memory graph can never be
  mistaken for a complete one. ``force=True`` always rebuilds.
* **Auditable**: each bank records the config it was built with, so a tag that
  mixes two ingest configurations (e.g. thinking on, then thinking off) is
  visible in ``manifest.json`` instead of silently mislabelled.
* **Error-isolated**: a failure while building one bank is recorded in the
  manifest as ``status: "error"`` with a traceback and the run continues
  (unless ``fail_fast=True``), so one bad conversation cannot waste hours.
* **Crash-safe**: the manifest and index are rewritten atomically after every
  system, so Ctrl-C at any moment leaves a valid, resumable artifact.

The heavy lifting is delegated to ``eval.phase_runner`` (session extraction,
system construction, ingestion, finalisation) so the ingest path behaves exactly
like the existing phase benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence

from asem.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Filesystem / hashing helpers
# ---------------------------------------------------------------------------

def atomic_write_json(path: str, obj: Any) -> None:
    """Write ``obj`` as JSON through a temp file + ``os.replace``.

    A reader (or a post-crash inspection) therefore never sees a half-written
    manifest.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


def sha256_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_name_from_config(config_path: str) -> str:
    """Model id from a config's inference block (backend-agnostic).

    Each backend names its model differently — ``model`` for the API backends,
    ``model_name_or_path`` for HuggingFace — so read the block selected by
    ``backend`` instead of assuming "langchain".
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    inf = cfg.get("inference", {}) or {}
    block = inf.get(inf.get("backend")) or {}
    return str(block.get("model") or block.get("model_name_or_path") or "?")


def embedder_name_from_config(config_path: str) -> str:
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    inf = cfg.get("inference", {}) or {}
    block = inf.get(inf.get("backend"), {}) or {}
    return str(block.get("embedder_name", ""))


def thinking_from_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Thinking/reasoning setting of a config, for provenance.

    Returns the raw ``thinking`` block when present (e.g.
    ``{"type": "disabled"}``), else ``{"reasoning_effort": ...}``, else
    ``{"enable_reasoning": ...}``, else ``None`` (model default = thinking on).
    """
    import yaml

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    inf = cfg.get("inference", {}) or {}
    block = inf.get(inf.get("backend"), {}) or {}
    if block.get("thinking") is not None:
        return {"thinking": block["thinking"]}
    if block.get("reasoning_effort") is not None:
        return {"reasoning_effort": block["reasoning_effort"]}
    if block.get("enable_reasoning") is not None:
        return {"enable_reasoning": block["enable_reasoning"]}
    return None


def count_notes(db_file: str) -> int:
    """Number of notes in a bank, read from SQLite without building FAISS."""
    if not db_file or not os.path.exists(db_file) or os.path.getsize(db_file) == 0:
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def count_links(db_file: str) -> int:
    """Number of typed link edges in a bank, read from SQLite without FAISS.

    ``finalize_system()`` reports only the edges created by the post-ingestion
    finalize pass, and that pass exists for ASEM v1 alone — FastASEM builds its
    links during ingestion. Recording that value made every FastASEM bank look
    link-less (`link_edges: 0`) even when it was densely connected, so the
    figure is now read from the ``L`` column instead. It counts both the current
    ``LinkRecord`` dicts and the legacy flat-ID strings.
    """
    if not db_file or not os.path.exists(db_file) or os.path.getsize(db_file) == 0:
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT L FROM notes WHERE L IS NOT NULL AND L != '[]'"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return 0

    total = 0
    for (raw,) in rows:
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(items, list):
            total += len(items)
    return total


def remove_bank_files(db_file: str) -> None:
    """Delete a bank plus its WAL/SHM/journal sidecars (used by ``force``)."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        stale = db_file + suffix
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Manifest bookkeeping
# ---------------------------------------------------------------------------

def new_manifest(
    dataset: str,
    tag: str,
    bank_root: str,
    input_path: str,
    ingest_config: str,
) -> Dict[str, Any]:
    return {
        "phase": "ingest",
        "dataset": dataset,
        "tag": tag,
        "bank_root": bank_root,
        "input": input_path,
        "input_sha256": sha256_file(input_path),
        "ingest_config": ingest_config,
        "config_sha256": sha256_file(ingest_config),
        "model": model_name_from_config(ingest_config) if os.path.exists(ingest_config) else "",
        "embedder_name": (
            embedder_name_from_config(ingest_config) if os.path.exists(ingest_config) else ""
        ),
        "systems": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "updated_at": None,
        "elapsed_sec": 0.0,
        "conversations": [],
        "totals": {},
        "run": {},
    }


def load_manifest(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def conversation_entry(manifest: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
    for entry in manifest["conversations"]:
        if entry.get("conversation_id") == conversation_id:
            return entry
    entry = {"conversation_id": conversation_id, "sessions": 0, "turns": 0, "systems": {}}
    manifest["conversations"].append(entry)
    return entry


def recompute_totals(manifest: Dict[str, Any]) -> None:
    notes: Dict[str, int] = {}
    ok = skipped = errors = 0
    for conv in manifest.get("conversations", []):
        for name, state in (conv.get("systems") or {}).items():
            notes[name] = notes.get(name, 0) + int(state.get("notes") or 0)
            status = state.get("status")
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            elif status == "error":
                errors += 1
    manifest["totals"] = {
        "conversations": len(manifest.get("conversations", [])),
        "banks_ok": ok,
        "banks_skipped": skipped,
        "banks_error": errors,
        "notes": notes,
    }


def bank_index(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{System: {conversation_id: {path, notes, link_edges, status, bytes}}}``.

    ``link_edges`` is the total number of typed edges in the bank. ``new_edges``
    counts only those the finalize pass added (ASEM v1 cross-chunk linking);
    it is 0 for systems that link during ingestion.
    """
    index: Dict[str, Dict[str, Any]] = {}
    for conv in manifest.get("conversations", []):
        for name, state in (conv.get("systems") or {}).items():
            index.setdefault(name, {})[conv["conversation_id"]] = {
                "path": state.get("bank", ""),
                "notes": int(state.get("notes") or 0),
                "link_edges": int(state.get("link_edges") or 0),
                "new_edges": int(state.get("new_edges") or 0),
                "status": state.get("status"),
                "bytes": state.get("bytes", 0),
                "config_sha256": state.get("config_sha256"),
                "thinking": state.get("thinking"),
            }
    return index


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _ingest_one(
    name: str,
    config_path: str,
    bank_dir: str,
    sessions: Sequence[Any],
    backend: Any,
    bank_file_path: str = "",
) -> Dict[str, int]:
    """Build a single (system, conversation) bank in-process. Raises on failure.

    Returns ``notes``, ``link_edges`` (total edges now in the bank, comparable
    across systems) and ``new_edges`` (those created by the finalize pass).
    """
    from eval.phase_runner import (
        build_eval_system,
        close_system,
        finalize_system,
        ingest_system,
        system_bank_size,
    )

    system = build_eval_system(name, config_path, bank_dir, backend=backend)
    try:
        ingest_system(system, name, sessions)
        new_edges = finalize_system(system)
        size = system_bank_size(system)
    finally:
        close_system(system)

    # Count AFTER closing, so every system is measured the same way (the
    # finalize pass only exists for ASEM v1; the others link while ingesting).
    # `max` keeps the total honest if the file is unreadable.
    edges = max(count_links(bank_file_path), int(new_edges))
    return {"notes": int(size), "link_edges": edges, "new_edges": int(new_edges)}


def build_static_banks(
    raw_dataset: Sequence[Dict[str, Any]],
    groups: Sequence[Sequence[Dict[str, Any]]],
    systems: Sequence[str],
    ingest_config: str,
    bank_root: str,
    tag: str,
    dataset: str,
    backend: Any = None,
    force: bool = False,
    allow_empty: bool = False,
    fail_fast: bool = False,
    should_stop: Optional[Callable[[], bool]] = None,
    input_path: str = "",
) -> Dict[str, Any]:
    """Build/persist every requested system bank for every selected conversation.

    Args:
        groups: One QA-item list per conversation (only ``session_id`` is used to
            identify the conversation; the raw record supplies the sessions).
        backend: Pre-built inference backend. When ``None`` it is built from
            ``ingest_config`` — tests inject a stub here.
        should_stop: Returns True to stop gracefully after the in-flight system
            (used by the CLI's SIGINT handler).
        input_path: Dataset path, recorded in the manifest and hashed.

    Returns:
        The manifest dict (also written to ``<bank_root>/<tag>/manifest.json``).
    """
    from eval.phase_runner import (
        bank_dir,
        bank_file,
        build_backend_from_config,
        conversation_index,
        extract_sessions,
    )

    _stop = should_stop or (lambda: False)
    backend = backend if backend is not None else build_backend_from_config(ingest_config)

    tag_dir = os.path.join(bank_root, tag)
    os.makedirs(tag_dir, exist_ok=True)
    manifest_path = os.path.join(tag_dir, "manifest.json")
    index_path = os.path.join(tag_dir, "banks.json")

    cfg_sha = sha256_file(ingest_config)
    cfg_model = model_name_from_config(ingest_config) if os.path.exists(ingest_config) else ""
    cfg_thinking = thinking_from_config(ingest_config) if os.path.exists(ingest_config) else None

    manifest = load_manifest(manifest_path)
    if manifest is None:
        manifest = new_manifest(dataset, tag, bank_root, input_path, ingest_config)

    # The tag's config BEFORE this run. Used to attribute banks that were written
    # before per-bank provenance existed (otherwise a re-used bank would be
    # mislabelled with whatever config was passed today).
    prev_tag_cfg = manifest.get("ingest_config")
    prev_tag_sha = manifest.get("config_sha256")
    prev_tag_model = manifest.get("model")
    prev_tag_thinking = None
    for _h in (manifest.get("config_history") or []):
        if _h.get("config_sha256") == prev_tag_sha:
            prev_tag_thinking = _h.get("thinking")
            break
    if prev_tag_thinking is None and prev_tag_cfg and os.path.exists(str(prev_tag_cfg)):
        prev_tag_thinking = thinking_from_config(str(prev_tag_cfg))

    if prev_tag_sha and prev_tag_sha != cfg_sha:
        logger.warning(
            "Tag {!r} already holds banks built with a DIFFERENT ingest config "
            "({} -> {}). Existing banks are reused as-is, so this tag will MIX "
            "both configurations — each bank records its own config, and the "
            "split is listed in `config_history`. Prefer a fresh --tag for a "
            "clean comparison, or --force to rebuild everything on one config.",
            tag, prev_tag_cfg, ingest_config,
        )

    manifest["dataset"] = dataset
    manifest["tag"] = tag
    manifest["bank_root"] = bank_root
    manifest["ingest_config"] = ingest_config
    manifest["config_sha256"] = cfg_sha
    manifest["model"] = cfg_model
    manifest["embedder_name"] = (
        embedder_name_from_config(ingest_config) if os.path.exists(ingest_config) else ""
    )
    manifest["systems"] = sorted(set(manifest.get("systems", [])) | set(systems))

    # Every distinct configuration used against this tag, so a mixed tag is
    # visible rather than mislabelled as the most recent config.
    history = manifest.setdefault("config_history", [])
    if not any(h.get("config_sha256") == cfg_sha for h in history):
        history.append({
            "config_sha256": cfg_sha,
            "ingest_config": ingest_config,
            "model": cfg_model,
            "thinking": cfg_thinking,
            "first_used": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    # Stamped on every NEWLY built bank so you can tell which settings produced it.
    bank_meta: Dict[str, Any] = {
        "ingest_config": ingest_config,
        "config_sha256": cfg_sha,
        "model": cfg_model,
        "thinking": cfg_thinking,
    }

    def _reuse_meta(previous: Dict[str, Any]) -> Dict[str, Any]:
        """Provenance for a bank that is being REUSED (skipped), not rebuilt.

        Per-bank fields are authoritative; for banks written before per-bank
        provenance existed, fall back to the tag's config at the time — which is
        the previous config, not the one being added in this run.
        """
        if previous.get("config_sha256"):
            return {
                "ingest_config": previous.get("ingest_config"),
                "config_sha256": previous.get("config_sha256"),
                "model": previous.get("model"),
                "thinking": previous.get("thinking"),
            }
        return {
            "ingest_config": prev_tag_cfg,
            "config_sha256": prev_tag_sha,
            "model": prev_tag_model,
            "thinking": prev_tag_thinking,
        }

    def _save() -> None:
        recompute_totals(manifest)
        manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(index_path, bank_index(manifest))

    run = {"built": 0, "skipped": 0, "errors": 0, "notes": 0, "discarded_partial": 0}
    t_start = time.perf_counter()

    try:
        for group in groups:
            if _stop():
                break

            conversation_id = str(group[0].get("session_id", ""))
            idx = conversation_index(conversation_id)
            if idx < 0 or idx >= len(raw_dataset):
                logger.warning("Skipping conversation with unknown id {!r}", conversation_id)
                continue

            sessions = extract_sessions(raw_dataset[idx].get("conversation", {}))
            if not sessions:
                logger.warning("Conversation {} has no sessions — skipping", conversation_id)
                continue

            entry = conversation_entry(manifest, conversation_id)
            entry["sessions"] = len(sessions)
            entry["turns"] = sum(len(s.turns) for s in sessions)
            entry.pop("status", None)
            logger.info(
                "Conversation {} | {} sessions | {} turns",
                conversation_id, entry["sessions"], entry["turns"],
            )

            for name in systems:
                if _stop():
                    break

                bdir = bank_dir(bank_root, tag, name, conversation_id)
                bfile = bank_file(bank_root, tag, name, conversation_id)
                os.makedirs(bdir, exist_ok=True)

                previous = entry["systems"].get(name) or {}
                existing = count_notes(bfile)
                # "Finished" is an explicit fact recorded in the manifest, NOT
                # something inferred from the note count. An interrupted run
                # leaves a partially written bank that already contains notes,
                # so keying off `existing > 0` would silently accept an
                # INCOMPLETE memory bank as complete on the next retry.
                # "skipped" means an earlier run already found it complete.
                complete = previous.get("status") in ("ok", "skipped") and existing > 0

                if complete and not force:
                    # Re-read both counts from the bank: earlier runs recorded
                    # `link_edges: 0` for every non-ASEM system, and a skipped
                    # bank is the only chance to repair that stale figure.
                    edges = count_links(bfile)
                    entry["systems"][name] = {
                        **_reuse_meta(previous),
                        "notes": existing,
                        "link_edges": edges if edges else int(previous.get("link_edges") or 0),
                        "new_edges": int(previous.get("new_edges") or 0),
                        "status": "skipped",
                        "bank": bfile,
                        "bytes": os.path.getsize(bfile),
                        "elapsed_sec": 0.0,
                    }
                    run["skipped"] += 1
                    logger.info(
                        "  [{}] skipped — already complete ({} notes, {} edges)",
                        name, existing, edges,
                    )
                    _save()
                    continue

                if existing:
                    # Leftover from an interrupted (or config-changed) run. The
                    # builders reuse an existing SQLite file instead of clearing
                    # it, so rebuilding would APPEND to the partial notes.
                    if not force:
                        logger.warning(
                            "  [{}] discarding PARTIAL bank ({} notes) — the previous run "
                            "did not finish this conversation; re-ingesting it from scratch",
                            name, existing,
                        )
                    remove_bank_files(bfile)
                    run["discarded_partial"] += 1

                t0 = time.perf_counter()
                try:
                    result = _ingest_one(name, ingest_config, bdir, sessions, backend, bfile)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.perf_counter() - t0
                    logger.opt(exception=exc).error(
                        "  [{}] FAILED on {} after {:.1f}s", name, conversation_id, elapsed
                    )
                    entry["systems"][name] = {
                        **bank_meta,
                        "notes": 0,
                        "link_edges": 0,
                        "new_edges": 0,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=12),
                        "bank": bfile,
                        "bytes": 0,
                        "elapsed_sec": round(elapsed, 2),
                    }
                    run["errors"] += 1
                    _save()
                    if fail_fast:
                        raise
                    continue

                elapsed = time.perf_counter() - t0
                notes = result["notes"]
                if notes == 0 and not allow_empty:
                    status = "error"
                    run["errors"] += 1
                    logger.error("  [{}] produced 0 notes — recorded as an error", name)
                else:
                    status = "ok"
                    run["built"] += 1
                    run["notes"] += notes

                entry["systems"][name] = {
                    **bank_meta,
                    "notes": notes,
                    "link_edges": result["link_edges"],
                    "new_edges": result["new_edges"],
                    "status": status,
                    "bank": bfile,
                    "bytes": os.path.getsize(bfile) if os.path.exists(bfile) else 0,
                    "elapsed_sec": round(elapsed, 2),
                }
                logger.info(
                    "  [{}] {} notes | {} edges ({} new this pass) | {:.1f}s",
                    name, notes, result["link_edges"], result["new_edges"], elapsed,
                )
                _save()

        if _stop():
            for group in groups:
                cid = str(group[0].get("session_id", ""))
                if not any(c.get("conversation_id") == cid for c in manifest["conversations"]):
                    manifest["conversations"].append(
                        {"conversation_id": cid, "sessions": 0, "turns": 0,
                         "systems": {}, "status": "not_started"}
                    )
    finally:
        manifest["elapsed_sec"] = round(time.perf_counter() - t_start, 2)
        manifest["run"] = dict(run, interrupted=bool(_stop()))
        _save()

    return manifest


__all__ = [
    "atomic_write_json",
    "bank_index",
    "build_static_banks",
    "count_notes",
    "embedder_name_from_config",
    "load_manifest",
    "model_name_from_config",
    "new_manifest",
    "remove_bank_files",
    "recompute_totals",
    "sha256_file",
]
