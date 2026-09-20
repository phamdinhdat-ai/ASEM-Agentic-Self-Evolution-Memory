"""Copy FastASEM's frozen banks from one tag into another.

FastASEM does not need re-ingesting: its extraction prompt already produces
entities/speaker and pronoun+date-resolved facts, and its banks were built with
the SAME `configs/models/deepseek_openai.yaml` (identical `config_sha256`).
So for a single-tag comparison we COPY the banks instead of paying for an
identical rebuild.

Writes a provenance sidecar so the copied banks are self-describing even though
`manifest.json`/`banks.json` are owned and rewritten by build_static_banks.py.

Usage:
  python scratch_diag/copy_fastasem_banks.py [SRC_TAG] [DST_TAG] [SYSTEM]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK_ROOT = os.path.join(ROOT, "static", "memory_banks", "locomo10")

SRC_TAG = sys.argv[1] if len(sys.argv) > 1 else "ds_nothink"
DST_TAG = sys.argv[2] if len(sys.argv) > 2 else "ds_fixed"
SYSTEM = sys.argv[3] if len(sys.argv) > 3 else "FastASEM"

BANK_FILE = "fast_asem.sqlite"


def main() -> None:
    src_index = os.path.join(BANK_ROOT, SRC_TAG, "banks.json")
    if not os.path.exists(src_index):
        raise SystemExit(f"no source banks.json at {src_index}")
    with open(src_index, "r", encoding="utf-8") as fh:
        index = json.load(fh)

    entries = index.get(SYSTEM)
    if not entries:
        raise SystemExit(f"{SYSTEM!r} not present in {src_index} (keys: {list(index)})")

    copied, skipped = [], []
    for conv, meta in sorted(entries.items()):
        src = os.path.join(BANK_ROOT, SRC_TAG, SYSTEM, conv, BANK_FILE)
        if not os.path.exists(src):
            skipped.append((conv, "source file missing"))
            continue
        dst_dir = os.path.join(BANK_ROOT, DST_TAG, SYSTEM, conv)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, BANK_FILE)
        shutil.copy2(src, dst)
        # Copy SQLite sidecars too, so the copy is transaction-consistent.
        for suffix in ("-wal", "-shm"):
            if os.path.exists(src + suffix):
                shutil.copy2(src + suffix, dst + suffix)
        copied.append({
            "conversation_id": conv,
            "notes": meta.get("notes"),
            "link_edges": meta.get("link_edges"),
            "bytes": os.path.getsize(dst),
        })

    provenance = {
        "system": SYSTEM,
        "copied_from_tag": SRC_TAG,
        "copied_to_tag": DST_TAG,
        "copied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": (
            "FastASEM's banks were built with an extraction prompt that already "
            "emits entities/speaker and resolves pronouns + relative dates, so the "
            "ASEM entities bug does not apply. Re-ingesting would reproduce the same "
            "banks (same config hash) at API cost with no added signal."
        ),
        "source_config_sha256": next(
            (m.get("config_sha256") for m in entries.values() if m.get("config_sha256")),
            None,
        ),
        "conversations": copied,
        "skipped": skipped,
    }
    out = os.path.join(BANK_ROOT, DST_TAG, f"{SYSTEM.upper()}_PROVENANCE.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)

    total_notes = sum(int(c["notes"] or 0) for c in copied)
    total_edges = sum(int(c["link_edges"] or 0) for c in copied)
    print(f"copied {len(copied)} {SYSTEM} bank(s)  {SRC_TAG} -> {DST_TAG}")
    print(f"  notes={total_notes}  link_edges={total_edges}")
    print(f"  config_sha256={provenance['source_config_sha256']}")
    for c in copied:
        print(f"    {c['conversation_id']}: {c['notes']} notes  {c['link_edges']} edges")
    for conv, why in skipped:
        print(f"    SKIPPED {conv}: {why}")
    print(f"\nprovenance -> {out}")


if __name__ == "__main__":
    main()
