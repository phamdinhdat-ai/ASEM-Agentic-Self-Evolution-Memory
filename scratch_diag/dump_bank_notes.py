"""Dump all notes mentioning Caroline or Melanie to see what the bank actually has."""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.getcwd())

from eval.phase_runner import bank_file
from asem.memory_bank import MemoryBank

bank_path = bank_file("static/memory_banks/locomo10", "ds_thg", "ASEM-THG", "locomo_0000")
mb = MemoryBank(db_path=bank_path)
notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()

print(f"Total notes: {len(notes)}\n")
for n in notes:
    desc = getattr(n, "description", "") or ""
    speaker = getattr(n, "speaker", "") or ""
    sd = getattr(n, "session_date", "") or ""
    tags = getattr(n, "tags", []) or []
    kw = getattr(n, "keywords", []) or []
    # Show notes mentioning Caroline or Melanie
    combined = (desc + " " + speaker).lower()
    if "caroline" in combined or "melanie" in combined:
        print(f"  [{n.id[:8]}] speaker={speaker:12s}  date={sd[:12]}  tags={tags[:3]}")
        print(f"    {desc[:200]}")
        print()
