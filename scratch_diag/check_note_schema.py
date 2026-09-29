from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.getcwd())
from eval.phase_runner import bank_file
from asem.memory_bank import MemoryBank

bank_path = bank_file("static/memory_banks/locomo10", "ds_thg", "ASEM-THG", "locomo_0000")
mb = MemoryBank(db_path=bank_path)
notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()
n = notes[0]
print("Note fields:", [k for k in dir(n) if not k.startswith("_")])
print("\nFirst note dict:")
for k, v in n.__dict__.items():
    print(f"  {k}: {repr(v)[:300]}")
print("\nSecond note dict:")
for k, v in notes[1].__dict__.items():
    print(f"  {k}: {repr(v)[:300]}")
