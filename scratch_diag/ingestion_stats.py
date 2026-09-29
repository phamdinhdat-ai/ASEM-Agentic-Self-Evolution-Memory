"""Check ingestion stats: how many sessions used fallback vs LLM extraction."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.getcwd())

BANK = "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite"
from asem.memory_bank import MemoryBank
mb = MemoryBank(db_path=BANK)
notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()

# Group by session_id
from collections import Counter
sess_counts = Counter(n.session_id for n in notes)
print(f"Total notes: {len(notes)}")
print(f"Sessions with notes: {len(sess_counts)}")
print(f"\nNotes per session:")
for sid, cnt in sorted(sess_counts.items(), key=lambda x: x[0]):
    print(f"  {sid}: {cnt}")

# Check speaker distribution
spk = Counter(n.speaker for n in notes)
print(f"\nSpeaker distribution: {dict(spk)}")

# Check how many notes have empty speaker
no_speaker = sum(1 for n in notes if not n.speaker)
print(f"Notes with no speaker: {no_speaker}")
