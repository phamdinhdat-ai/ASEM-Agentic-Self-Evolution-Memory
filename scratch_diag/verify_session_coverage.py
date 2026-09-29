"""Verify: how many real sessions exist, and how many were actually ingested?"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.getcwd())
from collections import Counter

DATA = "datasets/locomo/locomo10.json"
BANK = "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite"

data = json.load(open(DATA, encoding="utf-8"))
conv = data[0]["conversation"]

keys = list(conv.keys())
sess_keys = sorted([k for k in keys if re.fullmatch(r"session_\d+", k)],
                   key=lambda k: int(k.split("_")[1]))
other = [k for k in keys if k not in sess_keys]

print(f"Total conversation keys: {len(keys)}")
print(f"REAL session keys      : {len(sess_keys)}  -> {sess_keys}")
print(f"Non-session keys       : {len(other)}  -> {other}")

total_turns = 0
for k in sess_keys:
    v = conv[k]
    if isinstance(v, list):
        total_turns += len(v)
print(f"Total turns in sessions: {total_turns}")

from asem.memory_bank import MemoryBank
mb = MemoryBank(db_path=BANK)
notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()
ing = sorted({n.session_id for n in notes}, key=lambda s: int(s.lstrip("s")) if s.lstrip("s").isdigit() else 0)
print(f"\nSessions present in bank: {len(ing)} -> {ing}")
print(f"MISSING sessions        : {[k for k in sess_keys if k.replace('session_','s') not in ing]}")

# per-session notes vs turns
print("\n  session   turns  notes   notes/turn")
for k in sess_keys:
    sid = k.replace("session_", "s")
    nt = len(conv[k]) if isinstance(conv[k], list) else 0
    nn = sum(1 for n in notes if n.session_id == sid)
    flag = "  <-- EMPTY" if nn == 0 else ""
    print(f"  {k:10s} {nt:4d}  {nn:4d}   {nn/max(1,nt):.2f}{flag}")
