"""Print the evidence + locate the source utterances for the failing QA items,
to confirm the gold facts actually exist in the raw conversation (i.e. the
failure is an ingestion/extraction miss, not a bad gold label)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets" / "locomo" / "locomo10.json"

with open(DATA, "r", encoding="utf-8") as f:
    raw = json.load(f)
conv = raw[0]
qa = [q for q in conv.get("qa", []) if q.get("category", 0) != 5]
c = conv["conversation"]

# Build a flat list of (session_idx, speaker, text) for searching.
sessions = []
si = 1
while f"session_{si}" in c:
    for turn in c[f"session_{si}"]:
        sessions.append((si, turn.get("speaker", "?"), turn.get("text", "")))
    si += 1

print(f"Conversation has {si - 1} sessions, {len(sessions)} turns\n")

for idx in (0, 6, 7):  # Q1, Q7, Q8 (0-indexed after category filter)
    if idx >= len(qa):
        continue
    item = qa[idx]
    print("=" * 72)
    print(f"Q{idx + 1}: {item.get('question')}")
    print(f"  GOLD: {item.get('answer')}")
    print(f"  EVIDENCE: {item.get('evidence')}")
    print("-" * 72)
    # Search raw turns for a couple of keywords from the gold answer.
    gold = str(item.get("answer", ""))
    # pick distinctive tokens
    for si, spk, text in sessions:
        low = text.lower()
        # crude relevance: share a meaningful word with the question
        qwords = [w for w in item.get("question", "").lower().split() if len(w) > 4]
        if any(w in low for w in qwords):
            print(f"  [session {si}] {spk}: {text[:200]}")
    print()
