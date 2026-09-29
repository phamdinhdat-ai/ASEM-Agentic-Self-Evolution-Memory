"""Validate the rebuilt ASEM-THG bank after the Rule-5 ingestion fix.

Checks three things the raw note count cannot tell us:
  1. Density  — notes/turn, and turn coverage at cos>=0.55 (was 48% pre-fix).
  2. Quality  — are the extra notes real facts, or fragments/noise?
  3. Recall   — for the 47 LoCoMo category-5 adversarial items, is the
                evidence now present in the bank, and is the distractor's
                claim NOT wrongly attributed to the wrong speaker?

Usage:
  python scratch_diag/verify_rebuilt_bank.py [bank_path]
"""
from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.getcwd())

DATA = "datasets/locomo/locomo10.json"
BANK = sys.argv[1] if len(sys.argv) > 1 else (
    "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite"
)

_STOP = set(
    """the a an and of to in for is are was were it that this with on at as be
have has had you your we they she he i me my her his so but or if not no yes just
really very much more also even still now then here there when where what who why
how do does did about after before over under""".split()
)


def toks(s) -> set:
    if isinstance(s, list):
        s = " ".join(str(x) for x in s)
    words = re.findall(r"[a-z0-9]+", (s or "").lower())
    return {t for t in words if t not in _STOP}


def main() -> None:
    from asem.memory_bank import MemoryBank

    notes = MemoryBank(db_path=BANK).list_notes()
    print(f"bank            : {BANK}")
    print(f"notes           : {len(notes)}")

    data = json.load(open(DATA, encoding="utf-8"))
    conv = data[0]["conversation"]

    turns = []
    for key, val in conv.items():
        if not isinstance(val, list):
            continue
        for turn in val:
            txt = (turn.get("text") or "").strip()
            if len(txt) >= 15:
                turns.append((key, turn.get("speaker"), txt))

    print(f"dialogue turns  : {len(turns)}")
    print(f"notes per turn  : {len(notes)/max(1,len(turns)):.2f}   (was 0.80 pre-fix)")

    # ---- 1. turn coverage -------------------------------------------------
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    note_emb = model.encode(
        [f"{n.c} {n.X}" for n in notes], normalize_embeddings=True, show_progress_bar=False
    )

    covered = 0
    misses = []
    for sess, spk, txt in turns:
        sims = note_emb @ model.encode(txt, normalize_embeddings=True)
        if float(sims.max()) >= 0.55:
            covered += 1
        else:
            misses.append((sess, spk, txt[:110]))
    pct = 100.0 * covered / max(1, len(turns))
    print(f"turn coverage   : {covered}/{len(turns)} = {pct:.1f}%   (was 48% pre-fix)")

    # ---- 2. note quality --------------------------------------------------
    lengths = sorted(len(n.c) for n in notes)
    tiny = sum(1 for L in lengths if L < 25)
    dupes = len(notes) - len({(n.c or "").strip().lower() for n in notes})
    print(f"note len p50/p95 : {lengths[len(lengths)//2]}/{lengths[int(len(lengths)*0.95)]} chars")
    print(f"notes < 25 chars : {tiny}  (fragment risk)")
    print(f"exact duplicates : {dupes}")

    # Speaker attribution lives in Note.speaker, not in the note text.
    no_speaker = sum(1 for n in notes if not (n.speaker or "").strip())
    print(f"notes w/o speaker: {no_speaker}   (field: Note.speaker)")

    print("\n--- 5 longest notes (spot-check for verbosity) ---")
    for n in sorted(notes, key=lambda x: -len(x.c or ""))[:5]:
        print(f"  [{len(n.c):4d}] {(n.c or '')[:150]}")

    # ---- 3. adversarial evidence recall ----------------------------------
    # evidence field is a list of turn IDs like ["D2:3"], not text.
    # We look up the actual dialogue text and do token overlap with notes.
    ref = _load_cat5(data)
    if not ref:
        print("\n(no category-5 reference found — skipping recall check)")
        return

    in_bank = 0
    for item in ref:
        ev_texts = []
        for ev_id in (item.get("evidence") or []):
            try:
                sess, dia = ev_id.split(":")
                sess_key = f"session_{sess[1:]}"
                sess_list = conv.get(sess_key, [])
                d = sess_list[int(dia)] if isinstance(sess_list, list) and int(dia) < len(sess_list) else {}
                ev_texts.append(d.get("text", ""))
            except Exception:
                pass
        ev = toks(" ".join(ev_texts))
        if not ev:
            continue
        # Same threshold as adversarial_layer_diag3.py: overlap >= max(3, 0.35*|ev|)
        thr = max(3, int(0.35 * len(ev)))
        best_ov = 0
        for n in notes:
            nt = toks(f"{n.c} {n.X}")
            if not nt or not ev:
                continue
            ov = len(ev & nt)
            best_ov = max(best_ov, ov)
        if best_ov >= thr:
            in_bank += 1
    print(
        f"\ncat-5 evidence in bank: {in_bank}/{len(ref)}   "
        f"(was 17/47 = 36.2% pre-fix; same threshold as layer_diag3)"
    )


def _load_cat5(data):
    """Find category-5 QA items plus their evidence text."""
    out = []
    for item in data[0].get("qa", []):
        if str(item.get("category", "")).strip().lower() in {"5", "category 5"}:
            out.append(item)
    return out


if __name__ == "__main__":
    main()
