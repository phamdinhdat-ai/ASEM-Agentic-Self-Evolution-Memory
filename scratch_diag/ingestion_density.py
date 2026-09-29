"""Quantify ingestion density: notes created vs dialogue turns available.
Also test whether the masked-query channel recovers the evidence note.
"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from sentence_transformers import SentenceTransformer

DATA = "datasets/locomo/locomo10.json"
BANK = "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite"
_STOP = set("""the a an and of to in for is are was were it that this with on at as be
have has had you your we they she he i me my her his so but or if not no yes just really
very much more also even still now then here there when where what who why how do does did
about after before over under""".split())

def toks(s): return {t for t in " ".join(re.findall(r"[a-z0-9]+", (s or "").lower())).split() if t not in _STOP}


def main():
    data = json.load(open(DATA, encoding="utf-8"))
    conv = data[0]["conversation"]

    total_turns = 0
    for k, v in conv.items():
        if isinstance(v, list):
            total_turns += len(v)
    print(f"Dialogue sessions: {len(conv)}")
    print(f"Total dialogue turns: {total_turns}")

    from asem.memory_bank import MemoryBank
    mb = MemoryBank(db_path=BANK)
    notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()
    print(f"Notes created by ingestion: {len(notes)}")
    print(f"Notes per turn: {len(notes)/max(1,total_turns):.2f}")

    # Coverage: how many dialogue turns have their content covered by a note?
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    note_emb = model.encode([f"{n.c} {n.X}" for n in notes], normalize_embeddings=True,
                            show_progress_bar=False)

    covered = 0
    uncovered_examples = []
    turn_texts = []
    for k, v in conv.items():
        if not isinstance(v, list):
            continue
        for turn in v:
            txt = (turn.get("text") or "").strip()
            if len(txt) < 15:
                continue
            turn_texts.append((k, turn.get("speaker"), txt))

    for sess, spk, txt in turn_texts:
        e = model.encode(txt, normalize_embeddings=True)
        sims = note_emb @ e
        if float(sims.max()) >= 0.55:
            covered += 1
        elif len(uncovered_examples) < 25:
            uncovered_examples.append((sess, spk, txt, float(sims.max())))

    print(f"\nTurns with a note at cos>=0.55: {covered}/{len(turn_texts)} "
          f"({covered/max(1,len(turn_texts)):.1%})")
    print(f"\n--- Sample UNCOVERED turns (closest note below threshold) ---")
    for sess, spk, txt, s in uncovered_examples[:20]:
        print(f"  [{sess}] {spk}: {txt[:95]}   (best cos={s:.3f})")


if __name__ == "__main__":
    main()
