"""Corrected layer diagnosis. Note fields are the SHORT names: c (content),
X (description), K (keywords), G (tags), L (links), speaker, session_date.
"""
from __future__ import annotations
import json, os, re, sys
sys.path.insert(0, os.getcwd())

DATA = "datasets/locomo/locomo10.json"
PRED = "data/benchmarks/results/static/locomo10/preds/ds_thg__deepseek_v4_flash__ASEM-THG.jsonl"
BANK = "static/memory_banks/locomo10/ds_thg/ASEM-THG/locomo_0000/asem_thg.sqlite"

_REFUSAL_RE = re.compile(
    r"not mentioned|no information|i don'?t know|i do not know|"
    r"cannot (?:be )?(?:answer|determine|find|tell)|does ?n[o']t (?:mention|say|state)|"
    r"no memory|nothing (?:in the|is )", re.I)
_STOP = set("""the a an and of to in for is are was were it that this with on at as be
have has had you your we they she he i me my her his so but or if not no yes just really
very much more also even still now then here there when where what who why how do does did
about after before over under""".split())


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def content(s: str) -> set:
    return {t for t in norm(s).split() if t not in _STOP}


def main() -> None:
    data = json.load(open(DATA, encoding="utf-8"))
    conv = data[0]["conversation"]
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8") if l.strip()]
    cat5 = [r for r in rows if r.get("category") == 5]

    from asem.memory_bank import MemoryBank
    mb = MemoryBank(db_path=BANK)
    notes = mb.list_notes() if hasattr(mb, "list_notes") else mb.get_all_notes()
    print(f"notes in bank: {len(notes)}")

    # ---- L1: for each cat-5 question, is the *evidence* in the bank, and
    #         does the bank attribute it to the same speaker as the dialogue?
    # ---- L3: did the model refuse?
    ev_present = ev_attributed_right = 0
    refused = trapped = 0
    l1_ok_l3_bad = []      # bank has it, model still trapped -> PROMPT/REASONING bug
    l1_bad = []            # bank missing it -> INGESTION bug

    for r in cat5:
        idx = r["idx"]; q = data[0]["qa"][idx]
        pred = r.get("pred") or ""
        ref = bool(_REFUSAL_RE.search(pred))

        # Evidence dialogues
        ev_texts, ev_speaker = [], None
        for ev in q.get("evidence", []):
            sess, dia = ev.split(":")
            sess_key = f"session_{sess[1:]}"
            sess_list = conv.get(sess_key, [])
            d = sess_list[int(dia)] if isinstance(sess_list, list) and int(dia) < len(sess_list) else {}
            ev_texts.append(d.get("text", ""))
            if ev_speaker is None:
                ev_speaker = d.get("speaker")
        ev_tok = content(" ".join(ev_texts))

        # Best-matching note
        best, best_ov = None, 0
        for n in notes:
            nt = content(f"{n.c} {n.X}")
            if not ev_tok:
                continue
            ov = len(ev_tok & nt)
            if ov > best_ov:
                best, best_ov = n, ov
        thr = max(3, int(0.35 * len(ev_tok)))
        in_bank = best is not None and best_ov >= thr
        # Attribution: the note's speaker should match the dialogue speaker
        attrib_ok = in_bank and (ev_speaker is None or best.speaker == ev_speaker)

        if in_bank:
            ev_present += 1
        if attrib_ok:
            ev_attributed_right += 1
        if ref:
            refused += 1
        else:
            trapped += 1
        if in_bank and not ref:
            l1_ok_l3_bad.append((idx, q["question"], best, best_ov, len(ev_tok), pred))
        if not in_bank:
            l1_bad.append((idx, q["question"], " ".join(ev_texts)[:90], ev_speaker))

    n = len(cat5)
    print("\n" + "=" * 96)
    print("LAYER DIAGNOSIS (corrected) — 47 adversarial questions")
    print("=" * 96)
    print(f"L1 INGESTION  evidence fact present in bank        : {ev_present}/{n}")
    print(f"L1 ATTRIB     note speaker == dialogue speaker    : {ev_attributed_right}/{n}")
    print(f"L3 ANSWERING  refused (correct under official)     : {refused}/{n}")
    print(f"L3 ANSWERING  fell into the trap                  : {trapped}/{n}")
    print(f"\nOfficial-protocol accuracy                        : {refused/n:.4f}")
    print(f"Harness EM (scored vs distractor)                 : 0.0000")

    print(f"\n--- INGESTION LOSS: evidence not in bank ({len(l1_bad)}) ---")
    for idx, qq, ev, sp in l1_bad:
        print(f"  [idx {idx:3d}] {qq[:62]}")
        print(f"      dialogue({sp}): {ev[:85]}")

    print(f"\n--- REASONING LOSS: fact WAS in bank, model still trapped ({len(l1_ok_l3_bad)}) ---")
    for idx, qq, best, ov, tot, pred in l1_ok_l3_bad:
        print(f"  [idx {idx:3d}] {qq[:62]}")
        print(f"      note found (speaker={best.speaker}, overlap {ov}/{tot}): {best.c[:105]}")
        print(f"      predicted: {pred[:120]}")


if __name__ == "__main__":
    main()
