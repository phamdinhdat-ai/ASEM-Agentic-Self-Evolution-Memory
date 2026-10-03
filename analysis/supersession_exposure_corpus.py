"""Corpus-wide exposure of mislabelled `superseded_by` notes (all conversations).

Owned by WS2. Runs the exposure measurement from `supersession_exposure.py` over
every conversation in a tag, in ONE process (the retriever is built once per
conversation because each conversation has its own frozen bank).

Refuses to emit a corpus rate if any retrieval call errored.

Usage:
    python analysis/supersession_exposure_corpus.py \
        --root static/memory_banks/locomo10/thg_v2/ASEM-THG \
        --out analysis/_supersession_exposure_corpus.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"), override=False)

from analysis.graph_metrics import load_notes  # noqa: E402
from analysis.supersession_exposure import pairs_from_notes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--config", default="configs/models/deepseek_v41_flash.yaml")
    ap.add_argument("--out", default="analysis/_supersession_exposure_corpus.json")
    args = ap.parse_args()

    banks = sorted(glob.glob(os.path.join(args.root, "*", "asem_thg.sqlite")))
    print(f"banks: {len(banks)}")

    from eval.phase_runner import build_backend_from_config, close_system  # noqa: E402
    from eval.systems import build_system  # noqa: E402
    from asem.note import Note  # noqa: E402

    backend = build_backend_from_config(os.path.join(_ROOT, args.config))

    per_conv: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    tot_pairs = tot_ok = tot_err = tot_exposed = 0

    for bank_path in banks:
        conv = os.path.basename(os.path.dirname(bank_path))
        notes = load_notes(bank_path)
        pairs = pairs_from_notes(notes)
        scratch = os.path.join(_ROOT, "analysis", f"_scratch_exposure_{conv}")
        os.makedirs(scratch, exist_ok=True)
        system = build_system("ASEM-THG", os.path.join(_ROOT, args.config), scratch,
                              backend=backend)
        bank = getattr(system.pipeline, "memory_bank", None)
        retriever = getattr(system.pipeline, "retriever", None)

        for rec in notes:
            try:
                bank.add(Note.from_dict(dict(rec)))
            except Exception:  # noqa: BLE001
                pass

        n_ok = n_err = n_exposed = 0
        for p in pairs:
            a, b = p["a"], p["b"]
            query = (a.get("c") or "")[:300]
            if not query.strip():
                continue
            try:
                used = retriever.retrieve(query, bank)
            except Exception as exc:  # noqa: BLE001
                n_err += 1
                all_rows.append({"conversation": conv, "a": a["id"], "b": b["id"],
                                 "error": f"{type(exc).__name__}: {exc}"})
                continue
            ids = {getattr(n, "id", None) for n in (used or [])}
            hit_a, hit_b = a["id"] in ids, b["id"] in ids
            exposed = hit_a or hit_b
            n_ok += 1
            if exposed:
                n_exposed += 1
            all_rows.append({
                "conversation": conv, "a": a["id"], "b": b["id"],
                "a_session": a.get("session_id"), "b_session": b.get("session_id"),
                "hit_older": hit_a, "hit_newer": hit_b, "n_retrieved": len(ids),
            })

        close_system(system)
        rate = (n_exposed / n_ok) if n_ok else None
        per_conv.append({"conversation": conv, "n_pairs": len(pairs),
                         "n_evaluated": n_ok, "n_errors": n_err,
                         "n_exposed": n_exposed, "exposure_rate": rate})
        print(f"  {conv}: pairs={len(pairs):3d} ok={n_ok:3d} err={n_err:3d} "
              f"exposed={n_exposed:3d} rate={rate if rate is None else round(rate,4)}")
        tot_pairs += len(pairs); tot_ok += n_ok; tot_err += n_err
        tot_exposed += n_exposed

    print(f"\n=== CORPUS ({len(banks)} conversations) ===")
    print(f"  pairs total    : {tot_pairs}")
    print(f"  evaluated ok   : {tot_ok}")
    print(f"  errors         : {tot_err}")
    print(f"  exposed        : {tot_exposed}")

    payload: Dict[str, Any] = {
        "root": args.root, "n_conversations": len(banks),
        "n_pairs_total": tot_pairs, "n_evaluated": tot_ok, "n_errors": tot_err,
        "n_exposed": tot_exposed,
        "measurement": "EXPOSURE PROXY — not the accuracy ablation (that needs task-4's judge pipeline)",
        "per_conversation": per_conv, "rows": all_rows,
    }
    if tot_err:
        payload["exposure_rate"] = None
        payload["refused"] = True
        payload["refusal_reason"] = (
            f"{tot_err} retrieval call(s) raised; a rate over failed calls is not a measurement."
        )
        print("\n" + "!" * 70)
        print(f"REFUSING TO EMIT A CORPUS RATE: {tot_err} retrieval calls failed.")
        print("!" * 70)
    else:
        payload["exposure_rate"] = (tot_exposed / tot_ok) if tot_ok else None
        payload["refused"] = False
        print(f"  EXPOSURE RATE  : {payload['exposure_rate']:.4f} (n={tot_ok})")

    with open(os.path.join(_ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"json -> {args.out}")
    return 1 if tot_err else 0


if __name__ == "__main__":
    sys.exit(main())
