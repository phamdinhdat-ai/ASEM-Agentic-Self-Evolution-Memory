"""Exposure of mislabelled `superseded_by` notes to the retrieval context.

Owned by WS2. Measures the EXPOSURE proxy for the `superseded_by` defect:

For each `superseded_by` pair, build a query from the pair's own content (using
the older note's fact text, which is what a question about that fact would look
like) and ask the system's retriever which notes it would surface. Count whether
either endpoint of the pair reaches the retrieved set.

What this IS:     how often the mislabelled notes actually reach the context.
What this is NOT: the accuracy ablation (weight 1.3 vs disabled). That needs the
                  judge pipeline (task-4). This script does not score answers.

Usage:
    python analysis/supersession_exposure.py \
        --bank static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000/asem_thg.sqlite \
        --config configs/models/deepseek_v41_flash.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"), override=False)

from analysis.graph_metrics import load_notes  # noqa: E402


def pairs_from_notes(notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {n["id"]: n for n in notes}
    seen: Set[tuple] = set()
    out: List[Dict[str, Any]] = []
    for nid, rec in by_id.items():
        for link in rec.get("L") or []:
            if isinstance(link, dict):
                rel = str(link.get("relation") or "")
                tgt = str(link.get("target_id") or "")
            else:
                rel, tgt = "", ""
            if rel != "superseded_by" or not tgt or tgt not in by_id:
                continue
            key = tuple(sorted((nid, tgt)))
            if key in seen:
                continue
            seen.add(key)
            out.append({"a": by_id[key[0]], "b": by_id[key[1]]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--config", default="configs/models/deepseek_v41_flash.yaml")
    ap.add_argument("--k", type=int, default=10, help="top-k retrieved notes to inspect")
    ap.add_argument("--out", default="analysis/_supersession_exposure.json")
    args = ap.parse_args()

    notes = load_notes(args.bank)
    pairs = pairs_from_notes(notes)
    print(f"bank : {args.bank}")
    print(f"notes: {len(notes)}   superseded_by pairs: {len(pairs)}")

    # Load the real retriever through the existing system builder (no edits).
    from eval.phase_runner import build_backend_from_config, close_system  # noqa: E402
    from eval.systems import build_system  # noqa: E402

    backend = build_backend_from_config(os.path.join(_ROOT, args.config))
    scratch = os.path.join(_ROOT, "analysis", "_scratch_bank_exposure")
    os.makedirs(scratch, exist_ok=True)
    system = build_system("ASEM-THG", os.path.join(_ROOT, args.config), scratch, backend=backend)
    bank = getattr(system.pipeline, "memory_bank", None)
    retriever = getattr(system.pipeline, "retriever", None)
    if bank is None or retriever is None:
        print("!! could not reach bank/retriever")
        close_system(system)
        return 1

    # Load the frozen notes into the in-memory bank by reading the source bank.
    from asem.note import Note  # noqa: E402
    loaded = 0
    for rec in notes:
        try:
            n = Note.from_dict({**rec, "e": rec.get("e"), "z": rec.get("z")})
            bank.add(n)
            loaded += 1
        except Exception:  # noqa: BLE001
            continue
    print(f"loaded {loaded} notes into the retriever's bank")

    rows: List[Dict[str, Any]] = []
    n_exposed = 0
    n_errors = 0
    for p in pairs:
        a, b = p["a"], p["b"]
        # The query a question about this fact would look like.
        query = (a.get("c") or "")[:300]
        if not query.strip():
            continue
        try:
            used = retriever.retrieve(query, bank)
        except Exception as exc:  # noqa: BLE001
            n_errors += 1
            rows.append({"a": a["id"], "b": b["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        ids = {getattr(n, "id", None) for n in (used or [])}
        hit_a, hit_b = a["id"] in ids, b["id"] in ids
        exposed = hit_a or hit_b
        if exposed:
            n_exposed += 1
        rows.append({
            "a": a["id"], "b": b["id"],
            "a_session": a.get("session_id"), "b_session": b.get("session_id"),
            "a_sentence": a.get("c"), "b_sentence": b.get("c"),
            "hit_older": hit_a, "hit_newer": hit_b, "n_retrieved": len(ids),
        })

    n_ok = len(rows) - n_errors
    print(f"\npairs evaluated successfully : {n_ok}")
    print(f"pairs that ERRORED            : {n_errors}")

    if n_errors:
        # A rate computed over rows that errored is NOT a rate. Refuse to emit
        # one: a warning beside a plausible number loses to the number, so the
        # only safe behaviour is to withhold the number entirely.
        print("\n" + "!" * 70)
        print(f"REFUSING TO EMIT A RATE: {n_errors} of {len(rows)} retrieval calls failed.")
        print("A rate over failed calls is not a measurement. Fix the errors first.")
        print("!" * 70)
        with open(os.path.join(_ROOT, args.out), "w", encoding="utf-8") as fh:
            json.dump({
                "bank": args.bank, "n_notes": len(notes), "n_pairs": len(pairs),
                "n_evaluated": n_ok, "n_errors": n_errors,
                "n_exposed": n_exposed,
                "exposure_rate": None,
                "refused": True,
                "refusal_reason": (
                    f"{n_errors} retrieval call(s) raised; a rate over failed calls "
                    "is not a measurement."
                ),
                "measurement": "EXPOSURE PROXY — not the accuracy ablation",
                "rows": rows,
            }, fh, indent=2)
        close_system(system)
        return 1

    print(f"pairs with >=1 endpoint retrieved (EXPOSED): {n_exposed}")
    print(f"exposure rate   : {n_exposed/n_ok if n_ok else float('nan'):.4f}  (n={n_ok})")
    print("\nNOTE: this is EXPOSURE, not the accuracy ablation. The ablation needs")
    print("      the judge pipeline (task-4). Do not quote this as an accuracy delta.")

    with open(os.path.join(_ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump({
            "bank": args.bank, "n_notes": len(notes), "n_pairs": len(pairs),
            "n_evaluated": n_ok, "n_errors": n_errors,
            "n_exposed": n_exposed,
            "exposure_rate": (n_exposed / n_ok) if n_ok else None,
            "refused": False,
            "measurement": "EXPOSURE PROXY — not the accuracy ablation",
            "rows": rows,
        }, fh, indent=2)
    close_system(system)
    return 0
    print(f"json -> {args.out}")
    close_system(system)
    return 0


if __name__ == "__main__":
    sys.exit(main())
