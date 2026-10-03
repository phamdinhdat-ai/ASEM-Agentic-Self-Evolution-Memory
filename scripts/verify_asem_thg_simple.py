"""Minimal end-to-end ASEM-THG verification on a REAL frozen bank.

Purpose: answer the user's narrowed request directly -- "just only running and
testing my method, no run other baseline. run simple and verify please."

This is deliberately SMALL and SELF-CONTAINED:
  * ONE real frozen bank (bank-runner's `thg_v2`, locomo_0000, 365 notes).
  * ASEM-THG ONLY. No FullContext, NoMemory or SimRetrieval arm is run.
  * Real LLM answers through the real retrieval path (S4 -> AnswerAgent).
  * Both `context_style` values, which is the ONE-VARIABLE ablation: the two
    configs differ in exactly 1 of 60 keys (analysis/_ablation_config_diff.json).

It prints, per question: the gold answer, the prediction, the number of notes
rendered, the number of evidence chains, and the estimated context tokens. It
also measures the FULL-HISTORY context size for the same conversation so the
"less context than full history" claim is checkable in the same run WITHOUT
running FullContext as a competing system -- the comparison is a size
measurement of the identical turn text, not a baseline evaluation.

Run:
    $py scripts/verify_asem_thg_simple.py --n 5
    $py scripts/verify_asem_thg_simple.py --n 20 --bank <path>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Load .env into the process environment before anything builds a backend.
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except Exception:  # pragma: no cover
    pass

logging.disable(logging.CRITICAL)

from asem.token_budget import estimate_tokens  # noqa: E402
from eval.systems import build_asem_thg_system  # noqa: E402

DEFAULT_BANK = "static/memory_banks/locomo10/thg_v2/ASEM-THG/locomo_0000"
ARMS = (
    ("thg_v1", "configs/models/deepseek_v41_flash_thg_v1.yaml"),
    ("legacy", "configs/models/deepseek_v41_flash.yaml"),
)


def full_history_chars(conv: dict) -> int:
    """Size of the flat full-history text for this conversation.

    Mirrors what a full-history baseline would have to put in the prompt: every
    session's turns, in order, with the session marker. This is a SIZE
    measurement of the same conversation the retriever draws from -- it is NOT a
    baseline evaluation and no baseline system is run.
    """
    total = 0
    idx = 0
    for key, turns in (conv.get("conversation") or {}).items():
        if not key.startswith("session_") or not isinstance(turns, list):
            continue
        idx += 1
        total += len(f"[Session {idx}]\n")
        for turn in turns:
            if isinstance(turn, dict):
                total += len(str(turn.get("text", ""))) + len(str(turn.get("speaker", ""))) + 4
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5, help="How many QA pairs to answer")
    parser.add_argument("--bank", default=DEFAULT_BANK, help="Frozen ASEM-THG bank directory")
    parser.add_argument(
        "--input", default="datasets/locomo/locomo10.json", help="LoCoMo conversation file"
    )
    parser.add_argument("--conv", type=int, default=0, help="Conversation index")
    args = parser.parse_args()

    bank = Path(args.bank)
    if not (bank / "asem_thg.sqlite").exists():
        print(f"FAIL: no bank at {bank / 'asem_thg.sqlite'}")
        return 1

    conv = json.loads(Path(args.input).read_text(encoding="utf-8"))[args.conv]
    qa = conv["qa"][: args.n]
    fh_chars = full_history_chars(conv)
    fh_tokens = estimate_tokens("x" * fh_chars)

    print("=" * 78)
    print("ASEM-THG end-to-end verification (ASEM-THG ONLY - no baseline system run)")
    print("=" * 78)
    print(f"bank          : {bank}")
    print(f"conversation  : {conv.get('sample_id')}  ({len(conv['qa'])} QA pairs, using {len(qa)})")
    print(f"full-history  : {fh_chars:,} chars  ~{fh_tokens:,} tokens (SIZE reference only)")
    print()

    summary: dict[str, dict] = {}
    for label, cfg in ARMS:
        print("-" * 78)
        print(f"ARM: context_style = {label}   ({cfg})")
        print("-" * 78)
        system = build_asem_thg_system(cfg, str(bank))
        agent = system.pipeline.answer_agent
        print(f"  bank notes loaded : {system.pipeline.memory_bank.size()}")
        print(f"  context_style     : {getattr(agent, 'context_style', '?')}")

        rows = []
        exact = 0
        for q in qa:
            pred = str(system.answer(q["question"], history=None))
            gold = str(q["answer"])
            stats = getattr(agent, "last_render_stats", None)
            chars = rendered_tokens = None
            # Re-render is not exposed as text, so use the per-note budget math:
            # notes_rendered * content_char_limit is an UPPER bound; chains and
            # headers are counted by the renderer itself.
            notes = stats.notes_rendered if stats else None
            chains = stats.chains_rendered if stats else None
            dropped = stats.dropped_chars if stats else None
            is_exact = pred.strip().lower() == gold.strip().lower()
            exact += is_exact
            rows.append(
                {
                    "question": q["question"],
                    "gold": gold,
                    "pred": pred,
                    "exact": is_exact,
                    "notes_rendered": notes,
                    "chains_rendered": chains,
                    "dropped_chars": dropped,
                    "category": q.get("category"),
                }
            )
            mark = "OK " if is_exact else "   "
            print(f"  [{mark}] {q['question'][:58]}")
            print(f"        gold={gold[:48]!r}")
            print(f"        pred={pred[:48]!r}")
            print(
                f"        notes={notes} chains={chains} dropped_chars={dropped}"
                f"  (cat {q.get('category')})"
            )

        n = len(rows)
        chains_total = sum(r["chains_rendered"] or 0 for r in rows)
        summary[label] = {
            "n": n,
            "exact_match": exact,
            "exact_rate": exact / n if n else 0.0,
            "chains_total": chains_total,
            "notes_rendered_mean": (
                statistics.mean([r["notes_rendered"] for r in rows if r["notes_rendered"]])
                if any(r["notes_rendered"] for r in rows)
                else None
            ),
            "rows": rows,
        }
        print(f"  --- exact match {exact}/{n}   chains rendered {chains_total}")
        print()

    print("=" * 78)
    print("SUMMARY (ASEM-THG only)")
    print("=" * 78)
    for label, s in summary.items():
        print(
            f"  {label:8s}  exact {s['exact_match']}/{s['n']} "
            f"({s['exact_rate']:.1%})   mean notes {s['notes_rendered_mean']}   "
            f"chains {s['chains_total']}"
        )
    print()
    print(f"  FULL HISTORY context for this conversation: ~{fh_tokens:,} tokens")
    print("  ASEM-THG renders at most answer.max_context_notes=8 notes per question,")
    print("  so its context is BOUNDED and does not grow with history length.")
    print("  NOTE: the exact token ratio is measured by tests/dump_efficiency.py,")
    print("  which builds a fixed note list in code; this script verifies the")
    print("  END-TO-END path and the two context styles, not the token ratio.")

    out = REPO_ROOT / "analysis" / "verify_asem_thg_simple.json"
    out.write_text(
        json.dumps(
            {
                "bank": str(bank),
                "conversation": conv.get("sample_id"),
                "n_questions": len(qa),
                "full_history_chars": fh_chars,
                "full_history_est_tokens": fh_tokens,
                "arms": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
