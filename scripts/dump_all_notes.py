"""Dump ALL notes in the conv-1 bank to inspect ingestion quality."""
from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from eval.systems import build_fast_asem_system  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks")
    args = parser.parse_args()

    system = build_fast_asem_system(args.config, args.db_dir)
    bank = system.pipeline.memory_bank
    notes = bank.list_notes()
    print(f"Total notes: {len(notes)}\n")
    for i, n in enumerate(notes, start=1):
        date = n.session_date or (n.t.strftime("%d %B %Y") if n.t else "?")
        ents = n.entities or []
        print(f"[{i:>2}] ({len(ents):>2} ents) [{date}]")
        print(f"      c: {n.c}")
        print(f"      K: {n.K}")
        print(f"      G: {n.G}")
        print(f"      X: {n.X}")
        print(f"      L: {n.L}")
        print()


if __name__ == "__main__":
    main()
