"""Search the FastASEM bank for specific facts to determine whether
remaining QA failures are ingestion misses (fact not in bank) or
retrieval misses (fact in bank but not in top-k)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.systems import build_fast_asem_system  # noqa: E402

# (label, [keywords to search for in c/K/G/X/entities])
SEARCHES = [
    ("Q1 LGBTQ support group @ 7 May 2023", ["support group", "7 May", "LGBTQ support"]),
    ("Q7 camping planned June 2023", ["camping", "June", "planning", "plan to go camp"]),
    ("Q8 relationship status single", ["single", "relationship status", "not in a relationship", "no partner", "unpartnered"]),
]


def main() -> None:
    system = build_fast_asem_system("configs/presets/sota_benchmark.yaml", "data/benchmarks/eval_banks")
    bank = system.pipeline.memory_bank
    notes = bank.list_notes()
    print(f"Bank has {len(notes)} notes\n")

    for label, keywords in SEARCHES:
        print("=" * 70)
        print(f"{label}")
        print("=" * 70)
        hits = 0
        for n in notes:
            hay = " ".join(
                [n.c or "", " ".join(n.K or []), " ".join(n.G or []), n.X or "", " ".join(n.entities or [])]
            ).lower()
            matched = [kw for kw in keywords if kw.lower() in hay]
            if matched:
                hits += 1
                print(f"  [{n.id}] matched={matched}")
                print(f"      c: {n.c}")
                print(f"      date: {n.session_date}  entities: {n.entities}")
        if hits == 0:
            print("  NO MATCHES — fact likely NOT ingested")
        print()


if __name__ == "__main__":
    main()
