"""Audit the ds_thg / ASEM-THG banks: notes, edges, empty fields, provenance."""

from __future__ import annotations

import glob
import json
import os
import sqlite3

ROOT = "static/memory_banks/locomo10/ds_thg/ASEM-THG"


def main() -> None:
    files = sorted(glob.glob(os.path.join(ROOT, "*", "*.sqlite")))
    conn = sqlite3.connect(files[0])
    cols = [row[1] for row in conn.execute("PRAGMA table_info(notes)")]
    conn.close()
    print(f"notes columns: {cols}\n")

    tot = {"n": 0, "edges": 0, "no_entities": 0, "no_speaker": 0, "no_kw": 0}
    print(f"{'conv':<14}{'notes':>7}{'edges':>8}{'no_ent':>8}{'no_spk':>8}")
    for path in files:
        conn = sqlite3.connect(path)
        rows = conn.execute(
            "select K, entities, speaker, L from notes"
        ).fetchall()
        conn.close()

        n = len(rows)
        no_e = no_s = no_k = edges = 0
        for kw, ents, spk, links in rows:
            if not json.loads(ents or "[]"):
                no_e += 1
            if not spk:
                no_s += 1
            if not json.loads(kw or "[]"):
                no_k += 1
            edges += len(json.loads(links or "[]"))
        tot["n"] += n
        tot["edges"] += edges
        tot["no_entities"] += no_e
        tot["no_speaker"] += no_s
        tot["no_kw"] += no_k
        print(f"{os.path.basename(os.path.dirname(path)):<14}{n:>7}{edges:>8}"
              f"{no_e:>8}{no_s:>8}")

    n = max(tot["n"], 1)
    print(f"\nTOTAL notes={tot['n']} edges={tot['edges']}")
    print(f"  empty entities {tot['no_entities']} ({100 * tot['no_entities'] / n:.1f}%)")
    print(f"  empty speaker  {tot['no_speaker']} ({100 * tot['no_speaker'] / n:.1f}%)")
    print(f"  empty keywords {tot['no_kw']} ({100 * tot['no_kw'] / n:.1f}%)")


if __name__ == "__main__":
    main()
