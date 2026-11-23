"""Print source turns cited by CARE-Bridge on adversarial development cases."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_care_locomo import turns_for


def main():
    report = json.loads(Path("build/care/kaggle_bridge_dev_report.json").read_text(encoding="utf-8"))
    data = json.loads(Path("build/care/locomo10.json").read_text(encoding="utf-8"))
    conversations = {conversation["sample_id"]: conversation for conversation in data}
    for row in report["rows"]:
        if row["category"] != 5:
            continue
        sources = {turn.source_id: turn for turn in
                   turns_for(conversations[row["sample_id"]]["conversation"])}
        print("\n", row["id"], row["question"])
        for arm in ("proof_bm25", "bridge", "care_v1"):
            result = row["systems"][arm]
            print(arm, result["answer"], result["source_ids"],
                  "first_pass", result.get("first_pass"))
            for sid in result["source_ids"]:
                turn = sources.get(sid)
                if turn:
                    print("  ", sid, turn.date, turn.speaker, turn.text[:500])


if __name__ == "__main__":
    main()
