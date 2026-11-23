"""Freeze a blinded, stratified sample for claim-level CARE source auditing.

The output is an annotation queue, not a measured entailment result. Each
prediction must be judged using only its cited original turns. Gold answers,
benchmark scores, and gold evidence IDs are kept in a separate key file.
"""

from __future__ import annotations

from collections import Counter
import csv
import glob
import hashlib
import json
from pathlib import Path

from scripts.run_care_locomo import turns_for


SEED = "care-source-support-audit-v1"
PER_CELL = 10


def is_abstention(answer: str) -> bool:
    normalized = answer.casefold()
    return "no information available" in normalized or "not mentioned" in normalized


def main() -> None:
    reports = sorted(glob.glob("build/care/relation_full/**/report.json", recursive=True))
    if len(reports) != 5:
        raise ValueError(f"Expected five complete CARE reports, found {len(reports)}")
    rows = [row for path in reports for row in json.loads(Path(path).read_text(encoding="utf-8"))["rows"]]
    if len(rows) != 1986 or len({row["id"] for row in rows}) != 1986:
        raise ValueError("Expected 1,986 distinct LoCoMo predictions")

    dataset = json.loads(Path("datasets/locomo/locomo10.json").read_text(encoding="utf-8"))
    samples = {sample["sample_id"]: sample for sample in dataset}
    turns = {sid: {turn.source_id: turn for turn in turns_for(sample["conversation"])}
             for sid, sample in samples.items()}

    queue = []
    key = []
    counts = Counter()
    for category in range(1, 6):
        for escalated in (False, True):
            candidates = [row for row in rows
                          if row["category"] == category
                          and bool(row["systems"]["bridge_v2"].get("escalated")) == escalated
                          and not is_abstention(row["systems"]["bridge_v2"]["answer"])]
            candidates.sort(key=lambda row: hashlib.sha256(
                f"{SEED}:{row['id']}".encode("utf-8")).hexdigest())
            if len(candidates) < PER_CELL:
                raise ValueError(f"Insufficient non-abstentions for category={category}, repair={escalated}")
            for row in candidates[:PER_CELL]:
                result = row["systems"]["bridge_v2"]
                sid = row["sample_id"]
                cited = list(dict.fromkeys(result["source_ids"]))
                if not cited or any(source not in turns[sid] for source in cited):
                    raise ValueError(f"Invalid citation in {row['id']}")
                if not set(cited) <= set(result["bundle_ids"]):
                    raise ValueError(f"Citation was not visible in {row['id']}")
                audit_id = hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest()[:12]
                queue.append({
                    "audit_id": audit_id,
                    "question": row["question"],
                    "answer": result["answer"],
                    "cited_turns": [{"source_id": source,
                                     "session_date": turns[sid][source].date,
                                     "speaker": turns[sid][source].speaker,
                                     "text": turns[sid][source].text}
                                    for source in cited],
                    "claim_support": "",  # supported / partial / unsupported / uncertain
                    "speaker_time_consistent": "",  # yes / no / uncertain
                    "annotator_note": "",
                })
                qa = samples[sid]["qa"][int(row["id"].split(":")[-1])]
                key.append({"audit_id": audit_id, "question_id": row["id"],
                            "sample_id": sid, "category": category,
                            "escalated": escalated, "benchmark_score": result["f1"],
                            "reference_answer": qa.get("answer", ""),
                            "gold_evidence_ids": qa.get("evidence", [])})
                counts[f"category_{category}_{'repair' if escalated else 'one_pass'}"] += 1
    queue.sort(key=lambda item: item["audit_id"])
    key.sort(key=lambda item: item["audit_id"])
    if len(queue) != 100 or len({item["audit_id"] for item in queue}) != 100:
        raise ValueError("Audit queue must contain 100 distinct answers")
    output = Path("analysis")
    (output / "CARE_source_support_annotation_queue_20261114.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "CARE_source_support_annotation_queue_20261114.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "audit_id", "question", "answer", "cited_turns", "claim_support",
            "speaker_time_consistent", "annotator_note"))
        writer.writeheader()
        for item in queue:
            writer.writerow({**item, "cited_turns": "\n\n".join(
                f"[{turn['source_id']}] {turn['session_date']} | {turn['speaker']}: {turn['text']}"
                for turn in item["cited_turns"])})
    (output / "CARE_source_support_annotation_key_20261114.json").write_text(
        json.dumps({"protocol": SEED, "selection": dict(counts), "rows": key},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n": len(queue), "selection": dict(counts)}, indent=2))


if __name__ == "__main__":
    main()
