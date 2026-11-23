"""Validate and merge full Qwen3-8B SimRetrieval LoCoMo shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.run_care_full_locomo import SHARDS, selected_questions
from scripts.run_care_locomo import _save_json, score_locomo
from scripts.run_care_sim_retrieval import (ARM, EMBEDDER, EVIDENCE_CHARS,
                                            MAX_ANSWER_TOKENS, MODEL, PROMPT,
                                            SAFETY_TOKENS, TOP_K, WINDOW, summarize)


def merge(dataset_path=Path("datasets/locomo/locomo10.json"),
          root=Path("build/care/sim_retrieval")):
    data_bytes = dataset_path.read_bytes()
    data = json.loads(data_bytes)
    data_sha = hashlib.sha256(data_bytes).hexdigest()
    rows, receipts, seen = [], [], set()
    for index in range(5):
        base = root / f"shard-{index}" / f"care_sim_retrieval_s{index}"
        report_path, status_path = base / "results/report.json", base / "status.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        manifest = report["manifest"]
        expected = selected_questions(data, index)
        if (status["state"] != "complete"
                or manifest["protocol"] != "care-sim-retrieval-locomo-v1"
                or manifest["dataset_sha256"] != data_sha
                or manifest["model"] != MODEL or manifest["embedder"] != EMBEDDER
                or manifest["shard_index"] != index
                or manifest["shard_samples"] != list(SHARDS[index])
                or manifest["question_limit"] != 0
                or manifest["question_ids"] != [q["id"] for q in expected]
                or manifest["arm"] != ARM or manifest["top_k"] != TOP_K
                or manifest["evidence_chars"] != EVIDENCE_CHARS
                or manifest["window_tokens"] != WINDOW
                or manifest["max_answer_tokens"] != MAX_ANSWER_TOKENS
                or manifest["safety_tokens"] != SAFETY_TOKENS
                or manifest["prompt_sha256"] != hashlib.sha256(PROMPT.encode()).hexdigest()
                or len(report["rows"]) != len(expected)):
            raise ValueError(f"Retrieval shard {index} identity/coverage mismatch")
        for row, q in zip(report["rows"], expected):
            if (row["id"] != q["id"] or row["sample_id"] != q["sample_id"]
                    or row["category"] != q["category"]
                    or row["question"] != q["question"]
                    or set(row["systems"]) != {ARM} or row["id"] in seen):
                raise ValueError(f"Retrieval question mismatch {q['id']}")
            seen.add(row["id"])
            actual = score_locomo(row["systems"][ARM]["answer"], q["answer"], q["category"])
            if abs(actual - row["systems"][ARM]["score"]) > 1e-10:
                raise ValueError(f"Retrieval score mismatch {q['id']}")
        rows.extend(report["rows"])
        receipts.append({"index": index, "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                         "n": len(report["rows"]), "seconds": status["seconds"]})
    if len(rows) != 1986 or len(seen) != 1986:
        raise ValueError("Retrieval full coverage mismatch")
    care_rows = {}
    for index in range(5):
        path = (Path("build/care/relation_full") / f"shard-{index}" /
                f"care_relation_full_s{index}" / "results/report.json")
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]:
            care_rows[row["id"]] = row
    if set(care_rows) != seen:
        raise ValueError("CARE and SimRetrieval question IDs differ")
    paired = {}
    for name, subset in (("all", rows),
                         ("answerable", [r for r in rows if r["category"] != 5]),
                         ("adversarial", [r for r in rows if r["category"] == 5])):
        deltas = [care_rows[r["id"]]["systems"]["bridge_v2"]["f1"] -
                  r["systems"][ARM]["score"] for r in subset]
        paired[name] = {"n": len(subset), "care_mean": sum(
            care_rows[r["id"]]["systems"]["bridge_v2"]["f1"] for r in subset) / len(subset),
            "baseline_mean": sum(r["systems"][ARM]["score"] for r in subset) / len(subset),
            "mean_delta": sum(deltas) / len(deltas),
            "wins": sum(d > 1e-10 for d in deltas),
            "losses": sum(d < -1e-10 for d in deltas),
            "ties": sum(abs(d) <= 1e-10 for d in deltas)}
    return {"protocol": "care-sim-retrieval-merge-v1",
            "dataset_sha256": data_sha, "model": MODEL, "embedder": EMBEDDER,
            "n_questions": len(rows), "n_predictions": len(rows),
            "shards": receipts, "aggregate": {ARM: summarize(rows)},
            "by_category": {str(cat): summarize([r for r in rows if r["category"] == cat])
                            for cat in range(1, 6)},
            "paired_vs_care": paired, "rows": rows}


def main():
    result = merge()
    path = Path("build/care/sim_retrieval_merged.json")
    _save_json(path, result)
    print(json.dumps({"n_questions": result["n_questions"],
                      "aggregate": result["aggregate"],
                      "paired_vs_care": result["paired_vs_care"]}, indent=2))


if __name__ == "__main__":
    main()
