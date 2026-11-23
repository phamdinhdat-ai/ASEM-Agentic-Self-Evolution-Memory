"""Paired CARE comparison with completed old graph and BM25 LoCoMo outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random

from scripts.run_care_locomo import score_locomo


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/locomo/locomo10.json"
CURRENT = ROOT / "build/care/relation_full"
MERGED = ROOT / "build/care/relation_full_merged.json"
OLD = ROOT / "build/kaggle/old-full-locomo-v5-20261106"
BM25 = ROOT / "build/kaggle/audit-20261103/locomo/full/histories"
OUTPUT = ROOT / "build/care/care_previous_comparison.json"
REPORT = ROOT / "analysis/ASEM_CARE_vs_previous_20261112.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reference(qa: dict) -> str:
    if int(qa["category"]) == 5:
        return "Not mentioned"
    answer = str(qa["answer"]).strip()
    return answer.split(";", 1)[0].strip() if int(qa["category"]) == 3 else answer


def load_current() -> tuple[dict, list]:
    merged = read(MERGED)
    if merged["dataset_sha256"] != sha(DATASET) or merged["n_questions"] != 1986:
        raise ValueError("Current full merge identity/coverage mismatch")
    rows = {}
    receipts = []
    for index in range(5):
        path = CURRENT / f"shard-{index}" / f"care_relation_full_s{index}" / "results/report.json"
        report = read(path)
        if report["manifest"]["dataset_sha256"] != sha(DATASET):
            raise ValueError("CARE shard dataset mismatch")
        for row in report["rows"]:
            if row["id"] in rows:
                raise ValueError("Duplicate CARE question")
            rows[row["id"]] = row
        receipts.append({"shard": index, "sha256": sha(path), "n": len(report["rows"])})
    if len(rows) != 1986:
        raise ValueError("Incomplete CARE rows")
    return rows, receipts


def load_old_graph() -> tuple[dict, list]:
    result = defaultdict(dict)
    receipts = []
    for index in range(8):
        path = OLD / f"shard-{index:02d}" / "locomo/download/full"
        rows_path = path / "results/rows.json"
        if not rows_path.exists():
            continue
        protocol, completion = read(path / "protocol.json"), read(path / "completion.json")
        if (protocol["dataset_sha256"] != sha(DATASET)
                or protocol["answer_model"]["repo"] != "Qwen/Qwen3-8B"
                or protocol["answer_model"]["revision"] !=
                "b968826d9c46dd6066d109eabc6255188de91218"
                or not completion["complete"]):
            raise ValueError(f"Old graph shard {index} identity or completion mismatch")
        rows_hash = sha(rows_path)
        for arm in ("hybrid_attributed", "graph_memory"):
            official = completion["official"][arm]
            if not official["completed"] or official["rows_sha256"] != rows_hash:
                raise ValueError(f"Old graph shard {index} official receipt mismatch")
        rows = read(rows_path)
        if len(rows) != completion["completed_pairs"] == completion["expected_pairs"]:
            raise ValueError(f"Old graph shard {index} incomplete")
        for row in rows:
            qid, arm = row["question_id"], row["arm"]
            if arm in result[qid]:
                raise ValueError("Duplicate old graph prediction")
            result[qid][arm] = {"text": row["text"], "status": row["status"],
                                "category": int(row["category"]), "question": row["question"]}
        receipts.append({"shard": index, "rows_sha256": rows_hash, "n": len(rows)})
        del rows
        gc.collect()
    result = {qid: arms for qid, arms in result.items()
              if set(arms) == {"hybrid_attributed", "graph_memory"}}
    return result, receipts


def load_bm25() -> tuple[dict, list]:
    answers = {}
    present = defaultdict(set)
    receipts = []
    for path in sorted(BM25.glob("*/rows.json")):
        rows = read(path)
        receipts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)})
        for row in rows:
            qid, arm = row["question_id"], row["arm"]
            present[qid].add(arm)
            if arm == "bm25":
                answers[qid] = {"text": row["text"], "status": row["status"],
                                "category": int(row["category"]), "question": row["question"]}
        del rows
    required = {"bm25", "hybrid", "deterministic_graph", "ceg", "ceg_no_edges", "full_context"}
    selected = {qid: answers[qid] for qid, arms in present.items()
                if arms == required and qid in answers}
    if len(selected) != 1128:
        raise ValueError(f"Expected 1,128 paired BM25 questions; found {len(selected)}")
    return selected, receipts


def bootstrap(groups: dict[str, list[float]], seed: int = 20261112) -> list[float]:
    keys = list(groups)
    rng = random.Random(seed)
    draws = []
    for _ in range(10_000):
        values = [v for _ in keys for v in groups[rng.choice(keys)]]
        draws.append(sum(values) / len(values))
    draws.sort()
    return [draws[250], draws[9750]]


def compare(arm: str, previous: dict, current: dict, gold: dict) -> dict:
    paired = []
    for qid, old in previous.items():
        if qid not in current:
            continue
        qa = gold[qid]
        category = int(qa["category"])
        if old["category"] != category or old["question"] != qa["question"]:
            raise ValueError(f"Old question metadata mismatch: {qid}")
        old_score = score_locomo(old["text"], str(qa.get("answer", "")), category)
        now = current[qid]["systems"]["bridge_v2"]
        paired.append({"id": qid, "sample_id": current[qid]["sample_id"], "category": category,
                       "reference": reference(qa),
                       "old_answer": old["text"], "old_status": old["status"],
                       "old_score": old_score, "care_answer": now["answer"],
                       "care_score": now["f1"]})
    if not paired:
        raise ValueError("No overlapping questions")
    groups = {"all": paired, "answerable": [r for r in paired if r["category"] != 5],
              "adversarial": [r for r in paired if r["category"] == 5]}
    groups.update({f"category_{i}": [r for r in paired if r["category"] == i]
                   for i in range(1, 6)})
    summary = {}
    for name, rows in groups.items():
        if not rows:
            continue
        deltas = defaultdict(list)
        wins = Counter()
        for row in rows:
            delta = row["care_score"] - row["old_score"]
            deltas[row["sample_id"]].append(delta)
            wins["win" if delta > 1e-10 else "loss" if delta < -1e-10 else "tie"] += 1
        summary[name] = {
            "n": len(rows), "histories": len(deltas),
            "old_mean": sum(r["old_score"] for r in rows) / len(rows),
            "care_mean": sum(r["care_score"] for r in rows) / len(rows),
            "delta": sum(r["care_score"] - r["old_score"] for r in rows) / len(rows),
            "bootstrap_conversation_95": bootstrap(deltas), **wins}
    return {"arm": arm, "summary": summary, "rows": paired}


def main() -> None:
    dataset = read(DATASET)
    gold = {f"{sample['sample_id']}:{i}": qa for sample in dataset
            for i, qa in enumerate(sample["qa"])}
    current, care_receipts = load_current()
    graph, graph_receipts = load_old_graph()
    bm25, bm25_receipts = load_bm25()
    results = {}
    for arm in ("hybrid_attributed", "graph_memory"):
        results[arm] = compare(arm, {qid: row[arm] for qid, row in graph.items()},
                               current, gold)
    results["bm25"] = compare("bm25", bm25, current, gold)
    output = {"protocol": "care-vs-previous-paired-locomo-v1",
              "dataset_sha256": sha(DATASET), "merged_sha256": sha(MERGED),
              "care_receipts": care_receipts, "graph_receipts": graph_receipts,
              "bm25_receipts": bm25_receipts, "comparisons": results,
              "limitations": [
                  "Previous graph covers completed shards only, not full LoCoMo.",
                  "BM25 covers interrupted paired checkpoint subset only.",
                  "Same Qwen3-8B revision and dataset, but prompts, answer contracts, "
                  "retrieval budgets and other pipeline components differ.",
                  "Category 5 abstention phrase differs across older runs.",
                  "No feasible Qwen3-8B FullContext predictions under the earlier evidence budget.",
              ]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# CARE Bridge v2 versus previous LoCoMo methods", "",
             "Paired comparison on exact question IDs, same dataset SHA and same pinned "
             "Qwen3-8B answer-model revision. All old predictions were rescored with "
             "the CARE LoCoMo category-specific scorer; original predictions were not edited.",
             "Only completed old graph shards and the interrupted paired BM25 subset are included.",
             "", "| Previous method | Group | n | Old score | CARE score | Delta | 95% cluster bootstrap | W/L/T |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for arm, result in results.items():
        for group in ("all", "answerable", "adversarial"):
            s = result["summary"][group]
            lo, hi = s["bootstrap_conversation_95"]
            lines.append(f"| {arm} | {group} | {s['n']} | {s['old_mean']:.4f} | "
                         f"{s['care_mean']:.4f} | {s['delta']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] | "
                         f"{s.get('win', 0)}/{s.get('loss', 0)}/{s.get('tie', 0)} |")
    lines += ["", "## Per-category LoCoMo score", "",
              "| Previous method | Category | n | Old | CARE | Delta |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, result in results.items():
        for cat in range(1, 6):
            s = result["summary"][f"category_{cat}"]
            lines.append(f"| {arm} | {cat} | {s['n']} | {s['old_mean']:.4f} | "
                         f"{s['care_mean']:.4f} | {s['delta']:+.4f} |")
    lines += ["", "## Limits", "",
              "- FullContext in the earlier Qwen3-8B CEG run exceeded the 4,096-token "
              "evidence allowance and did not produce usable answers. Its zero score "
              "is a runtime-budget failure, not a quality comparison.",
              "- The earlier Qwen3-4B FullContext/SimRetrieval runs use a different "
              "backbone and protocol, so their reported scores cannot isolate a CARE gain.",
              "- These are public development questions. Old graph and BM25 completion "
              "was interrupted; the overlapping cohorts are not an independent full benchmark.",
              "- The comparison measures whole pipelines. It does not isolate the "
              "contribution of CARE memory, retrieval, answer prompt or refusal phrase.",
              "", f"Machine-readable rows: `{OUTPUT.relative_to(ROOT).as_posix()}`.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "graph_shards": [r["shard"] for r in graph_receipts],
                      "summary": {arm: {group: result["summary"][group] for group in
                                        ("all", "answerable", "adversarial")}
                                  for arm, result in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
