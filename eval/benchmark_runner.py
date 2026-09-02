"""Unified Benchmark Runner for Long-Term Memory (LoCoMo, LoCoMo10, LongMemEval).

Usage:
    # Run Fast-ASEM on LoCoMo conversation 1
    python eval/benchmark_runner.py --data datasets/locomo/locomo_conversation_1.json --config configs/presets/fast_eval.yaml --systems FastASEM

    # Run comparison on LoCoMo10
    python eval/benchmark_runner.py --data datasets/locomo/locomo10.json --config configs/presets/sota_benchmark.yaml --systems FastASEM NoMemory FullContext

    # Run with LLM-as-a-Judge
    python eval/benchmark_runner.py --data datasets/locomo/locomo10.json --config configs/presets/sota_benchmark.yaml --systems FastASEM --judge
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_dotenv = os.path.join(_PROJECT_ROOT, ".env")
if os.path.exists(_dotenv):
    try:
        from dotenv import load_dotenv
        load_dotenv(_dotenv, override=False)
    except ImportError:
        with open(_dotenv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from eval.systems import build_baselines, build_fast_asem_system, build_asem_system, build_asem_v2_system
from asem.backends import build_backend
from asem.config import ASEMConfig

CATEGORY_NAMES = {
    1: "Single-Hop",
    2: "Temporal Reasoning",
    3: "Multi-Hop / Commonsense",
    4: "Conversational Context",
    5: "Adversarial",
}


def normalize_text(text: Any) -> str:
    """Normalize text for exact match scoring."""
    # Coerce non-string gold answers (e.g. int years/counts in LoCoMo) to str.
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    t = text.lower().strip()
    # Remove articles
    t = re.sub(r"\b(a|an|the)\b", " ", t)
    # Remove punctuation
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse whitespace
    return " ".join(t.split())


def compute_em(pred: str, ref: str) -> float:
    """Check if prediction matches ground truth or contains key gold entities."""
    norm_p = normalize_text(pred)
    norm_r = normalize_text(ref)
    if not norm_p or not norm_r:
        return 0.0
    if norm_p == norm_r:
        return 1.0
    # Also check substring match if gold answer is fully present
    if norm_r in norm_p or norm_p in norm_r:
        return 1.0
    return 0.0


def compute_rouge_l(pred: str, ref: str) -> float:
    """Compute lightweight ROUGE-L LCS score."""
    p_tokens = normalize_text(pred).split()
    r_tokens = normalize_text(ref).split()
    if not p_tokens or not r_tokens:
        return 0.0

    # LCS dynamic programming
    m, n = len(p_tokens), len(r_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == r_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return (2 * prec * rec) / (prec + rec)


def compute_bertscore_batch(preds: List[str], refs: List[str]) -> List[float]:
    """Compute BERTScore-F1 for a batch of (pred, ref) pairs.

    Uses roberta-base on CPU (lazy import so the heavy dependency is only
    loaded when BERTScore is actually requested). Returns a list of F1 scores
    aligned to ``preds``; falls back to 0.0 for the whole batch on failure.
    """
    if not preds:
        return []
    try:
        from bert_score import score as _bs_score
        _P, _R, F = _bs_score(
            preds, refs, model_type="roberta-base", device="cpu", verbose=False,
        )
        return [float(x) for x in F]
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] BERTScore failed: {e}")
        return [0.0] * len(preds)


def evaluate_with_judge(
    backend,
    question: str,
    gold: str,
    pred: str,
) -> Tuple[bool, str]:
    """LLM-as-a-Judge evaluation following LoCoMo accuracy prompt."""
    prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You are given:
(1) Question: {question}
(2) Gold Answer: {gold}
(3) Generated Answer: {pred}

Grading rules:
- Be generous: as long as the generated answer touches on the same core fact, topic, entity, or date period as the gold answer, count it as CORRECT.
- For time/date questions, if it refers to the same date, month, or timeframe, consider it CORRECT.
- If it contradicts the gold answer or is completely off-topic, mark WRONG.

Return JSON with "label" ("CORRECT" or "WRONG") and "reasoning" (one sentence).
"""
    try:
        raw = backend.generate(prompt)
        from asem.note import _try_extract_json
        data = _try_extract_json(raw, expect_array=False)
        if isinstance(data, dict):
            label = str(data.get("label", "WRONG")).upper()
            return label == "CORRECT", str(data.get("reasoning", ""))
        is_corr = "CORRECT" in raw.upper() and "WRONG" not in raw.upper()
        return is_corr, raw[:100]
    except Exception as e:
        return False, str(e)


def extract_sessions_from_conv(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ordered sessions with timestamps and turn contents from a conversation dict."""
    sess_keys = sorted(
        [k for k in conv if k.startswith("session_") and "_date_time" not in k],
        key=lambda k: int(re.findall(r"\d+", k.split("session_")[-1])[0]) if re.findall(r"\d+", k.split("session_")[-1]) else 0,
    )

    sessions = []
    for sk in sess_keys:
        nums = re.findall(r"\d+", sk.split("session_")[-1])
        sn = int(nums[0]) if nums else 1
        date_str = conv.get(f"session_{sn}_date_time", "")
        raw_turns = conv[sk]
        formatted_turns = []
        for t in raw_turns:
            spk = t.get("speaker", "Unknown")
            txt = t.get("text", "")
            blip = t.get("blip_caption", "")
            turn_str = f"[{spk}] {txt}"
            if blip:
                turn_str += f" (photo: {blip})"
            formatted_turns.append(turn_str)

        sessions.append({
            "session_id": f"session_{sn}",
            "date": date_str,
            "turns": formatted_turns,
        })
    return sessions


def run_benchmark(
    data_path: str,
    config_path: str,
    system_names: List[str],
    out_file: Optional[str] = None,
    limit: Optional[int] = None,
    use_judge: bool = False,
    use_bertscore: bool = False,
    db_dir: str = "data/benchmarks/eval_banks",
) -> Dict[str, Any]:
    print("=" * 70)
    print("FAST-ASEM LONG-TERM MEMORY BENCHMARK RUNNER")
    print(f"Data file:   {data_path}")
    print(f"Config:      {config_path}")
    print(f"Systems:     {', '.join(system_names)}")
    print(f"Judge mode:  {'ENABLED' if use_judge else 'DISABLED'}")
    print(f"BERTScore:   {'ENABLED' if use_bertscore else 'DISABLED'}")
    print("=" * 70)

    # Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        conversations = raw_data
    elif isinstance(raw_data, dict):
        if "conversations" in raw_data:
            conversations = raw_data["conversations"]
        elif any(k.startswith("session_") for k in raw_data):
            conversations = [raw_data]
        else:
            conversations = [raw_data]
    else:
        raise ValueError(f"Unknown data structure in {data_path}")

    if limit is not None:
        conversations = conversations[:limit]

    print(f"Loaded {len(conversations)} conversation(s) to evaluate.\n")

    # Load config and backend
    cfg = ASEMConfig.load(config_path)
    backend = build_backend(cfg.inference)

    # System instances
    systems: Dict[str, Any] = {}
    for name in system_names:
        if name == "FastASEM":
            systems[name] = build_fast_asem_system(config_path, db_dir)
        elif name == "ASEM":
            systems[name] = build_asem_system(config_path, db_dir)
        elif name == "ASEMv2":
            systems[name] = build_asem_v2_system(config_path, db_dir)
        else:
            baselines = build_baselines(config_path, db_dir)
            if name in baselines:
                systems[name] = baselines[name]
            else:
                raise ValueError(f"Unknown system name: {name}")

    results: Dict[str, Any] = {
        "metadata": {
            "data_path": data_path,
            "config_path": config_path,
            "num_conversations": len(conversations),
            "timestamp": datetime.now().isoformat(),
        },
        "systems": {},
    }

    for sys_name, sys_instance in systems.items():
        print(f"--- Running System: {sys_name} ---")
        sys_results = {
            "total_questions": 0,
            "em_score": 0.0,
            "rouge_l": 0.0,
            "judge_score": 0.0,
            "ingest_time_sec": 0.0,
            "qa_time_sec": 0.0,
            "by_category": defaultdict(lambda: {"total": 0, "em": 0.0, "rouge_l": 0.0, "judge": 0.0}),
            "qa_log": [],
        }

        for conv_idx, conv in enumerate(conversations):
            conv_id = conv.get("conversation_id", f"conv_{conv_idx + 1}")
            # LoCoMo nests the actual session turns under conv["conversation"];
            # fall back to conv itself for raw-conversation inputs.
            conv_data = conv.get("conversation", conv) if isinstance(conv, dict) else conv
            sessions = extract_sessions_from_conv(conv_data)
            qa_items = conv.get("qa", [])

            # Skip category 5 (adversarial) if desired or keep all
            qa_items = [q for q in qa_items if q.get("category", 0) != 5]

            # Ingest conversation
            t0_ingest = time.time()
            if hasattr(sys_instance, "reset"):
                sys_instance.reset()

            if sys_name == "FastASEM":
                sys_instance.ingest_conversation(sessions)
            elif hasattr(sys_instance, "ingest_conversation"):
                all_turns = []
                for s in sessions:
                    all_turns.extend(s["turns"])
                sys_instance.ingest_conversation(all_turns)
            elif hasattr(sys_instance, "ingest"):
                for s in sessions:
                    for t in s["turns"]:
                        sys_instance.ingest(t)
            elif hasattr(sys_instance, "ingest_turn"):
                for s in sessions:
                    for t in s["turns"]:
                        sys_instance.ingest_turn({"content": t})

            ingest_dur = time.time() - t0_ingest
            sys_results["ingest_time_sec"] += ingest_dur

            print(f"  [{conv_id}] Ingested {len(sessions)} sessions in {ingest_dur:.2f}s. Answering {len(qa_items)} QA pairs...")

            # Run QA
            t0_qa = time.time()
            for q_idx, qa in enumerate(qa_items):
                question = qa.get("question", "")
                gold_answer = qa.get("answer", "")
                cat_id = qa.get("category", 1)
                cat_name = CATEGORY_NAMES.get(cat_id, f"Category {cat_id}")

                if not question or not gold_answer:
                    continue

                if sys_name in ["NoMemory", "FullContext", "SimRetrieval", "AtomicLinking", "RLManagerOnly", "ValueRetrievalOnly"]:
                    # Gather history for baselines if needed
                    history = [t for s in sessions for t in s["turns"]]
                    pred = sys_instance.answer(question, history)
                else:
                    pred = sys_instance.answer(question)

                em = compute_em(pred, gold_answer)
                rouge = compute_rouge_l(pred, gold_answer)

                judge_ok = False
                if use_judge:
                    judge_ok, _ = evaluate_with_judge(backend, question, gold_answer, pred)

                sys_results["total_questions"] += 1
                sys_results["em_score"] += em
                sys_results["rouge_l"] += rouge
                if use_judge:
                    sys_results["judge_score"] += 1.0 if judge_ok else 0.0

                cat_stats = sys_results["by_category"][cat_name]
                cat_stats["total"] += 1
                cat_stats["em"] += em
                cat_stats["rouge_l"] += rouge
                if use_judge:
                    cat_stats["judge"] += 1.0 if judge_ok else 0.0

                sys_results["qa_log"].append({
                    "conv_id": conv_id,
                    "category": cat_name,
                    "question": question,
                    "gold": gold_answer,
                    "pred": pred,
                    "em": em,
                    "rouge_l": rouge,
                    "judge_correct": judge_ok if use_judge else None,
                })

            qa_dur = time.time() - t0_qa
            sys_results["qa_time_sec"] += qa_dur

        # BERTScore-F1 (batched over all QA pairs for this system)
        if use_bertscore:
            preds = [entry["pred"] for entry in sys_results["qa_log"]]
            refs = [entry["gold"] for entry in sys_results["qa_log"]]
            bs_scores = compute_bertscore_batch(preds, refs)
            for entry, bs in zip(sys_results["qa_log"], bs_scores):
                entry["bertscore_f1"] = bs
            sys_results["bertscore_f1"] = sum(bs_scores) / max(1, len(bs_scores))

        # Compute averages
        n_q = max(1, sys_results["total_questions"])
        sys_results["avg_em"] = sys_results["em_score"] / n_q
        sys_results["avg_rouge_l"] = sys_results["rouge_l"] / n_q
        if use_judge:
            sys_results["avg_judge"] = sys_results["judge_score"] / n_q

        for cname, cstat in sys_results["by_category"].items():
            cn = max(1, cstat["total"])
            cstat["avg_em"] = cstat["em"] / cn
            cstat["avg_rouge_l"] = cstat["rouge_l"] / cn
            if use_judge:
                cstat["avg_judge"] = cstat["judge"] / cn

        results["systems"][sys_name] = sys_results
        summary = f"  => {sys_name} Summary: EM={sys_results['avg_em']:.3f} | ROUGE-L={sys_results['avg_rouge_l']:.3f}"
        if use_judge:
            summary += f" | Judge={sys_results.get('avg_judge', 0.0):.3f}"
        if use_bertscore:
            summary += f" | BERTScore-F1={sys_results.get('bertscore_f1', 0.0):.3f}"
        summary += f" | Ingest={sys_results['ingest_time_sec']:.1f}s | QA={sys_results['qa_time_sec']:.1f}s\n"
        print(summary)

    # Save output
    if out_file:
        os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results written to: {out_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Long-Term Memory Benchmark Runner")
    parser.add_argument("--data", default="datasets/locomo/locomo_conversation_1.json", help="Path to JSON dataset file")
    parser.add_argument("--config", default="configs/presets/sota_benchmark.yaml", help="Path to config or preset name")
    parser.add_argument("--systems", nargs="+", default=["FastASEM"], help="List of systems to evaluate (FastASEM, NoMemory, FullContext, etc.)")
    parser.add_argument("--out", default="outputs/benchmark_results.json", help="Output results file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of conversations")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-a-Judge evaluation")
    parser.add_argument("--bertscore", action="store_true", help="Enable BERTScore-F1 evaluation")
    parser.add_argument("--db-dir", default="data/benchmarks/eval_banks", help="Database storage directory")
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data,
        config_path=args.config,
        system_names=args.systems,
        out_file=args.out,
        limit=args.limit,
        use_judge=args.judge,
        use_bertscore=args.bertscore,
        db_dir=args.db_dir,
    )


if __name__ == "__main__":
    main()
