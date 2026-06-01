"""
Generate ASEM training data from the LoCoMo dataset (locomo10.json).

The script converts LoCoMo conversations + QA pairs into the format expected
by training/train_answer.py:

    {
        "query":       str,          # the question
        "candidates":  List[dict],   # note-like dicts the answer agent must select from
        "gold_answer": str,          # ground-truth answer
        "category":    int,          # 1-5  (see below)
        "session_id":  str,
        "evidence":    List[str]     # original evidence keys e.g. ["D1:3"]
    }

LoCoMo QA categories
---------------------
1 – Single-hop factual
2 – Temporal reasoning
3 – Commonsense / inference
4 – Conversational (answer is a direct quote from the dialogue)
5 – Adversarial  (question is deliberately mis-attributed; answer = adversarial_answer)

Candidate notes are built from the dialogue turns that are referenced by the
evidence keys.  For each QA item we also add a small number of distractor turns
(turns NOT in the evidence) so the model must learn to select the right ones.

Usage
-----
    python scripts/generate_training_data.py \
        --input  datasets/locomo/locomo10.json \
        --output data/training \
        --distractors 3 \
        --split 0.9
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_NAMES = {
    1: "single_hop",
    2: "temporal",
    3: "commonsense",
    4: "conversational",
    5: "adversarial",
}


def _parse_dia_id(dia_id: str) -> Tuple[int, int]:
    """Parse 'D3:7' → (session=3, turn=7)."""
    m = re.match(r"D(\d+):(\d+)", dia_id)
    if m:
        return int(m.group(1)), int(m.group(2))
    return -1, -1


def _build_session_index(conversation: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return a flat dict  dia_id → turn dict  for all sessions."""
    index: Dict[str, Dict[str, Any]] = {}
    for key, value in conversation.items():
        if not key.startswith("session_") or not isinstance(value, list):
            continue
        for turn in value:
            dia_id = turn.get("dia_id")
            if dia_id:
                index[dia_id] = turn
    return index


def _turn_to_candidate(turn: Dict[str, Any], note_id: Optional[str] = None) -> Dict[str, Any]:
    """Convert a dialogue turn into a candidate note dict."""
    speaker = turn.get("speaker", "Unknown")
    text = turn.get("text", "")
    dia_id = turn.get("dia_id", "")
    blip = turn.get("blip_caption", "")

    content = f"[{speaker}] {text}"
    if blip:
        content += f" (image: {blip})"

    # Derive lightweight keywords / tags from the turn
    words = re.findall(r"\b[A-Za-z]{4,}\b", text)
    keywords = list(dict.fromkeys(w.lower() for w in words))[:6]

    session_num, _ = _parse_dia_id(dia_id)
    tags = [f"session_{session_num}", speaker.lower()]

    return {
        "id": note_id or str(uuid.uuid4()),
        "dia_id": dia_id,
        "content": content,
        "keywords": keywords,
        "tags": tags,
        "description": f"Turn {dia_id} by {speaker}",
        "utility": 0.5,
    }


def _get_session_date(conversation: Dict[str, Any], session_num: int) -> str:
    """Return the date string for a session, e.g. '1:56 pm on 8 May, 2023'."""
    key = f"session_{session_num}_date_time"
    return conversation.get(key, "unknown date")


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_examples(
    record: Dict[str, Any],
    session_id: str,
    n_distractors: int = 3,
    rng: random.Random = random.Random(42),
) -> List[Dict[str, Any]]:
    """Convert one LoCoMo record into a list of training examples."""

    conversation = record.get("conversation", {})
    qa_list = record.get("qa", [])

    # Build a flat index of all turns
    turn_index = _build_session_index(conversation)
    all_dia_ids = list(turn_index.keys())

    examples: List[Dict[str, Any]] = []

    for qa in qa_list:
        question: str = qa.get("question", "").strip()
        category: int = qa.get("category", 1)
        evidence_keys: List[str] = qa.get("evidence", [])

        # Determine gold answer
        if category == 5:
            gold_answer = str(qa.get("adversarial_answer", ""))
        else:
            raw_answer = qa.get("answer", "")
            gold_answer = str(raw_answer).strip()

        if not question or not gold_answer:
            continue

        # Build evidence candidates (turns referenced by evidence keys)
        evidence_candidates: List[Dict[str, Any]] = []
        evidence_ids: List[str] = []
        for eid in evidence_keys:
            # evidence keys may contain semicolons: "D8:6; D9:17"
            for part in re.split(r"[;,]", eid):
                part = part.strip()
                if part and part in turn_index:
                    note_id = f"ev_{part.replace(':', '_')}"
                    cand = _turn_to_candidate(turn_index[part], note_id=note_id)
                    evidence_candidates.append(cand)
                    evidence_ids.append(note_id)

        # Build distractor candidates (random turns NOT in evidence)
        non_evidence = [d for d in all_dia_ids if d not in evidence_keys]
        n_dist = min(n_distractors, len(non_evidence))
        distractor_dia_ids = rng.sample(non_evidence, n_dist) if n_dist > 0 else []
        distractor_candidates = [
            _turn_to_candidate(turn_index[d]) for d in distractor_dia_ids
        ]

        # Shuffle so evidence is not always first
        candidates = evidence_candidates + distractor_candidates
        rng.shuffle(candidates)

        # Enrich query with speaker context
        speaker_a = conversation.get("speaker_a", "Speaker A")
        speaker_b = conversation.get("speaker_b", "Speaker B")
        enriched_query = (
            f"Conversation between {speaker_a} and {speaker_b}. "
            f"Question: {question}"
        )

        examples.append({
            "session_id": session_id,
            "query": enriched_query,
            "candidates": candidates,
            "gold_answer": gold_answer,
            "gold_candidate_ids": evidence_ids,
            "category": category,
            "category_name": CATEGORY_NAMES.get(category, "unknown"),
            "evidence": evidence_keys,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
        })

    return examples


# ---------------------------------------------------------------------------
# Split & save
# ---------------------------------------------------------------------------

def split_examples(
    examples: List[Dict[str, Any]],
    train_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = examples[:]
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


def save_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def save_json(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        json.dump(records, f, indent=2, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_stats(name: str, examples: List[Dict[str, Any]]) -> None:
    by_cat: Dict[int, int] = defaultdict(int)
    for ex in examples:
        by_cat[ex["category"]] += 1
    print(f"\n{name}: {len(examples)} examples")
    for cat in sorted(by_cat):
        print(f"  category {cat} ({CATEGORY_NAMES.get(cat,'?'):>14s}): {by_cat[cat]:>4d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ASEM training data from LoCoMo")
    parser.add_argument(
        "--input",
        default="datasets/locomo/locomo10.json",
        help="Path to locomo10.json",
    )
    parser.add_argument(
        "--output",
        default="data/training",
        help="Output directory",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=3,
        help="Number of distractor turns to add per example",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Train/val split ratio (default 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--format",
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output file format",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    all_examples: List[Dict[str, Any]] = []
    for idx, record in enumerate(dataset):
        session_id = f"locomo_{idx:04d}"
        examples = build_examples(record, session_id, n_distractors=args.distractors, rng=rng)
        all_examples.extend(examples)
        print(f"  record {idx:>3d}: {len(examples)} examples")

    train_examples, val_examples = split_examples(all_examples, args.split, rng)

    print_stats("TRAIN", train_examples)
    print_stats("VAL  ", val_examples)
    print_stats("TOTAL", all_examples)

    # Save files
    save_fn = save_jsonl if args.format == "jsonl" else save_json
    ext = "jsonl" if args.format == "jsonl" else "json"

    train_path = os.path.join(args.output, f"train.{ext}")
    val_path   = os.path.join(args.output, f"val.{ext}")
    all_path   = os.path.join(args.output, f"all.{ext}")

    save_fn(train_path, train_examples)
    save_fn(val_path,   val_examples)
    save_fn(all_path,   all_examples)

    # Also save per-category splits for targeted training
    by_cat: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ex in all_examples:
        by_cat[ex["category"]].append(ex)

    for cat, exs in by_cat.items():
        cat_name = CATEGORY_NAMES.get(cat, f"cat{cat}")
        cat_path = os.path.join(args.output, "by_category", f"{cat_name}.{ext}")
        save_fn(cat_path, exs)
        print(f"  saved {len(exs):>4d} examples -> {cat_path}")

    print(f"\nDone. Files written to: {args.output}")
    print(f"  {train_path}  ({len(train_examples)} examples)")
    print(f"  {val_path}    ({len(val_examples)} examples)")


if __name__ == "__main__":
    main()
