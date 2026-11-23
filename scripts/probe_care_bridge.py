"""CPU-only retrieval probe on untouched LoCoMo development IDs.

Gold evidence is consulted only after each rank list has been computed. This
script helps select a cheap retrieval front end before any Kaggle model run.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from nltk.stem import PorterStemmer

from asem.care.sourcefirst import CareSourceFirst, _words
from scripts.run_care_locomo import select_questions, turns_for


def stemmed_rank(turns, question):
    stem = PorterStemmer().stem
    documents = [Counter(stem(word) for word in _words(turn.speaker + " " + turn.text))
                 for turn in turns]
    lengths = [sum(document.values()) for document in documents]
    average = sum(lengths) / len(lengths) or 1.0
    df = Counter(word for document in documents for word in document)
    query = set(stem(word) for word in _words(question))
    scores = []
    for i, document in enumerate(documents):
        score = 0.0
        for word in query & document.keys():
            frequency = document[word]
            idf = math.log(1 + (len(turns) - df[word] + 0.5) / (df[word] + 0.5))
            score += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * lengths[i] / average))
        scores.append(score)
    return [turns[i] for i in sorted(range(len(turns)), key=lambda i: (-scores[i], i))]


def main():
    dataset = json.loads(Path("build/care/locomo10.json").read_text(encoding="utf-8"))
    by_sample = {conversation["sample_id"]: conversation for conversation in dataset}
    questions = select_questions(dataset, per_category=4, start_per_category=9,
                                 categories=(1, 2, 3, 4, 5))
    summary = defaultdict(lambda: defaultdict(list))
    for item in questions:
        turns = turns_for(by_sample[item["sample_id"]]["conversation"])
        memory = CareSourceFirst(turns)
        base = memory.rank(item["question"], limit=8)
        stemmed = stemmed_rank(turns, item["question"])
        union_4_4_ids = {turn.source_id for turn in base[:4] + stemmed[:4]}
        union_8_8_ids = {turn.source_id for turn in base + stemmed[:8]}
        stem_pool_ids = {turn.source_id for turn in stemmed[:18]}
        for seed in stemmed[:4]:
            stem_pool_ids.update(turn.source_id for turn in memory._neighbors(seed.source_id))
        hybrid_ids = stem_pool_ids | {turn.source_id for turn in memory.candidate_pool(item["question"])}
        variants = {"bm25_8": base,
                    "stem_8": stemmed[:8],
                    "bm25_16": memory.rank(item["question"], limit=16),
                    "stem_16": stemmed[:16],
                    "union_4_4": [turn for turn in turns if turn.source_id in union_4_4_ids],
                    "union_8_8": [turn for turn in turns if turn.source_id in union_8_8_ids],
                    "stem_pool": [turn for turn in turns if turn.source_id in stem_pool_ids],
                    "hybrid_pool": [turn for turn in turns if turn.source_id in hybrid_ids],
                    "pool": memory.candidate_pool(item["question"])}
        evidence = set(item["evidence"])
        if not evidence:
            continue
        for name, sources in variants.items():
            summary[item["category"]][name].append(
                len(evidence & {turn.source_id for turn in sources}) / len(evidence))
        for name in ("pool", "stem_pool", "hybrid_pool"):
            visible, _ = memory._visible_lines(variants[name], max_chars=10000)
            summary[item["category"]][name + "_visible"].append(
                len(evidence & {turn.source_id for turn in visible}) / len(evidence))
    print(json.dumps({category: {name: round(sum(values) / len(values), 4)
        for name, values in variants.items()} for category, variants in summary.items()}, indent=2))
    names = ("bm25_8", "stem_8", "bm25_16", "stem_16", "union_4_4", "union_8_8",
             "stem_pool", "stem_pool_visible", "pool", "pool_visible", "hybrid_pool", "hybrid_pool_visible")
    print("answerable_1_to_4", json.dumps({name: round(
        sum(value for category in (1, 2, 3, 4) for value in summary[category][name]) /
        sum(len(summary[category][name]) for category in (1, 2, 3, 4)), 4)
        for name in names}))


if __name__ == "__main__":
    main()
