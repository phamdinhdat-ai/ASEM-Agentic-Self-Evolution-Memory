"""Profile pipeline stage latency over multiple turns."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict

import yaml

from asem.backends import build_backend
from asem.link_evolver import LinkEvolver
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager
from asem.note import NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater
from asem.answer_agent import AnswerAgent


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def build_pipeline(config_path: str, db_path: str) -> ASEMPipeline:
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    note_prompt = _load_text("data/prompts/P1_note_construction.txt")
    link_prompt = _load_text("data/prompts/P2_link_generation.txt")
    evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

    note_constructor = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
    memory_manager = MemoryManager(
        backend=backend,
        prompt_template="{content}\n{memory}",
    )
    link_evolver = LinkEvolver(
        backend=backend,
        link_prompt_template=link_prompt,
        evolve_prompt_template=evolve_prompt,
        k=hp["k"],
    )
    retriever = HybridRetriever(
        backend=backend,
        k1=hp["k1"],
        k2=hp["k2"],
        delta=hp["delta"],
        lambda_weight=hp["lambda"],
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template="{query} {candidates}",
        baseline_prompt_template="{query} {context}",
    )
    updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"],
        q0=hp["q0"],
        summary_prompt_template="{query} {answer} {reward}",
        note_constructor=note_constructor,
    )

    return ASEMPipeline(
        memory_bank=MemoryBank(db_path),
        note_constructor=note_constructor,
        memory_manager=memory_manager,
        link_evolver=link_evolver,
        retriever=retriever,
        answer_agent=answer_agent,
        utility_updater=updater,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ASEM pipeline stage latency")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--db", default="data/benchmarks/profile_bank.sqlite")
    parser.add_argument("--turns", type=int, default=5)

    args = parser.parse_args()

    pipeline = build_pipeline(args.config, args.db)

    summary = None
    for i in range(args.turns):
        answer, profiler = pipeline.profile_turn(
            content=f"content {i}",
            query=f"query {i}",
            reward=1.0,
            timestamp=datetime.utcnow(),
        )
        summary = profiler.summary()
        print(f"Turn {i} answer: {answer}")
        print(f"Stage timings (ms): {summary}")


if __name__ == "__main__":
    main()
