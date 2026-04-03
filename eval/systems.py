"""System builders for evaluation runners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from typing import Dict, List

import yaml

from asem.answer_agent import AnswerAgent
from asem.backends import build_backend
from asem.link_evolver import LinkEvolver
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager
from asem.note import NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater
from eval.baselines import (
    AtomicLinking,
    FullContext,
    NoMemory,
    RLManagerOnly,
    SimRetrieval,
    ValueRetrievalOnly,
)


@dataclass
class ASEMSystem:
    """Wrapper that exposes the ASEM pipeline as a baseline-style interface."""

    pipeline: ASEMPipeline

    def answer(self, query: str, history: List[str]) -> str:
        for item in history:
            self.pipeline.write_path(item, datetime.utcnow())
        _, answer = self.pipeline.read_path(query)
        return answer


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_asem_system(config_path: str, db_dir: str) -> ASEMSystem:
    cfg = _load_config(config_path)
    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    note_prompt = _load_text("data/prompts/P1_note_construction.txt")
    link_prompt = _load_text("data/prompts/P2_link_generation.txt")
    evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

    note_constructor = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
    memory_manager = MemoryManager(backend=backend, prompt_template="{content}\n{memory}")
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
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"],
        q0=hp["q0"],
        summary_prompt_template="{query} {answer} {reward}",
        note_constructor=note_constructor,
    )

    _ensure_dir(db_dir)
    bank_path = os.path.join(db_dir, "asem.sqlite")
    pipeline = ASEMPipeline(
        memory_bank=MemoryBank(bank_path),
        note_constructor=note_constructor,
        memory_manager=memory_manager,
        link_evolver=link_evolver,
        retriever=retriever,
        answer_agent=answer_agent,
        utility_updater=utility_updater,
    )

    return ASEMSystem(pipeline=pipeline)


def build_baselines(config_path: str, db_dir: str) -> Dict[str, object]:
    cfg = _load_config(config_path)
    backend = build_backend(cfg["inference"])
    hp = cfg["hyperparameters"]

    note_prompt = _load_text("data/prompts/P1_note_construction.txt")
    link_prompt = _load_text("data/prompts/P2_link_generation.txt")
    evolve_prompt = _load_text("data/prompts/P3_memory_evolution.txt")

    _ensure_dir(db_dir)
    bank_path = os.path.join(db_dir, "baseline.sqlite")
    bank = MemoryBank(bank_path)

    note_constructor = NoteConstructor(backend=backend, prompt_template=note_prompt, q0=hp["q0"])
    memory_manager = MemoryManager(backend=backend, prompt_template="{content}\n{memory}")
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
    utility_updater = UtilityUpdater(
        backend=backend,
        alpha=hp["alpha"],
        q0=hp["q0"],
        summary_prompt_template="{query} {answer} {reward}",
        note_constructor=note_constructor,
    )

    return {
        "NoMemory": NoMemory(backend=backend, prompt_template="{query}"),
        "FullContext": FullContext(backend=backend, prompt_template="{query}\n{context}"),
        "SimRetrieval": SimRetrieval(
            backend=backend,
            memory_bank=bank,
            top_k=hp["k2"],
            prompt_template="{query}\n{context}",
        ),
        "AtomicLinking": AtomicLinking(
            backend=backend,
            memory_bank=bank,
            note_constructor=note_constructor,
            link_evolver=link_evolver,
            top_k=hp["k2"],
            prompt_template="{query}\n{context}",
        ),
        "RLManagerOnly": RLManagerOnly(
            backend=backend,
            memory_bank=bank,
            note_constructor=note_constructor,
            memory_manager=memory_manager,
            top_k=hp["k2"],
            prompt_template="{query}\n{context}",
        ),
        "ValueRetrievalOnly": ValueRetrievalOnly(
            backend=backend,
            memory_bank=bank,
            note_constructor=note_constructor,
            retriever=retriever,
            utility_updater=utility_updater,
            answer_agent=answer_agent,
        ),
    }


def get_systems(
    config_path: str = "configs/default.yaml",
    db_dir: str = "data/benchmarks/eval_banks",
) -> Dict[str, object]:
    """Build ASEM and baseline systems for evaluation runners."""

    systems = build_baselines(config_path, db_dir)
    systems["ASEM"] = build_asem_system(config_path, db_dir)
    return systems
