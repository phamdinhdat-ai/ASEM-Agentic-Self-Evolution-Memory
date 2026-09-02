"""Configuration loader and schema definition for ASEM and Fast-ASEM."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
_PRESETS_DIR = _CONFIGS_DIR / "presets"


@dataclass
class HyperparametersConfig:
    k1: int = 20
    k2: int = 5
    k: int = 5
    delta: float = 0.30
    lambda_weight: float = 0.40  # mapped from 'lambda' in YAML
    alpha: float = 0.10
    q0: float = 0.50


@dataclass
class WriteGateConfig:
    enabled: bool = True
    tau_high: float = 0.45
    tau_redund: float = 0.92


@dataclass
class RetrieverConfig:
    mode: str = "rrf"  # "rrf" | "standard" | "enhanced"
    use_bm25: bool = True
    use_entity_filter: bool = True
    use_temporal_boost: bool = True
    max_hops: int = 2
    hop_decay: float = 0.70
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 0.8
    entity_weight: float = 0.6
    temporal_weight: float = 0.5


@dataclass
class AnswerConfig:
    direct_mode: bool = True  # single-pass direct QA generation
    include_dates: bool = True
    max_context_notes: int = 8
    max_tokens: int = 512
    temperature: float = 0.1


@dataclass
class IngestionConfig:
    mode: str = "session_batch"  # "session_batch" | "turn_by_turn"
    lazy_evolution: bool = True
    link_tau: float = 0.35
    max_notes_per_session: int = 20


@dataclass
class ASEMConfig:
    preset: str = "sota_benchmark"  # "fast_eval" | "sota_benchmark" | "deep_evolution" | "custom"
    inference: Dict[str, Any] = field(default_factory=lambda: {
        "backend": "langchain",
        "langchain": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "embedder_provider": "huggingface",
            "embedder_name": "sentence-transformers/all-MiniLM-L6-v2",
        }
    })
    hyperparameters: HyperparametersConfig = field(default_factory=HyperparametersConfig)
    write_gate: WriteGateConfig = field(default_factory=WriteGateConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    answer: AnswerConfig = field(default_factory=AnswerConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    link_tau: float = 0.35
    llm_retry: Dict[str, Any] = field(default_factory=lambda: {"max_retries": 1})
    logging: Dict[str, Any] = field(default_factory=lambda: {"level": "INFO"})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ASEMConfig:
        cfg = cls()
        if not data:
            return cfg

        if "preset" in data:
            cfg.preset = data["preset"]

        if "inference" in data:
            cfg.inference = data["inference"]

        if "hyperparameters" in data:
            hp = data["hyperparameters"]
            cfg.hyperparameters = HyperparametersConfig(
                k1=hp.get("k1", 20),
                k2=hp.get("k2", 5),
                k=hp.get("k", 5),
                delta=hp.get("delta", 0.30),
                lambda_weight=hp.get("lambda", hp.get("lambda_weight", 0.40)),
                alpha=hp.get("alpha", 0.10),
                q0=hp.get("q0", 0.50),
            )

        if "write_gate" in data:
            wg = data["write_gate"]
            cfg.write_gate = WriteGateConfig(
                enabled=wg.get("enabled", True),
                tau_high=wg.get("tau_high", 0.45),
                tau_redund=wg.get("tau_redund", 0.92),
            )

        if "retriever" in data:
            rt = data["retriever"]
            cfg.retriever = RetrieverConfig(
                mode=rt.get("mode", "rrf"),
                use_bm25=rt.get("use_bm25", True),
                use_entity_filter=rt.get("use_entity_filter", True),
                use_temporal_boost=rt.get("use_temporal_boost", True),
                max_hops=rt.get("max_hops", 2),
                hop_decay=rt.get("hop_decay", 0.70),
                rrf_k=rt.get("rrf_k", 60),
                dense_weight=rt.get("dense_weight", 1.0),
                bm25_weight=rt.get("bm25_weight", 0.8),
                entity_weight=rt.get("entity_weight", 0.6),
                temporal_weight=rt.get("temporal_weight", 0.5),
            )

        if "answer" in data:
            ans = data["answer"]
            cfg.answer = AnswerConfig(
                direct_mode=ans.get("direct_mode", True),
                include_dates=ans.get("include_dates", True),
                max_context_notes=ans.get("max_context_notes", 8),
                max_tokens=ans.get("max_tokens", 512),
                temperature=ans.get("temperature", 0.1),
            )

        if "ingestion" in data:
            ing = data["ingestion"]
            cfg.ingestion = IngestionConfig(
                mode=ing.get("mode", "session_batch"),
                lazy_evolution=ing.get("lazy_evolution", True),
                link_tau=ing.get("link_tau", 0.35),
                max_notes_per_session=ing.get("max_notes_per_session", 20),
            )

        if "link_tau" in data:
            cfg.link_tau = float(data["link_tau"])

        if "llm_retry" in data:
            cfg.llm_retry = data["llm_retry"]

        if "logging" in data:
            cfg.logging = data["logging"]

        return cfg

    @classmethod
    def load(cls, config_path_or_preset: Optional[str] = None) -> ASEMConfig:
        """Load configuration from a YAML file path or a preset name."""
        if config_path_or_preset is None:
            config_path_or_preset = str(_CONFIGS_DIR / "default.yaml")

        # Check if it's a preset name
        preset_file = _PRESETS_DIR / f"{config_path_or_preset}.yaml"
        if preset_file.exists():
            path = preset_file
        else:
            path = Path(config_path_or_preset)
            if not path.exists():
                alt_path = _CONFIGS_DIR / config_path_or_preset
                if alt_path.exists():
                    path = alt_path

        if not path.exists():
            raise FileNotFoundError(f"Config file or preset not found: {config_path_or_preset}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # If a base preset is specified inside the YAML, load it first and merge
        if "preset" in data and data["preset"] != "custom":
            preset_name = data["preset"]
            base_preset_file = _PRESETS_DIR / f"{preset_name}.yaml"
            if base_preset_file.exists() and base_preset_file.resolve() != path.resolve():
                with open(base_preset_file, "r", encoding="utf-8") as bf:
                    base_data = yaml.safe_load(bf) or {}
                # Deep merge: base_data updated with data
                merged = _deep_merge(base_data, data)
                return cls.from_dict(merged)

        return cls.from_dict(data)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
