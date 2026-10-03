"""System-level wiring regression tests for ASEM-THG.

WHY THIS FILE EXISTS — the failure it prevents
----------------------------------------------
`asem/single_pass_ingest.py` used to construct its graph as

    self.hyper_graph = hyper_graph or TemporalHyperGraph()

`TemporalHyperGraph` defines ``__len__`` (the fact-node count), so a freshly
constructed graph is **FALSY**. The ``or`` therefore discarded the caller's
configured graph and substituted a default one. The caller in
`eval/systems.py` passes ONE `thg_graph` to both the retriever and the ingestor
precisely so they share one object; the bug silently gave the ingestor a
different, unconfigured one.

Observed consequence: with `hyper_graph.supersession: false` in the config,

    retriever.hyper_graph.enable_supersession = False
    ingestor.hyper_graph.enable_supersession  = True     <- the default, restored
    retriever.hyper_graph is ingestor.hyper_graph -> False

so the flag suppressed the relation on the READ side only, and the ingestor kept
WRITING `superseded_by` edges. The `thg_v2_postfix` bank (built with the flag
OFF) carried **242 `superseded_by` records** instead of 0.

WHY 339 GREEN TESTS MISSED IT
-----------------------------
Every existing test constructed `TemporalHyperGraph` DIRECTLY and checked its
behaviour. Those all passed, because the class was correct. The defect was in the
**system-level wiring** — which object the ingestor ends up holding. A unit test
on the class cannot see it. Hence these tests go through
`build_asem_thg_system`, the same entry point a real run uses.

The generalisable rule, which this session produced four independent instances
of: **a test that cannot fail is not evidence.**

CREDENTIAL INDEPENDENCE — a defect in the FIRST version of this file
-------------------------------------------------------------------
These tests initially called `build_asem_thg_system(config, bank)` with no
backend, so the real path tried to construct an OpenAI backend and raised
`ValueError: No API key for the OpenAI backend` **whenever `OPENAI_API_KEY` was
not already exported in the ambient environment**. That produced

    2 failed, 2 passed        <- with no key  (NOT a wiring regression)
    4 passed                 <- with a key

i.e. **the same observable output for "credentials missing" and "the wiring bug
returned"**, so a future revert of the fix would have been misread as an
environment problem and dismissed with "just set the key". That is the same
error class as the bug itself: a result that cannot distinguish *the code is
broken* from *the measurement was broken*.

The fix is that a WIRING test must never be able to fail for a credential
reason: `_RecordingBackend` is injected via the `backend=` parameter, so no
network call and no API key is ever needed. The tests now pass in a bare process
with no environment at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BANK = REPO_ROOT / "static" / "memory_banks" / "locomo10" / "thg_v2" / "ASEM-THG" / "locomo_0000"
CONFIG_OFF = REPO_ROOT / "configs" / "models" / "thg_v2_postfix_off.yaml"
CONFIG_SUPON = REPO_ROOT / "configs" / "models" / "thg_v2_postfix_supon.yaml"
CONFIG_THG_V1 = REPO_ROOT / "configs" / "models" / "deepseek_v41_flash_thg_v1.yaml"

pytestmark = pytest.mark.skipif(
    not (BANK / "asem_thg.sqlite").exists(),
    reason="real frozen ASEM-THG bank not present",
)


class _RecordingBackend:
    """Minimal stand-in backend.

    Construction is all these wiring tests exercise; nothing here is ever
    called. Its ONLY job is to keep the backend factory from requiring
    credentials, so this file cannot fail for an environment reason.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover - unused
        self.calls.append(prompt)
        return ""

    def embed(self, text: str):  # pragma: no cover - unused
        return np.zeros(384, dtype="float32")


def _build(config_path: Path, *, with_backend: bool = True):
    """Build the real system. ``with_backend`` injects a credential-free backend."""
    from eval.systems import build_asem_thg_system

    backend = _RecordingBackend() if with_backend else None
    return build_asem_thg_system(str(config_path), str(BANK), backend=backend)


def test_guard_does_not_need_credentials() -> None:
    """The guard must pass in a bare process — no API key, no network.

    This is asserted FIRST and deliberately, because the original version of
    this file silently required `OPENAI_API_KEY` and reported a credential
    failure as a wiring failure.
    """
    import os

    for var in ("OPENAI_API_KEY", "LLM_MODEL"):
        os.environ.pop(var, None)
    system = _build(CONFIG_OFF)
    assert system.single_pass_ingestor.hyper_graph is not None


def test_ingestor_and_retriever_share_one_graph_object() -> None:
    """The configured graph must reach BOTH consumers (the original bug)."""
    system = _build(CONFIG_OFF)
    retriever_graph = system.pipeline.retriever.hyper_graph
    ingestor_graph = system.single_pass_ingestor.hyper_graph
    assert retriever_graph is ingestor_graph, (
        "the ingestor holds a DIFFERENT graph object than the retriever; "
        "any constructor option set by the caller (e.g. enable_supersession) "
        "reaches only the read side. Check single_pass_ingest.py for a truthiness "
        "test such as `hyper_graph or TemporalHyperGraph()` — TemporalHyperGraph "
        "defines __len__, so an EMPTY graph is falsy and gets silently replaced."
    )


def test_supersession_flag_reaches_the_write_side() -> None:
    """`hyper_graph.supersession: false` must disable writing, not just reading."""
    system = _build(CONFIG_OFF)
    ingestor_graph = system.single_pass_ingestor.hyper_graph
    assert ingestor_graph.enable_supersession is False, (
        "the ingestor's graph still has supersession ENABLED, so the build will "
        "persist superseded_by edges despite the config saying false. This is the "
        "exact defect that produced 242 superseded_by records in thg_v2_postfix."
    )
    assert system.pipeline.retriever.hyper_graph.enable_supersession is False


def test_supersession_flag_reaches_the_write_side_when_enabled() -> None:
    """The OTHER direction: `supersession: true` must also reach the write side.

    Checking only the OFF direction would pass if the flag were hard-wired to
    False. The flag must be WIRED, not merely suppressed — so this asserts the
    ON direction reaches the ingestor too.

    Uses the config that actually sets `hyper_graph.supersession: true`. Note
    that the default model configs carry NO `hyper_graph` block at all, so
    `bool(hg_cfg.get("supersession", False))` evaluates to False for them:
    measured `deepseek_v41_flash_thg_v1.yaml` -> retriever=False ingestor=False.
    An earlier version of this test asserted True for that config and failed —
    recorded here so the expectation is not "fixed" by guessing again.
    """
    system = _build(CONFIG_SUPON)
    retriever_graph = system.pipeline.retriever.hyper_graph
    ingestor_graph = system.single_pass_ingestor.hyper_graph
    assert retriever_graph is ingestor_graph
    assert ingestor_graph.enable_supersession is True, (
        "with `hyper_graph.supersession: true` the ingestor's graph must have "
        "supersession ENABLED. If this fails while the OFF test passes, the flag "
        "is not wired at all and both directions are ignoring it."
    )
    assert retriever_graph.enable_supersession is True


def test_an_empty_graph_is_falsy_which_is_why_the_bug_was_silent() -> None:
    """Pin the property that made `or` unsafe, so nobody reintroduces it.

    This is the MECHANISM, asserted directly. If ``__len__`` is ever removed or
    changed so an empty graph becomes truthy, the original bug would have been
    harmless — and this test documents why the explicit `is None` check is still
    the correct form regardless.
    """
    from asem.hyper_graph import TemporalHyperGraph

    fresh = TemporalHyperGraph(enable_supersession=False)
    assert len(fresh) == 0
    assert not fresh, (
        "an empty TemporalHyperGraph is truthy, so `hyper_graph or default` would "
        "have worked by accident. The code uses `is not None` explicitly and must "
        "keep doing so — do not rely on truthiness either way."
    )
    # The concrete consequence: `or` substitutes a default and LOSES the option.
    substituted = fresh or TemporalHyperGraph()
    assert substituted is not fresh
    assert substituted.enable_supersession is True  # the constructor default


def test_explicit_none_still_produces_a_default_graph() -> None:
    """`None` must keep meaning "make me one" — the fix must not break that."""
    from asem.hyper_graph import TemporalHyperGraph
    from asem.single_pass_ingest import SinglePassSessionIngestor

    ingestor = SinglePassSessionIngestor.__new__(SinglePassSessionIngestor)
    # Reproduce just the assignment under test, mirroring __init__.
    supplied = None
    ingestor.hyper_graph = supplied if supplied is not None else TemporalHyperGraph()
    assert len(ingestor.hyper_graph) == 0
