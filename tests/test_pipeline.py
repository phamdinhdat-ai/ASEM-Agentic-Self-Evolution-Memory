"""ASEM pipeline integration tests."""

from __future__ import annotations

from datetime import datetime
import json
import tempfile

import numpy as np
import pytest

from asem.answer_agent import AnswerAgent, _render_notes_block
from asem.memory_bank import MemoryBank
from asem.memory_manager import MemoryManager
from asem.note import LinkRecord, Note, NoteConstructor
from asem.pipeline import ASEMPipeline
from asem.retriever import HybridRetriever
from asem.utility_updater import UtilityUpdater
from asem.write_gate import WriteGate


class _TaggedBackend:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> str:
        if prompt.startswith("NC:"):
            self.calls.append("NC")
            return '{"keywords": ["k"], "tags": ["t"], "description": "d"}'
        if prompt.startswith("MM:"):
            self.calls.append("MM")
            return '{"op": "ADD"}'
        if prompt.startswith("AA:"):
            self.calls.append("AA")
            return '{"selected_ids": [], "answer": "ok"}'
        if prompt.startswith("SUM:"):
            self.calls.append("SUM")
            return "summary"
        return "{}"

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


class _NoopLinkEvolver:
    def link_and_evolve(self, m_new, M):
        return None


def test_pipeline_five_turns() -> None:
    pytest.importorskip("faiss")

    backend = _TaggedBackend()

    note_constructor = NoteConstructor(
        backend=backend,
        prompt_template="NC:{content}",
        q0=0.5,
    )
    memory_manager = MemoryManager(
        backend=backend,
        prompt_template="MM:{content} {memory}",
    )
    retriever = HybridRetriever(
        backend=backend,
        k1=5,
        k2=2,
        delta=0.0,
        lambda_weight=0.5,
    )
    answer_agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
    )
    updater = UtilityUpdater(
        backend=backend,
        alpha=0.1,
        q0=0.5,
        summary_prompt_template="SUM:{query} {answer} {reward}",
        note_constructor=note_constructor,
    )

    with tempfile.TemporaryDirectory() as tmp:
        bank = MemoryBank(f"{tmp}/bank.sqlite")
        pipeline = ASEMPipeline(
            memory_bank=bank,
            note_constructor=note_constructor,
            memory_manager=memory_manager,
            link_evolver=_NoopLinkEvolver(),
            retriever=retriever,
            answer_agent=answer_agent,
            utility_updater=updater,
            # The write gate is disabled here: _TaggedBackend returns the same
            # embedding for every text, which the novelty gate would (correctly)
            # classify as a near-duplicate and never write. The gate itself is
            # covered by tests/test_write_gate.py.
            write_gate=WriteGate(enabled=False),
        )

        for i in range(5):
            answer = pipeline.run_turn(
                content=f"content {i}",
                query=f"query {i}",
                reward=1.0,
                timestamp=datetime(2024, 1, 1),
            )
            assert answer == "ok"

        assert len(bank.list_notes()) == 10
        assert backend.calls[:5] == ["NC", "MM", "AA", "SUM", "NC"]

        bank.close()


# ---------------------------------------------------------------------------
# Abstention recovery ("I don't know" handling)
# ---------------------------------------------------------------------------

class _AnswerSequenceBackend:
    """Returns a scripted sequence of answers for the answer (AA:) prompt."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.prompts = []

    def generate(self, prompt: str, **kwargs) -> str:
        if "AA:" in prompt:
            self.prompts.append(prompt)
            idx = min(len(self.prompts) - 1, len(self._answers) - 1)
            return '{"selected_ids": [], "answer": "%s"}' % self._answers[idx]
        return "{}"

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def _note(nid: str, links=None, vec=None):
    return Note(
        id=nid,
        c=f"[Caroline] {nid}",
        t=datetime(2023, 5, 8),
        K=["k"],
        G=["personal"],
        X=f"{nid}.",
        e=np.asarray(vec if vec is not None else [1.0, 0.0], dtype=float),
        L=list(links or []),
        z=np.asarray([1.0, 0.0], dtype=float),
        q=0.5,
        session_date="8 May 2023",
        entities=["Caroline"],
        speaker="Caroline",
    )


def _bank_with(count: int):
    tmp = tempfile.TemporaryDirectory()
    bank = MemoryBank(f"{tmp.name}/bank.sqlite")
    for i in range(count):
        bank.add(_note(f"n{i + 1}"))
    return tmp, bank


def _read_path_pipeline(backend, bank, *, k2: int, recovery_enabled: bool):
    """Minimal pipeline wired only for the read path."""
    note_constructor = NoteConstructor(
        backend=backend, prompt_template="NC:{content}", q0=0.5
    )
    return ASEMPipeline(
        memory_bank=bank,
        note_constructor=note_constructor,
        memory_manager=MemoryManager(
            backend=backend, prompt_template="MM:{content} {memory}"
        ),
        link_evolver=_NoopLinkEvolver(),
        retriever=HybridRetriever(
            backend=backend, k1=5, k2=k2, delta=0.0, lambda_weight=0.5
        ),
        answer_agent=AnswerAgent(
            backend=backend,
            prompt_template="AA:{query} {candidates}",
            baseline_prompt_template="BASE:{query} {context}",
        ),
        utility_updater=UtilityUpdater(
            backend=backend,
            alpha=0.1,
            q0=0.5,
            summary_prompt_template="SUM:{query} {answer} {reward}",
            note_constructor=note_constructor,
        ),
        write_gate=WriteGate(enabled=False),
        recovery_enabled=recovery_enabled,
        recovery_k2=12,
        recovery_delta=0.0,
    )


def test_read_path_recovers_from_abstention() -> None:
    """An 'I don't know' triggers exactly one widened retry, and it wins."""
    backend = _AnswerSequenceBackend(["I don't know", "a sunset"])
    tmp, bank = _bank_with(2)
    try:
        pipeline = _read_path_pipeline(backend, bank, k2=1, recovery_enabled=True)
        _, answer = pipeline.read_path("What did Caroline paint?")

        assert len(backend.prompts) == 2, "abstention must trigger exactly one recovery"
        assert "too hasty" in backend.prompts[1], "recovery prompt must carry the nudge"
        assert answer == "a sunset"
    finally:
        bank.close()
        tmp.cleanup()


def test_read_path_keeps_refusal_when_recovery_disabled() -> None:
    backend = _AnswerSequenceBackend(["I don't know", "a sunset"])
    tmp, bank = _bank_with(2)
    try:
        pipeline = _read_path_pipeline(backend, bank, k2=1, recovery_enabled=False)
        _, answer = pipeline.read_path("What did Caroline paint?")

        assert len(backend.prompts) == 1, "recovery disabled must not retry"
        assert answer == "I don't know"
    finally:
        bank.close()
        tmp.cleanup()


def test_read_path_does_not_loop_when_recovery_also_refuses() -> None:
    """A genuinely absent fact still ends as 'I don't know' — with ONE retry."""
    backend = _AnswerSequenceBackend(["I don't know", "I don't know"])
    tmp, bank = _bank_with(2)
    try:
        pipeline = _read_path_pipeline(backend, bank, k2=1, recovery_enabled=True)
        _, answer = pipeline.read_path("What is Caroline's phone number?")

        assert len(backend.prompts) == 2, "exactly one retry, never a loop"
        assert answer == "I don't know"
    finally:
        bank.close()
        tmp.cleanup()


class _StubRetriever:
    """Retriever stub returning a fixed pool.

    Deliberately NOT a dataclass, so ``dataclasses.replace`` fails and
    ``_recover`` falls back to this same object — giving the test full control
    over what the widened pass returns.
    """

    def __init__(self, backend, pool):
        self.backend = backend
        self.k2 = 1
        self.delta = 0.0
        self.link_traversal_topn = 3
        self.pool = list(pool)

    def retrieve(self, query: str, bank, **kwargs):
        return list(self.pool)


def test_recovery_reorders_pool_by_query_similarity() -> None:
    """The widened pool must be re-ranked before the answer agent trims it.

    ``AnswerAgent._fit`` drops notes from the TAIL when the context window is
    tight, so a high-similarity note that only the widened pass found must not
    be left sitting at the end of the merged list.
    """
    far = _note("far", vec=[0.0, 1.0])
    near = _note("near", vec=[1.0, 0.0])
    backend = _AnswerSequenceBackend(["by the sea"])

    tmp, bank = _bank_with(0)
    try:
        pipeline = _read_path_pipeline(backend, bank, k2=2, recovery_enabled=True)
        # Widened pass returns [far, near]; `first` already holds `far`.
        pipeline.retriever = _StubRetriever(backend, [far, near])

        merged, answer = pipeline._recover("what did Caroline paint?", [far])

        assert [n.id for n in merged] == ["near", "far"], (
            "re-rank must put the closest note first so tail-trimming keeps it"
        )
        assert answer == "by the sea"
    finally:
        bank.close()
        tmp.cleanup()


def test_traverse_links_prefers_typed_relations() -> None:
    """`contradicts` must outrank `same-topic` at equal similarity/utility."""
    backend = _AnswerSequenceBackend(["ok"])
    tmp = tempfile.TemporaryDirectory()
    bank = MemoryBank(f"{tmp.name}/bank.sqlite")
    try:
        seed = _note(
            "s",
            links=[
                LinkRecord(target_id="a", relation="same-topic"),
                LinkRecord(target_id="b", relation="contradicts"),
            ],
        )
        bank.add(seed)
        bank.add(_note("a"))
        bank.add(_note("b"))

        retriever = HybridRetriever(
            backend=backend, k1=5, k2=1, delta=0.0, lambda_weight=0.5,
            link_traversal_topn=1,
        )
        linked = retriever._traverse_links([seed], np.asarray([1.0, 0.0]), bank)

        assert [n.id for n in linked] == ["b"]
        assert "contradicts" in retriever.stats.get("traversed_relations", [])
    finally:
        bank.close()
        tmp.cleanup()


# ---------------------------------------------------------------------------
# Prompt budget (small-context backbones)
# ---------------------------------------------------------------------------

class _PromptRecordingBackend:
    def __init__(self, answer: str = '{"selected_ids": [], "answer": "ok"}') -> None:
        self.prompts = []
        self.kwargs = []
        self._answer = answer

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        return self._answer

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def _fat_note(nid: str, chars: int = 600):
    note = _note(nid)
    note.c = f"[Caroline] {nid} " + ("padding sentence. " * (chars // 18))
    note.X = note.c
    return note


def test_answer_prompt_fits_the_context_window() -> None:
    """A small window must drop low-ranked notes instead of overflowing (HTTP 400).

    Regression for: 6145 input + 2048 output > 8192 window on qwen3-4b.
    """
    backend = _PromptRecordingBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        max_tokens=512,
        context_window=2000,
    )
    notes = [_fat_note(f"n{i}") for i in range(30)]

    _, answer = agent.distil_and_answer("what happened?", notes)

    assert answer == "ok"
    prompt = backend.prompts[0]
    est = len(prompt) // 4
    assert est <= 2000 - 512, f"prompt ~{est} tokens exceeds the budget"
    # It still sent a usable prompt (trimmed, not emptied).
    assert "[1]" in prompt
    assert "said by" in prompt


def test_answer_max_tokens_is_passed_per_call() -> None:
    backend = _PromptRecordingBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        max_tokens=256,
    )

    agent.distil_and_answer("what happened?", [_note("n1")])

    assert backend.kwargs[0].get("max_tokens") == 256


def test_answer_without_budget_keeps_every_note() -> None:
    backend = _PromptRecordingBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
    )
    notes = [_fat_note(f"n{i}") for i in range(6)]

    agent.distil_and_answer("what happened?", notes)

    prompt = backend.prompts[0]
    # Context is rendered as numbered blocks: every note keeps a rank and no JSON
    # `id` field is printed (the model selects by rank).
    assert '"id"' not in prompt
    assert all(f"[{i + 1}]" in prompt for i in range(len(notes)))


# ---------------------------------------------------------------------------
# Payload budget: fewer + leaner notes for small-context backbones
# ---------------------------------------------------------------------------

def _long_note(nid: str, chars: int = 3000):
    """A note with a LONG raw turn but a SHORT distilled description.

    This is the realistic shape: `content` is the verbatim conversation turn
    while `description` is what the extractor distilled from it.
    """
    note = _note(nid)
    note.c = f"[Caroline] {nid} " + ("padding sentence. " * (chars // 18))
    note.X = f"{nid} is a short distilled fact."
    return note


def test_note_payload_drops_fields_with_no_answer_signal() -> None:
    """`tags`, `utility` and `timestamp_iso` cost tokens but answer nothing."""
    payload = AnswerAgent._note_payload(_note("n1"))

    assert set(payload) == {
        "id", "keywords", "description", "session_date",
        "entities", "speaker", "relations", "content",
    }


def test_note_payload_clips_the_raw_turn() -> None:
    note = _long_note("n1", chars=3000)

    payload = AnswerAgent._note_payload(note, content_chars=200)

    assert len(payload["content"]) <= 202, "raw turn must be clipped"
    assert payload["description"] == note.X, "the distilled fact is never trimmed"


def test_note_payload_does_not_repeat_the_description() -> None:
    """When `description == content` (FastASEM shape) only one copy is sent."""
    note = _fat_note("n1", chars=3000)  # sets X = c

    payload = AnswerAgent._note_payload(note, content_chars=0)

    assert "content" not in payload
    assert payload["description"] == note.c


def test_content_can_be_dropped_entirely() -> None:
    payload = AnswerAgent._note_payload(_long_note("n1"), content_chars=0)

    assert "content" not in payload
    assert payload["description"] == "n1 is a short distilled fact."


def test_lean_payload_is_a_fraction_of_the_raw_note() -> None:
    note = _long_note("n1", chars=3000)

    lean = len(json.dumps(AnswerAgent._note_payload(note)))

    assert lean < len(note.c) * 0.2, f"lean payload {lean} chars vs {len(note.c)} raw"


def test_relations_are_pruned_to_the_notes_in_the_prompt() -> None:
    """Regression: the full edge list was 49.9% of the prompt (873 chars/note).

    Only edges whose target is ALSO in the prompt are actionable — the model can
    read both ends. The rest collapse to a count map.
    """
    links = [LinkRecord(target_id=f"x{i}", relation="same-topic") for i in range(200)]
    links.append(LinkRecord(target_id="n2", relation="contradicts"))
    note = _note("n1", links=links)

    payload = AnswerAgent._note_payload(note, in_context={"n1", "n2"})

    assert payload["relations"] == [{"relation": "contradicts", "target_id": "n2"}]
    assert payload["also_linked"] == "same-topic:200"
    assert len(json.dumps(payload)) < 700, "payload must not grow with the edge list"


def test_relation_edges_are_capped_per_note() -> None:
    links = [LinkRecord(target_id=f"n{i}", relation="temporal") for i in range(20)]
    payload = AnswerAgent._note_payload(_note("n1", links=links), in_context={f"n{i}" for i in range(20)})

    assert len(payload["relations"]) == 6
    assert payload["also_linked"] == "temporal:14"


def test_a_note_without_links_stays_compact() -> None:
    payload = AnswerAgent._note_payload(_note("n1"))

    assert payload["relations"] == []
    assert "also_linked" not in payload


def test_max_context_notes_caps_the_prompt() -> None:
    """Notes arrive relevance-ordered, so the cap drops the lowest-ranked."""
    backend = _PromptRecordingBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        max_context_notes=2,
    )
    notes = [_note(f"n{i}") for i in range(6)]

    agent.distil_and_answer("what happened?", notes)

    prompt = backend.prompts[0]
    assert "[1]" in prompt and "[2]" in prompt
    assert "[3]" not in prompt, "the cap must drop the lowest-ranked notes"


def test_no_cap_keeps_every_note_reachable() -> None:
    backend = _PromptRecordingBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        max_context_notes=None,
    )
    notes = [_note(f"n{i}") for i in range(6)]

    agent.distil_and_answer("what happened?", notes)

    prompt = backend.prompts[0]
    assert '"id"' not in prompt
    assert all(f"[{i + 1}]" in prompt for i in range(len(notes)))


def test_max_tokens_reaches_the_retry_path() -> None:
    """The retry handler must not drop the per-call cap.

    Regression: `LLMRetryHandler(self.backend.generate)` calls
    `generate_fn(prompt)` with NO kwargs, so the cap was silently dropped and the
    client-wide `max_tokens` (2048) went out instead of 512 — overflowing an 8192
    window with a 6145-token prompt.
    """
    backend = _PromptRecordingBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        max_tokens=256,
        max_retries=1,
    )

    agent.distil_and_answer("what happened?", [_note("n1")])

    assert backend.kwargs[0].get("max_tokens") == 256


class _OverflowThenOkBackend:
    """Raises a vLLM-style context-length 400 on the first call."""

    def __init__(self) -> None:
        self.caps = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.caps.append(kwargs.get("max_tokens"))
        if len(self.caps) == 1:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': \"This model's maximum "
                "context length is 8192 tokens. However, you requested 2048 output "
                "tokens and your prompt contains at least 6145 input tokens\"}}"
            )
        return '{"selected_ids": [], "answer": "ok"}'

    def embed(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=float)


def test_context_overflow_retries_with_a_smaller_output_cap() -> None:
    """A context-length 400 must be absorbed, not kill the benchmark run."""
    backend = _OverflowThenOkBackend()
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
        max_tokens=2048,          # deliberately too large for the window
        context_window=400,       # smaller than prompt + cap, so the cap must shrink
    )

    _, answer = agent.distil_and_answer("what happened?", [_fat_note("n1", 4000)])

    assert answer == "ok"
    assert len(backend.caps) == 2, "must retry once after a context overflow"
    assert backend.caps[0] == 2048
    assert backend.caps[1] is not None and backend.caps[1] < 2048


# ---------------------------------------------------------------------------
# Retrieval context format (numbered blocks instead of a JSON payload)
# ---------------------------------------------------------------------------

_UID = "7b09e5df-5cd4-441c-b912-8f181b4868fe"


def _plain_note(nid: str, *, x: str = "A fact.", c: str = "[Caroline] hello", links=None):
    note = _note(nid, links=links)
    note.X = x
    note.c = c
    return note


def test_render_notes_block_numbers_notes_and_hides_ids() -> None:
    """A UUID is a 36-char copy target for a small model; ranks are not."""
    block = _render_notes_block([_plain_note(_UID, x="Calvin got advice from a producer.")])

    assert block.startswith("[1] ")
    assert _UID not in block, "ids must never reach the prompt"
    assert "Calvin got advice from a producer." in block


def test_render_notes_block_shows_speaker_date_and_topics() -> None:
    note = _plain_note("n1", x="Caroline moved to Sweden.")
    note.session_date = "2023-04-20T16:15:00Z"
    note.speaker = "Caroline"
    note.entities = ["Caroline", "Sweden"]
    note.K = [f"kw{i}" for i in range(20)]

    block = _render_notes_block([note])

    assert "20 April 2023" in block, "ISO timestamps must be humanised"
    assert "said by Caroline" in block
    assert "about: Caroline, Sweden" in block
    assert block.count("kw") <= 8, "keyword list is capped"


def test_relation_targets_render_as_rank_numbers() -> None:
    a = _plain_note("node-alpha", links=[LinkRecord(target_id="node-beta", relation="contradicts")])
    b = _plain_note("node-beta")

    block = _render_notes_block([a, b])

    assert "links: [2] contradicts" in block
    assert "node-beta" not in block, "the neighbour is referred to by rank, not by id"


def test_edges_outside_the_block_collapse_to_counts() -> None:
    a = _plain_note("node-alpha", links=[LinkRecord(target_id="not-here", relation="same-topic")])

    block = _render_notes_block([a])

    assert "not-here" not in block
    assert "same-topic:1" in block and "not in this list" in block


def test_long_description_becomes_capped_bullets() -> None:
    """Evolved notes reach ~1,700 chars; bullets + a cap keep them readable."""
    note = _plain_note("n1")
    note.X = "First fact about Alex. Second fact about Alex. Third fact about Alex."

    block = _render_notes_block([note], max_bullets=2)

    assert "First fact about Alex." in block
    assert "Third fact about Alex." not in block
    assert "not shown" in block, "truncation must be visible to the model"


def test_single_huge_sentence_is_char_capped_and_flagged() -> None:
    note = _plain_note("n1", x="word " * 300)

    block = _render_notes_block([note], fact_chars=200, max_bullets=4)

    assert len(block) < 700
    assert "…" in block and "not shown" in block


def test_description_duplicated_in_the_raw_turn_is_printed_once() -> None:
    note = _plain_note("n1", x="Alex works at Google.", c="Alex works at Google.")

    block = _render_notes_block([note], content_chars=200)

    assert block.count("Alex works at Google.") == 1
    assert "turn:" not in block


def test_numeric_selected_ids_map_back_to_the_ranked_notes() -> None:
    """`{"selected_ids": [2]}` must select the SECOND note in the context."""
    backend = _PromptRecordingBackend(answer='{"selected_ids": [2], "answer": "Sweden"}')
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
    )
    notes = [_plain_note("first"), _plain_note("second"), _plain_note("third")]

    kept, answer = agent.distil_and_answer("where?", notes)

    assert answer == "Sweden"
    assert [n.id for n in kept] == ["second"]
    assert "[2]" in backend.prompts[0], "the rank the model selected is printed in the context"


def test_id_shaped_selected_ids_still_work() -> None:
    """Back-compat: an older prompt (or caller) may return ids instead of ranks."""
    backend = _PromptRecordingBackend(answer='{"selected_ids": ["second"], "answer": "ok"}')
    agent = AnswerAgent(
        backend=backend,
        prompt_template="AA:{query} {candidates}",
        baseline_prompt_template="BASE:{query} {context}",
    )
    notes = [_plain_note("first"), _plain_note("second")]

    kept, _ = agent.distil_and_answer("where?", notes)

    assert [n.id for n in kept] == ["second"]


def test_distil_prompt_file_formats_with_the_new_placeholders() -> None:
    """P_distil.txt is a `.format()` template: a stray brace breaks every run."""
    from pathlib import Path

    template = (Path(__file__).resolve().parent.parent / "data" / "prompts" / "P_distil.txt").read_text(encoding="utf-8")
    rendered = template.format(query="Q?", candidates="[1] 8 May 2023 · said by Alex")

    assert '"selected_ids"' in rendered
    assert "{query}" not in rendered and "{candidates}" not in rendered
