"""W10 — retrieval-context builder: provenance, budget contract, question awareness.

LLM-free and bank-free by construction: every note is built in-code and no
backend is ever called. Covers the four pieces of evidence task-2 requires:

1. legacy vs thg_v1 rendered from the SAME note list, with measured sizes;
2. the byte-for-byte legacy guarantee (a pinned golden string);
3. the budget contract — the rendered prompt never exceeds the budget it was
   fitted to, proven on a pathological note set;
4. one test per question kind asserting the right structural marker is/is not
   present.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from asem.answer_agent import (
    AnswerAgent,
    RenderStats,
    _render_chains,
    _render_notes_block,
    _session_label,
    detect_question_kind,
)
from asem.note import LinkRecord, Note
from asem.token_budget import estimate_tokens, fit_items, fit_render_exact, fits


# ---------------------------------------------------------------------------
# helpers — notes are built in-code; no bank, no embedder, no API
# ---------------------------------------------------------------------------
def mk(
    nid: str,
    x: str = "A fact.",
    c: str = "",
    *,
    day: int = 0,
    sid: str | None = "session_1",
    speaker: str | None = None,
    entities=None,
    links=None,
    K=None,
    session_date: str | None = None,
) -> Note:
    t = datetime(2023, 4, 1) + timedelta(days=day)
    return Note(
        id=nid,
        c=c or x,
        t=t,
        K=list(K or []),
        G=[],
        X=x,
        e=None,
        L=list(links or []),
        z=None,
        q=0.5,
        session_id=sid,
        session_date=session_date or t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        entities=list(entities or []),
        speaker=speaker,
    )


class _RecordingBackend:
    """Captures prompts; never contacts a network."""

    default_max_tokens = None

    def __init__(self, answer: str = "ok"):
        self.answer = answer
        self.prompts: list = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.answer

    def embed(self, text: str):  # pragma: no cover - never used here
        raise AssertionError("no embedding in context tests")


def _agent(**kw) -> AnswerAgent:
    base = dict(
        backend=_RecordingBackend(),
        prompt_template="{query}\n\nCONTEXT:\n{context}",
        baseline_prompt_template="{query}\n\nCONTEXT:\n{context}",
    )
    base.update(kw)
    return AnswerAgent(**base)


# ===========================================================================
# EVIDENCE 1 — legacy vs thg_v1 on the SAME note list, measured
# ===========================================================================
def _corpus():
    return [
        mk("a", "Caroline attended an LGBTQ support group.", day=0, sid="session_1",
           speaker="Caroline", entities=["Caroline", "LGBTQ support group"],
           links=[LinkRecord(target_id="b", relation="temporal")]),
        mk("b", "Melanie signed up for a pottery class.", day=30, sid="session_2",
           speaker="Melanie", entities=["Melanie", "pottery"],
           links=[LinkRecord(target_id="a", relation="same-entity")]),
        mk("c", "Caroline lives in Boston.", day=100, sid="session_3",
           entities=["Caroline", "Boston"],
           links=[LinkRecord(target_id="d", relation="superseded_by",
                             valid_to="2023-07-13T09:00:00Z")]),
        mk("d", "Caroline lives in Seattle.", day=102, sid="session_4",
           entities=["Caroline", "Seattle"],
           links=[LinkRecord(target_id="c", relation="superseded_by",
                             valid_from="2023-07-13T09:00:00Z")]),
    ]


def test_legacy_and_thg_v1_render_the_same_notes_differently() -> None:
    """Evidence 1: same input, two contexts, both measured."""
    notes = _corpus()
    legacy = _render_notes_block(notes, context_style="legacy")
    thg = _render_notes_block(notes, context_style="thg_v1",
                              question_kind="multi_hop")

    assert legacy != thg, "thg_v1 must add structure"
    # The evidence text survives; thg_v1 only inserts structural markers around
    # it (the session label goes between the date and the speaker).
    for note in notes:
        assert note.X in thg
    # Every legacy header's date and speaker are still present in thg_v1.
    for line in legacy.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("Memory "):
            continue
        # header is "Memory N — recorded on DD Month YYYY..."
        tail = stripped.split(" — recorded on ")[1]
        date = " ".join(tail.split(" ")[:3]).rstrip(",")
        assert date in thg, f"date {date!r} lost in thg_v1"

    # and the sizes are measurable, which is what the efficiency table needs
    assert len(legacy) > 0 and len(thg) > len(legacy)
    assert estimate_tokens(thg) >= estimate_tokens(legacy)


def test_legacy_is_the_default_so_nothing_changes_by_accident() -> None:
    notes = _corpus()
    assert _render_notes_block(notes) == _render_notes_block(
        notes, context_style="legacy"
    )
    agent = _agent()
    assert agent.context_style == "legacy"
    assert agent._thg_context is False


# ===========================================================================
# EVIDENCE 2 (acceptance) — `legacy` reproduces the pre-W10 bytes exactly
# ===========================================================================
# This is the GOLDEN pin. It is the verbatim output of the pre-W10 renderer for
# `_corpus()` (captured before the W10 edit; see .dsh_tmp/legacy_proof.txt). If
# any future change to the renderer moves a legacy byte, this test fails.
_GOLDEN_LEGACY = """\
Memory 1 — recorded on 1 April 2023, said by Caroline, about Caroline, LGBTQ support group.
      • Caroline attended an LGBTQ support group.
      Connections: is from the same session as Memory 2.

Memory 2 — recorded on 1 May 2023, said by Melanie, about Melanie, pottery.
      • Melanie signed up for a pottery class.
      Connections: shares the same entity (Melanie) with Memory 1.

Memory 3 — recorded on 10 July 2023 (true until 13 July 2023, superseded by Memory 4), about Caroline, Boston.
      • Caroline lives in Boston.
      Connections: is replaced by a later version of this fact in Memory 4.

Memory 4 — recorded on 12 July 2023 (true from 13 July 2023), about Caroline, Seattle.
      • Caroline lives in Seattle.
      Connections: replaces an earlier version of this fact in Memory 3."""


def test_legacy_rendering_is_pinned_and_unchanged() -> None:
    """The acceptance criterion: legacy == today's bytes, pinned as a golden.

    A golden-file style pin is used (rather than a diff against git) because the
    W1-W9 work is uncommitted, so `git show HEAD:` is the PRE-W1 state and is NOT
    a valid baseline for this file.
    """
    produced = _render_notes_block(_corpus(), context_style="legacy")
    # Compare line-by-line for a readable failure message.
    assert produced.split("\n") == _GOLDEN_LEGACY.split("\n"), (
        "the legacy rendering moved — frozen-bank runs are no longer reproducible"
    )


def test_empty_note_list_is_stable_in_both_styles() -> None:
    assert _render_notes_block([]) == "(no memory notes retrieved)"
    assert _render_notes_block([], context_style="thg_v1") == (
        "(no memory notes retrieved)"
    )


def test_thg_v1_never_alters_the_legacy_content_words() -> None:
    """thg_v1 adds structure; it must not rewrite the evidence itself."""
    notes = _corpus()
    legacy = _render_notes_block(notes, context_style="legacy")
    thg = _render_notes_block(notes, context_style="thg_v1")
    # Every rendered fact still appears verbatim in the thg_v1 context.
    for note in notes:
        assert note.X in thg


# ===========================================================================
# W10.1 — provenance and graph-as-graph
# ===========================================================================
def test_session_label_appears_in_the_thg_header() -> None:
    """W10.1: `session_id` is the only session provenance, and was invisible."""
    notes = [mk("a", "Caroline attended a support group.", sid="session_3",
                entities=["Caroline"])]
    legacy = _render_notes_block(notes, context_style="legacy")
    thg = _render_notes_block(notes, context_style="thg_v1")

    assert "session_3" not in legacy, "pre-W10 the session id was rendered nowhere"
    assert "[session: session_3]" in thg


def test_long_session_id_is_reduced_to_a_short_ordinal() -> None:
    """The budget guard: a long id must not be paid for verbatim."""
    long_id = "locomo_0042_a_very_long_descriptive_identifier"
    note = mk("a", "A fact.", sid=long_id)
    label = _session_label(note)
    assert label == "session 42"
    assert len(label) < len(long_id)


def test_missing_session_id_degrades_silently() -> None:
    """A bank that predates session ids must not gain a bogus placeholder."""
    note = mk("a", "A fact.", sid=None)
    assert _session_label(note) == ""
    thg = _render_notes_block([note], context_style="thg_v1")
    assert "[session:" not in thg
    assert "None" not in thg


def test_evidence_chains_render_the_bridge_entity() -> None:
    """W10.1/ADDITION C: the model must see WHY two memories connect."""
    a = mk("a", "Melanie told Caroline about it.", sid="session_1",
           entities=["Melanie", "Caroline"],
           links=[LinkRecord(target_id="b", relation="same-entity")])
    b = mk("b", "Caroline liked the pottery class.", sid="session_2",
           entities=["Caroline", "pottery"],
           links=[LinkRecord(target_id="a", relation="same-entity")])

    legacy = _render_notes_block([a, b], context_style="legacy")
    thg = _render_notes_block([a, b], context_style="thg_v1",
                              question_kind="multi_hop")

    assert "Evidence chains" not in legacy
    assert "Evidence chains" in thg
    assert "Memory 1 → Memory 2" in thg
    assert "via Caroline" in thg


def test_chains_degrade_to_nothing_when_note_L_is_empty() -> None:
    """A bank loaded without a graph object must not emit a fake bridge."""
    notes = [mk("a", "A fact."), mk("b", "Another fact.")]
    chains = _render_chains(notes, {"a": 1, "b": 2})
    assert chains == []
    thg = _render_notes_block(notes, context_style="thg_v1", question_kind="multi_hop")
    assert "Evidence chains" not in thg
    assert "reachable via unknown" not in thg


def test_each_chain_pair_is_rendered_once() -> None:
    """The bridge is stated once, not twice (once from each endpoint)."""
    a = mk("a", "Fact one.", entities=["X"],
           links=[LinkRecord(target_id="b", relation="same-entity")])
    b = mk("b", "Fact two.", entities=["X"],
           links=[LinkRecord(target_id="a", relation="same-entity")])
    chains = _render_chains([a, b], {"a": 1, "b": 2})
    assert len(chains) == 1


# ===========================================================================
# W10.2 — the budget contract
# ===========================================================================
def test_rendered_prompt_never_exceeds_the_fitted_budget_pathological() -> None:
    """Evidence 3: the budget contract on a pathological note set.

    The pathological set is chosen to defeat *every* non-contractual trimming
    strategy at once: notes with no spaces (no word boundary to wrap on), huge
    single notes, and a count that forces the tail-drop path.
    """
    notes = [mk(f"n{i}", "X" * 4000, c="Y" * 4000) for i in range(40)]
    for budget in (1, 2, 5, 64, 500, 2000, 20000):
        agent = _agent(
            context_window=budget + 512 + 64,   # so prompt_budget == budget
            max_tokens=512,
        )
        assert agent.prompt_budget() == budget, "budget setup must be exact"
        render = lambda ns: agent.baseline_prompt_template.format(
            query="a very long question " * 50,
            context=_render_notes_block(ns, context_style="thg_v1"),
        )
        prompt, kept = agent._fit(notes, render)
        assert fits(prompt, budget), (
            f"budget {budget} exceeded: {estimate_tokens(prompt)} tokens "
            f"({len(prompt)} chars), kept={len(kept)}"
        )


def test_budget_contract_holds_when_scaffolding_alone_overflows() -> None:
    """The case `fit_items` alone cannot satisfy: budget < fixed scaffolding.

    Prompt-editing a template whose instruction block is longer than the entire
    budget. `fit_items` would log and return it anyway; `fit_render_exact` must
    truncate it.
    """
    huge_instruction = "INSTRUCTION. " * 400
    render = lambda ns: huge_instruction + _render_notes_block(ns)
    for budget in (1, 10, 100, 1000):
        prompt, kept = fit_render_exact(
            [mk("a", "A fact.")], render, budget=budget,
            fallback="minimal prompt",
        )
        assert fits(prompt, budget), (
            f"budget {budget}: got {estimate_tokens(prompt)} tokens"
        )


def test_fit_render_exact_truncates_a_non_shrinkable_render() -> None:
    """A render whose scaffolding cannot be trimmed is cut, not returned over budget."""
    render = lambda ns: "Z" * 100000
    prompt, kept = fit_render_exact([mk("a", "x")], render, budget=1, fallback="bare")
    assert fits(prompt, 1)
    assert prompt != "Z" * 100000, "it must actually have been truncated"


def test_fit_render_exact_falls_back_when_truncation_still_cannot_fit() -> None:
    """The true last resort: a budget so small that even a 1-char prompt is too big.

    `estimate_tokens` floors at 1, so budget=0 is a budget no non-empty string can
    satisfy. The fallback is then the only correct answer.
    """
    render = lambda ns: "Z" * 100000
    prompt, kept = fit_render_exact([mk("a", "x")], render, budget=0, fallback="")
    assert prompt == ""
    assert kept == []


def test_fit_render_exact_is_a_noop_without_a_budget() -> None:
    notes = [mk("a", "A fact.")]
    render = lambda ns: _render_notes_block(ns)
    prompt, kept = fit_render_exact(notes, render, budget=None, fallback="bare")
    assert prompt == _render_notes_block(notes)
    assert len(kept) == 1


def test_clip_block_never_grows_so_the_shrink_loop_terminates() -> None:
    from asem.token_budget import clip_block
    for text in ("x" * 1000, "word " * 200, "a" * 25):
        for keep in ("head", "tail"):
            smaller = clip_block(text, keep=keep)
            if smaller is not None:
                assert len(smaller) < len(text)


def test_direct_answer_respects_the_budget_end_to_end() -> None:
    """The production path, not just the renderer, honours the contract."""
    backend = _RecordingBackend(answer="answer")
    agent = AnswerAgent(
        backend=backend,
        prompt_template="{query} {candidates}",
        baseline_prompt_template="{query}\n\n{context}",
        direct_mode=True,
        context_window=4096,
        max_tokens=512,
    )
    notes = [mk(f"n{i}", "F" * 3000) for i in range(20)]
    agent.direct_answer("When did it happen?", notes)
    assert backend.prompts, "the backend must have been called"
    budget = agent.prompt_budget()
    assert fits(backend.prompts[0], budget), (
        f"{estimate_tokens(backend.prompts[0])} > {budget}"
    )


def test_drop_counters_report_what_was_removed() -> None:
    """W10.2: the block header hinted at truncation but nothing counted it."""
    long_x = "One. Two. Three. Four. Five."
    st = RenderStats()
    _render_notes_block([mk("a", long_x)], max_bullets=2, stats=st,
                        context_style="thg_v1")

    assert st.notes_rendered == 1
    assert st.sentences_total == 5
    assert st.sentences_shown == 2
    assert st.dropped_sentences == 3
    assert st.dropped_chars > 0
    assert set(st.as_dict()) >= {"dropped_sentences", "dropped_chars"}


def test_drop_counters_are_zero_for_a_short_note() -> None:
    st = RenderStats()
    _render_notes_block([mk("a", "Just one short fact.")], stats=st)
    assert st.dropped_sentences == 0
    assert st.dropped_chars == 0
    assert st.sentences_shown == st.sentences_total == 1


def test_agent_exposes_the_counters_from_a_real_render() -> None:
    agent = _agent(context_style="thg_v1", max_context_notes=4)
    notes = [mk(f"n{i}", f"Fact {i} one. Fact {i} two. Fact {i} three.")
             for i in range(4)]
    agent._render_with_stats(notes, query="what?")
    assert agent.last_render_stats is not None
    assert agent.last_render_stats.notes_rendered == 4


# ===========================================================================
# W10.3 — question-aware context, one test per kind
# ===========================================================================
@pytest.mark.parametrize("query,expected", [
    ("How many days between the mass and Ash Wednesday?", "arithmetic"),
    ("How long after the move did she start the job?", "arithmetic"),
    ("How many days before the deadline did it happen?", "arithmetic"),
    ("Which event did I attend first?", "ordering"),
    ("In what order did these happen?", "ordering"),
    ("Which came first, the bike or the car?", "ordering"),
    ("Where does Caroline live now?", "knowledge_update"),
    ("What is the current status of the project?", "knowledge_update"),
    ("Did she use to live in Boston?", "knowledge_update"),
    ("How many kids does Melanie have?", "count"),
    ("Why did Caroline switch careers?", "multi_hop"),
    ("Who introduced Caroline to the group?", "multi_hop"),
    ("What did Caroline research?", "single_fact"),
    ("", "single_fact"),
])
def test_question_kind_detection(query: str, expected: str) -> None:
    assert detect_question_kind(query) == expected


def test_kind_detection_is_deterministic() -> None:
    q = "How many days between the two events?"
    assert len({detect_question_kind(q) for _ in range(20)}) == 1


def test_arithmetic_question_gets_pre_extracted_date_anchors() -> None:
    """W10.3: the operands are listed so the model need not hunt for them."""
    notes = [
        mk("a", "Sunday mass was held.", day=0, sid="session_1",
           entities=["Church"]),
        mk("b", "Ash Wednesday happened.", day=30, sid="session_2",
           entities=["Church"]),
        mk("c", "An unrelated dated event.", day=45, sid="session_3"),
    ]
    legacy = _render_notes_block(notes, context_style="legacy")
    thg = _render_notes_block(notes, context_style="thg_v1",
                              question_kind="arithmetic")

    assert "Date anchors" not in legacy
    assert "Date anchors" in thg
    # chronological order, earliest first
    body = thg.split("Date anchors")[1]
    assert body.index("1 April 2023") < body.index("1 May 2023") < body.index("16 May 2023")


def test_arithmetic_with_a_single_distinct_date_omits_the_anchor_line() -> None:
    """One date is not an arithmetic operand pair; the line must not mislead."""
    notes = [mk("a", "One event.", day=0), mk("b", "Same day event.", day=0)]
    thg = _render_notes_block(notes, context_style="thg_v1",
                              question_kind="arithmetic")
    assert "Date anchors" not in thg


def test_ordering_question_gets_explicit_positions() -> None:
    """W10.3: position is stated, not left to be inferred from print order."""
    notes = [mk(f"n{i}", f"Event {i}.", day=i * 10) for i in range(4)]
    legacy = _render_notes_block(notes, context_style="legacy")
    thg = _render_notes_block(notes, context_style="thg_v1",
                              question_kind="ordering")

    assert "position" not in legacy
    assert "(position 1 of 4)" in thg
    assert "(position 4 of 4)" in thg


def test_knowledge_update_puts_the_current_fact_first() -> None:
    """W10.3: the replacement is rendered before the fact it replaced."""
    old = mk("old", "Caroline lives in Boston.", day=0,
             links=[LinkRecord(target_id="new", relation="superseded_by",
                               valid_to="2023-04-03T09:00:00Z")])
    new = mk("new", "Caroline lives in Seattle.", day=2,
             links=[LinkRecord(target_id="old", relation="superseded_by",
                               valid_from="2023-04-03T09:00:00Z")])

    legacy = _render_notes_block([old, new], context_style="legacy")
    thg = _render_notes_block([old, new], context_style="thg_v1",
                              question_kind="knowledge_update")

    # legacy prints the input order, so the stale fact comes first
    assert legacy.index("Boston") < legacy.index("Seattle")
    # thg_v1 reorders so the CURRENT fact is read first
    assert thg.index("Seattle") < thg.index("Boston")
    assert "superseded" in thg


def test_knowledge_update_marker_is_absent_for_the_generic_kind() -> None:
    """The reordering is gated on the question kind, so nothing else moves."""
    old = mk("old", "Caroline lives in Boston.", day=0,
             links=[LinkRecord(target_id="new", relation="superseded_by",
                               valid_to="2023-04-03T09:00:00Z")])
    new = mk("new", "Caroline lives in Seattle.", day=2,
             links=[LinkRecord(target_id="old", relation="superseded_by",
                               valid_from="2023-04-03T09:00:00Z")])
    thg = _render_notes_block([old, new], context_style="thg_v1",
                              question_kind="single_fact")
    assert thg.index("Boston") < thg.index("Seattle")


def test_knowledge_update_reorder_keeps_connection_targets_valid() -> None:
    """A positional rank must still point at the right memory after reordering."""
    old = mk("old", "Caroline lives in Boston.", day=0,
             links=[LinkRecord(target_id="new", relation="superseded_by",
                               valid_to="2023-04-03T09:00:00Z")])
    new = mk("new", "Caroline lives in Seattle.", day=2,
             links=[LinkRecord(target_id="old", relation="superseded_by",
                               valid_from="2023-04-03T09:00:00Z")])
    thg = _render_notes_block([old, new], context_style="thg_v1",
                              question_kind="knowledge_update")
    # Memory 1 is now 'new'; the reverse pointer from it must say Memory 2.
    first_block = thg.split("\n\n")[0]
    assert first_block.startswith("Memory 1")
    assert "Memory 2" in first_block


def test_multi_hop_question_gets_chains_but_single_fact_still_does() -> None:
    """Chains are a W10.1 provenance feature, not gated on the question kind."""
    a = mk("a", "Melanie told Caroline.", entities=["Melanie", "Caroline"],
           links=[LinkRecord(target_id="b", relation="same-entity")])
    b = mk("b", "Caroline liked it.", entities=["Caroline", "pottery"],
           links=[LinkRecord(target_id="a", relation="same-entity")])
    thg = _render_notes_block([a, b], context_style="thg_v1",
                              question_kind="single_fact")
    assert "Evidence chains" in thg


def test_question_kind_markers_are_absent_from_legacy_for_every_kind() -> None:
    """The whole W10.3 surface must be off in legacy, for every kind."""
    notes = _corpus()
    legacy = _render_notes_block(notes, context_style="legacy")
    for marker in ("Date anchors", "Evidence chains", "(position ", "[session:"):
        assert marker not in legacy, f"{marker!r} leaked into the legacy context"


def test_context_style_is_case_insensitive_and_validated_by_behaviour() -> None:
    notes = _corpus()
    for value in ("thg_v1", "THG_V1", "Thg_v1"):
        assert _render_notes_block(notes, context_style=value) == (
            _render_notes_block(notes, context_style="thg_v1")
        )
    for value in ("legacy", "LEGACY", "", None):
        assert _render_notes_block(notes, context_style=value) == (
            _render_notes_block(notes, context_style="legacy")
        )


def test_prompt_template_integration_uses_the_configured_style() -> None:
    """End to end: the style reaches the prompt actually sent."""
    q = "How many days between the mass and Ash Wednesday?"
    notes = [mk("a", "Mass was held.", day=0, entities=["Church"]),
             mk("b", "Ash Wednesday happened.", day=30, entities=["Church"])]

    legacy_backend = _RecordingBackend()
    _agent_legacy = AnswerAgent(
        backend=legacy_backend,
        prompt_template="{query} {candidates}",
        baseline_prompt_template="{query}\n\n{context}",
        direct_mode=True,
        context_style="legacy",
    )
    _agent_legacy.direct_answer(q, notes)

    thg_backend = _RecordingBackend()
    thg_agent = AnswerAgent(
        backend=thg_backend,
        prompt_template="{query} {candidates}",
        baseline_prompt_template="{query}\n\n{context}",
        direct_mode=True,
        context_style="thg_v1",
    )
    thg_agent.direct_answer(q, notes)

    assert "Date anchors" not in legacy_backend.prompts[0]
    assert "Date anchors" in thg_backend.prompts[0]
    assert "[session:" not in legacy_backend.prompts[0]
    assert "[session:" in thg_backend.prompts[0]


def test_question_kind_is_recorded_on_the_agent() -> None:
    agent = _agent(context_style="thg_v1")
    agent._render_with_stats([mk("a", "A fact.")], query="Where does she live now?")
    assert agent.last_question_kind == "knowledge_update"


def test_temporal_prompt_file_still_formats() -> None:
    """P_temporal_qa.txt is a `.format()` template: a stray brace breaks a run."""
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "data" / "prompts" / "P_temporal_qa.txt"
    template = path.read_text(encoding="utf-8")
    rendered = template.format(query="Q?", context="CTX")
    assert "Q?" in rendered and "CTX" in rendered
    # The new markers must be documented for the model.
    for marker in ("session:", "Evidence chains", "Date anchors", "position"):
        assert marker in template, f"{marker} must be explained in the prompt"


def test_config_flag_is_present_and_defaults_to_legacy() -> None:
    """The flag must exist in the config and must default to legacy."""
    from pathlib import Path
    import re
    cfg = Path(__file__).resolve().parent.parent / "configs" / "models" / "deepseek_v41_flash.yaml"
    text = cfg.read_text(encoding="utf-8")
    m = re.search(r"^\s*context_style:\s*[\"']?(\w+)[\"']?\s*$", text, re.M)
    assert m, "answer.context_style must be settable from the config"
    assert m.group(1) == "legacy", "the default must reproduce today's bytes"
