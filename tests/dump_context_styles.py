"""W10 EVIDENCE 1 — legacy vs thg_v1 side-by-side, measured.

Renders the SAME in-code note list under both styles and reports chars/tokens.
No bank, no API. Run:

  $py="C:\\Users\\Dat Pham\\anaconda3\\envs\\memory-r1\\python.exe"
  $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONPATH="."
  & $py tests/dump_context_styles.py

CONTENT ASSERTIONS — READ BEFORE ADDING ONE
-------------------------------------------
Any check of the form "is this text present in the rendered block?" MUST go
through the helpers in `tests/assert_content.py`, e.g.

    from tests.assert_content import contains, assert_contains, assert_absent

    present = contains(block, "some fact")     # wrap-safe
    assert_contains(block, "some fact")        # raises with provenance label

Do NOT write `"some fact" in block`. `_render_notes_block` calls `_wrap`, which
inserts a newline plus an indent MID-SENTENCE, so a raw substring check is False
for any text long enough to wrap. It passes only while a fixture happens to be
short — i.e. it is green for the wrong reason and stops testing anything the
moment someone lengthens the fixture. See tests/assert_content.py for details.
"""
from __future__ import annotations

import io
import sys
from datetime import datetime, timedelta

sys.path.insert(0, ".")

from asem.answer_agent import (
    AnswerAgent,
    RenderStats,
    _render_notes_block,
    detect_question_kind,
)
from asem.note import LinkRecord, Note
from asem.token_budget import estimate_tokens
from tests.assert_content import assert_absent, assert_contains, contains

OUT = io.StringIO()


def p(*a):
    print(*a, file=OUT)


def note(nid, x, c="", *, day=0, sid="session_1", speaker=None, entities=None,
         links=None, K=None, valid_to=None, valid_from=None):
    """Build a note in-code (no bank, no API)."""
    t = datetime(2023, 4, 1) + timedelta(days=day)
    return Note(
        id=nid, c=c or x, t=t, K=list(K or []), G=[], X=x, e=None,
        L=list(links or []), z=None, q=0.5,
        session_id=sid,
        session_date=t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        entities=list(entities or []),
        speaker=speaker,
    )


def stat_line(label, text):
    p("  %-8s chars=%5d  tokens~%5d" % (label, len(text), estimate_tokens(text)))


# ---------------------------------------------------------------------------
# Scenario A: 6-session haystack, provenance is the question
# ---------------------------------------------------------------------------
p("=" * 78)
p("SCENARIO A — 6-session haystack; 'which session did X happen in'")
p("=" * 78)
sessions = [
    ("session_1", "Caroline", ["Caroline", "adoption agency"], "Caroline started looking into adoption."),
    ("session_2", "Melanie", ["Melanie", "pottery class"], "Melanie signed up for a pottery class."),
    ("session_3", "Caroline", ["Caroline", "LGBTQ support group"], "Caroline attended an LGBTQ support group."),
    ("session_4", "Melanie", ["Melanie", "dinosaur museum"], "Melanie took her kids to a dinosaur museum."),
    ("session_5", "Caroline", ["Caroline", "counseling"], "Caroline decided to pursue counseling."),
    ("session_6", "Melanie", ["Melanie", "nature reserve"], "Melanie's kids like the nature reserve."),
]
notes_a = []
for i, (sid, spk, ents, fact) in enumerate(sessions):
    notes_a.append(note(
        f"n{i}", fact, f"[{spk}] {fact}", day=i * 20, sid=sid, speaker=spk,
        entities=ents, K=["memory", "event", f"topic{i}"],
        links=[LinkRecord(target_id=f"n{(i + 1) % 6}", relation="temporal")],
    ))

q_a = "Which session did Caroline attend the LGBTQ support group in?"
p("question kind: %s" % detect_question_kind(q_a))
legacy_a = _render_notes_block(notes_a, context_style="legacy")
thg_a = _render_notes_block(notes_a, context_style="thg_v1",
                            question_kind=detect_question_kind(q_a))
stat_line("legacy", legacy_a)
stat_line("thg_v1", thg_a)
p("  delta: %+d chars (%+.1f%%)" % (
    len(thg_a) - len(legacy_a),
    100.0 * (len(thg_a) - len(legacy_a)) / max(1, len(legacy_a))))
p("  legacy says 'session': %s   thg_v1 says 'session:': %s"
  % (contains(legacy_a, "session"), contains(thg_a, "session:")))
p("")
p("--- legacy block 1..3 ---")
p("\n".join(legacy_a.split("\n")[:6]))
p("--- thg_v1 block 1..3 ---")
p("\n".join(thg_a.split("\n")[:6]))

# ---------------------------------------------------------------------------
# Scenario B: arithmetic — date anchors
# ---------------------------------------------------------------------------
p("")
p("=" * 78)
p("SCENARIO B — date arithmetic; 'how many days between X and Y'")
p("=" * 78)
notes_b = [
    note("a", "Sunday mass was held.", day=0, sid="session_1",
         entities=["Church"], links=[LinkRecord(target_id="b", relation="same-entity")]),
    note("b", "Ash Wednesday service happened.", day=30, sid="session_2",
         entities=["Church"], links=[LinkRecord(target_id="a", relation="same-entity")]),
    note("c", "A third dated event took place.", day=45, sid="session_3", entities=["Other"]),
]
q_b = "How many days between the Sunday mass and Ash Wednesday?"
p("question kind: %s" % detect_question_kind(q_b))
legacy_b = _render_notes_block(notes_b, context_style="legacy")
thg_b = _render_notes_block(notes_b, context_style="thg_v1", question_kind="arithmetic")
stat_line("legacy", legacy_b)
stat_line("thg_v1", thg_b)
p("  has 'Date anchors': legacy=%s thg_v1=%s"
  % (contains(legacy_b, "Date anchors"), contains(thg_b, "Date anchors")))
for line in thg_b.split("\n"):
    if line.startswith("Date anchors"):
        p("  " + line)

# ---------------------------------------------------------------------------
# Scenario C: knowledge update — current fact first + superseded marker
# ---------------------------------------------------------------------------
p("")
p("=" * 78)
p("SCENARIO C — knowledge update; 'where does X live now'")
p("=" * 78)
old = note("old", "Caroline lives in Boston.", day=0, sid="session_1",
           entities=["Caroline", "Boston"],
           links=[LinkRecord(target_id="new", relation="superseded_by",
                             valid_to="2023-04-03T09:00:00Z")])
new = note("new", "Caroline lives in Seattle.", day=2, sid="session_2",
           entities=["Caroline", "Seattle"],
           links=[LinkRecord(target_id="old", relation="superseded_by",
                             valid_from="2023-04-03T09:00:00Z")])
q_c = "Where does Caroline live now?"
p("question kind: %s" % detect_question_kind(q_c))
legacy_c = _render_notes_block([old, new], context_style="legacy")
thg_c = _render_notes_block([old, new], context_style="thg_v1",
                            question_kind="knowledge_update")
stat_line("legacy", legacy_c)
stat_line("thg_v1", thg_c)
p("  legacy  first fact line: %r" % legacy_c.split("\n")[1].strip())
p("  thg_v1  first fact line: %r" % thg_c.split("\n")[1].strip())
p("  superseded marker before the replacement in thg_v1: %s"
  % (thg_c.index("superseded") > thg_c.index("Seattle")))

# ---------------------------------------------------------------------------
# Scenario D: multi-hop — evidence chains
# ---------------------------------------------------------------------------
p("")
p("=" * 78)
p("SCENARIO D — multi-hop; do the bridges render?")
p("=" * 78)
h1 = note("h1", "Melanie told Caroline about the pottery class.", day=0,
          sid="session_1", speaker="Melanie", entities=["Melanie", "Caroline", "pottery"],
          links=[LinkRecord(target_id="h2", relation="same-entity")])
h2 = note("h2", "The pottery class meets on Thursdays.", day=5, sid="session_2",
          speaker="Melanie", entities=["Melanie", "pottery"],
          links=[LinkRecord(target_id="h1", relation="same-entity")])
q_d = "Why did Caroline sign up for the pottery class?"
p("question kind: %s" % detect_question_kind(q_d))
legacy_d = _render_notes_block([h1, h2], context_style="legacy")
thg_d = _render_notes_block([h1, h2], context_style="thg_v1", question_kind="multi_hop")
stat_line("legacy", legacy_d)
stat_line("thg_v1", thg_d)
p("  has 'Evidence chains': legacy=%s thg_v1=%s"
  % (contains(legacy_d, "Evidence chains"), contains(thg_d, "Evidence chains")))
for line in thg_d.split("\n"):
    if line.startswith("Evidence chains"):
        p("  " + line)

# --- degrade case: no links at all (bank loaded without a graph object) -----
nolinks = [note("z1", "A fact.", day=0, sid="session_1"),
           note("z2", "Another fact.", day=1, sid="session_2")]
thg_nl = _render_notes_block(nolinks, context_style="thg_v1", question_kind="multi_hop")
p("  empty note.L -> 'Evidence chains' present: %s (must be False)"
  % contains(thg_nl, "Evidence chains"))

# ---------------------------------------------------------------------------
# Scenario E: drop counters (W10.2)
# ---------------------------------------------------------------------------
p("")
p("=" * 78)
p("SCENARIO E — W10.2 drop counters")
p("=" * 78)
long_x = "First sentence of the fact. Second sentence of the fact. Third sentence of the fact. Fourth sentence of the fact."
st = RenderStats()
blk = _render_notes_block([note("L", long_x)], context_style="thg_v1",
                          max_bullets=2, stats=st)
p("  note.X len=%d, max_bullets=2" % len(long_x))
p("  counters: %s" % st.as_dict())
p("  block contains the 'more fact(s)' notice: %s"
  % contains(blk, "more fact(s)"))

# ---------------------------------------------------------------------------
# Scenario F: heterogeneous realistic set, both styles (the headline dump)
# ---------------------------------------------------------------------------
p("")
p("=" * 78)
p("SCENARIO F — realistic mixed set (the side-by-side the task asks for)")
p("=" * 78)
mixed = notes_a[:3] + [old, new] + notes_b[:2] + [h1, h2]
p("N notes = %d" % len(mixed))
legacy_f = _render_notes_block(mixed, context_style="legacy")
thg_f = _render_notes_block(mixed, context_style="thg_v1", question_kind="multi_hop")
stat_line("legacy", legacy_f)
stat_line("thg_v1", thg_f)
p("  delta: %+d chars (%+.1f%%), %+d tokens"
  % (len(thg_f) - len(legacy_f),
     100.0 * (len(thg_f) - len(legacy_f)) / len(legacy_f),
     estimate_tokens(thg_f) - estimate_tokens(legacy_f)))
p("")
p("--- FULL thg_v1 rendering ---")
p(thg_f)

with open("tests/context_styles_dump.txt", "w", encoding="utf-8") as fh:
    fh.write(OUT.getvalue())
print("written tests/context_styles_dump.txt")

# Optional: echo to the console for a quick look. The rendered context contains
# U+2192 (the chain arrow) and the console may be cp1252, so encode defensively
# rather than crashing AFTER the report was written.
if "--echo" in sys.argv:
    enc = sys.stdout.encoding or "utf-8"
    sys.stdout.write(OUT.getvalue().encode(enc, errors="replace").decode(enc))
