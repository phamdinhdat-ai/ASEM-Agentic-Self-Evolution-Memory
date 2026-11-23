"""Query-slot proof and nearby counterevidence contracts."""

import json

from asem.care import SourceTurn
from asem.care.relation_bridge import CareRelationBridge
from asem.care.relation_proof import RelationProofGuard, compile_query


def turn(sid, speaker, text, ordinal, session="s1"):
    return SourceTurn(sid, session, "1 January 2024", speaker, text, ordinal)


def test_compiler_keeps_explicit_constraints_only():
    frame = compile_query(
        "What was one of Deborah's favorite games to play with her mom on the PlayStation game system?",
        {"deborah", "jolene"})
    assert frame.subject == "deborah"
    assert [(slot.kind, slot.value) for slot in frame.slots] == [
        ("played_with", "mother"), ("played_on", "playstation")]
    assert compile_query("What does Deborah like?", {"deborah"}) is None


def test_pet_name_cannot_be_reassigned_to_a_different_species():
    turns = [turn("D1", "James", "My dog came to me today.", 1),
             turn("D2", "John", "Your pup is cute, what's his name?", 2),
             turn("D3", "James", "His name's Ned since I adopted him.", 3)]
    proof = RelationProofGuard(turns).check(
        "What is the name of the kitten adopted by James?", "Ned", [turns[2]])
    assert proof.status == "contradicted"
    assert proof.missing == ("adopted_pet_type",)
    assert proof.counterevidence_ids == ("D1",)


def test_companion_antecedent_and_platform_must_be_bound():
    turns = [turn("D1", "Deborah", "I used to play games with my husband.", 1),
             turn("D2", "Jolene", "What game did you play?", 2),
             turn("D3", "Deborah", "We prefer detective games together.", 3)]
    proof = RelationProofGuard(turns).check(
        "Which game did Deborah play with her mom on the PlayStation game system?",
        "detective games", [turns[2]])
    assert proof.status == "contradicted"
    assert set(proof.missing) == {"played_with", "played_on"}
    assert proof.counterevidence_ids == ("D1",)


def test_vocative_and_venue_do_not_prove_that_addressee_saw_something_there():
    cited = turn("D1", "Melanie",
                 "Caroline, the cafe had a serious sign. [image caption: a sign on a door]", 1)
    caroline = turn("D2", "Caroline", "That sign looks serious.", 2)
    proof = RelationProofGuard([cited, caroline]).check(
        "What precautionary sign did Caroline see at the cafe?", "a sign on a door", [cited])
    assert proof.status == "incomplete"
    assert proof.missing == ("observed_at",)


def test_direct_source_covers_slots_without_using_uncited_neighbors():
    cited = turn("D1", "James", "I adopted a kitten named Ned.", 1)
    assert RelationProofGuard([cited]).check(
        "What is the name of the kitten adopted by James?", "Ned", [cited]).status == "covered"
    missing = turn("D2", "James", "I adopted Ned.", 2)
    assert RelationProofGuard([cited, missing]).check(
        "What is the name of the kitten adopted by James?", "Ned", [missing]).status == "incomplete"
    mixed = turn("D3", "James", "I adopted a kitten. My dog Ned is playful.", 3)
    assert RelationProofGuard([mixed]).check(
        "What is the name of the kitten adopted by James?", "Ned", [mixed]).status == "incomplete"


def test_qualifier_can_be_proved_separately_when_answer_is_a_location():
    source = turn("D1", "James", "I adopted a pup from a shelter in Stamford.", 1)
    proof = RelationProofGuard([source]).check(
        "In which state is the shelter from which James adopted the puppy?",
        "Connecticut", [source])
    assert proof.status == "covered"


def test_bridge_retries_incomplete_proof_with_two_call_cap():
    turns = [turn("D1", "James", "His name's Ned since I adopted him.", 0),
             turn("D2", "James", "I adopted a kitten named Ned.", 1)]

    class Backend:
        def __init__(self):
            self.calls = 0
        def generate(self, prompt, **kwargs):
            self.calls += 1
            sid = "D1" if self.calls == 1 else "D2"
            return json.dumps({"answer": "Ned", "source_ids": [sid], "search_queries": []})

    backend = Backend()
    result = CareRelationBridge(turns).answer(
        "What is the name of the kitten adopted by James?", backend)
    assert result["answer"] == "Ned"
    assert result["first_pass"]["invalid_reason"] == "relation_incomplete"
    assert [part["status"] for part in result["relation_proof_trace"]] == ["incomplete", "covered"]
    assert backend.calls == 2
