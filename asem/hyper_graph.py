"""Temporal Hyper-Graph Memory (THG) representation and indexing.

The graph carries four deterministic (LLM-free) relation types:

1. ``same-entity``   -- note <-> note, bridged by the entity nodes they share
2. ``temporal``      -- chronological neighbours inside one session
3. ``semantic``      -- ``cosine(e_i, e_j) >= semantic_tau``
4. ``superseded_by`` -- the same (subject, predicate) seen again in a newer fact

Two views of the same graph are kept side by side:

* ``self._graph`` -- an in-memory ``networkx`` graph with explicit ENTITY nodes,
  which is what :meth:`personalized_pagerank` walks;
* the persisted note-to-note projection on ``Note.L``, which is what the
  retriever traverses and the only view that survives the process boundary
  between the ingest phase and the evaluation phase.

Every relation is written to BOTH views on purpose. A relation that lived only
in the nx graph would be silently lost at eval time -- which is how an earlier
version of this module ended up persisting just two of its four edge types.

All emitted edges are SYMMETRIC. The retriever's link traversal
(``HybridRetriever._traverse_links``) only follows outgoing links from a seed,
so a one-directional edge is invisible whenever the seed is the older endpoint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is a hard dep in practice
    nx = None

from .logging_utils import get_logger
from .note import LinkRecord, Note, TripletRecord

_log = get_logger("THG.graph")

# Relation labels. `temporal` and `same-entity` deliberately reuse the labels the
# deterministic FastASEM weaver already emits, so the shared retriever weights
# apply to both systems.
_REL_ENTITY = "same-entity"
_REL_TEMPORAL = "temporal"
_REL_SEMANTIC = "semantic"
_REL_SUPERSEDED = "superseded_by"


@dataclass
class Triplet:
    """Atomic fact triplet: (subject, predicate, object, timestamp)."""

    subject: str
    predicate: str
    object: str
    timestamp: str


class TemporalHyperGraph:
    """Deterministic Temporal Hyper-Graph for associative memory traversal."""

    def __init__(
        self,
        semantic_tau: float = 0.70,
        max_entity_links: int = 5,
        max_semantic_links: int = 5,
        entity_idf_floor: float = float("-inf"),
        enable_supersession: bool = True,
    ) -> None:
        self.semantic_tau = semantic_tau
        # Fan-out caps. A dominant entity (the conversation's own speaker name,
        # "Caroline") is shared by most notes, so an uncapped entity channel
        # would link every note to every other note and collapse the graph into
        # a clique. Peers are ranked by weighted co-occurrence + similarity
        # before the cut.
        self.max_entity_links = max_entity_links
        self.max_semantic_links = max_semantic_links
        # Hub suppression threshold (GB5). A shared entity contributes its
        # smoothed idf to the peer score; keys whose weight is at or below this
        # value are ignored entirely. The default is -inf = "filter nothing", so
        # the channel never loses peers by default and a hub is only ever
        # DOWN-WEIGHTED (which is what restores selectivity without severing the
        # rare-entity tail). Raise it to drop hubs outright.
        self.entity_idf_floor = entity_idf_floor
        # GB2e: whether the `superseded_by` relation is emitted at all.
        #
        # Defaults to True here so every existing direct caller of
        # `TemporalHyperGraph()` keeps its behaviour; the ASEM-THG SYSTEM passes
        # the config value, whose own default is OFF (`hyper_graph.supersession`
        # in the config, wired in eval/systems.py). The reason is measured, not
        # stylistic: auditing every supersession in the frozen 365-note LoCoMo
        # bank found **0 of 5 pairs correct** — each marked a STILL-TRUE fact
        # obsolete, one of them replacing a richer note with a lossier
        # paraphrase. Because `enhanced_retriever.relation_weights` gives
        # `superseded_by` the highest weight in its table (1.3), a wrong
        # supersession is not merely noise, it actively promotes the wrong
        # evidence and suppresses the right evidence. A relation with no correct
        # result and that failure mode is worse than no relation.
        #
        # Setting this False suppresses ONLY the versioning edge. The W4
        # bi-temporal machinery (`LinkRecord.valid_from`/`valid_to`, interval
        # rendering) is untouched and still applies to any interval-bearing link
        # that does exist, so this is a clean ablation rather than a revert.
        self.enable_supersession = bool(enable_supersession)

        self._graph = nx.Graph() if nx is not None else None
        self._fact_nodes: Dict[str, Note] = {}
        self._entity_nodes: Set[str] = set()
        self._entity_index: Dict[str, Set[str]] = {}  # note_id -> normalised entity keys
        # INVERSE index: entity key -> note ids that carry it. This is what makes
        # `_entity_peers` O(matching peers) instead of O(N): before it existed,
        # every add_note scanned every note in the bank and tested set
        # intersection inline (O(N) per note, O(N^2) per build), which is
        # measurable at bank scale. Maintained by add_note / hydrate / invalidate
        # / clear.
        self._entity_to_notes: Dict[str, Set[str]] = {}
        # (subject, predicate, object) -> latest note id. Exact-triplet index for
        # restatements: the same fact asserted again.
        self._subj_pred_index: Dict[Tuple[str, str, str], str] = {}
        # (subject, predicate) -> latest note id, consulted ONLY for predicates
        # that denote a single-valued attribute ("lives in"), where a new object
        # replaces the old value. See `_is_single_valued_predicate`.
        self._subj_pred_index_slot: Dict[Tuple[str, str], str] = {}
        # Ids of notes whose ``L`` was extended as the *peer* of a link, so the
        # caller can re-persist them (they may pre-date this ingest call).
        self._touched: Set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_note(self, note: Note, triplet: Optional[Triplet] = None) -> List[LinkRecord]:
        """Insert ``note`` and return the links originating from it.

        The returned records are NOT applied to ``note.L`` -- the caller owns
        that, so it can decide the persistence order. Mirror edges pointing back
        at ``note`` are applied in place to the peer notes; collect those with
        :meth:`pop_touched_peers` and re-persist them.
        """
        self._fact_nodes[note.id] = note
        entity_keys = self._entity_keys(note)
        self._entity_index[note.id] = entity_keys
        # Record the triplet ON the note if the caller only passed it here, so
        # the object of the old fact is available to the GB2 object-difference
        # guard below (and to hydrate() when this note is reloaded from the
        # bank). The ingest path already stores it, so this is idempotent there.
        if triplet is not None and getattr(triplet, "subject", "") and triplet.predicate:
            existing = {
                (self._norm(t.subject), self._norm(t.predicate), self._norm(t.object))
                for t in getattr(note, "triplets", None) or []
            }
            key3 = (
                self._norm(triplet.subject),
                self._norm(triplet.predicate),
                self._norm(triplet.object),
            )
            if key3 not in existing:
                note.triplets = list(getattr(note, "triplets", None) or []) + [
                    TripletRecord(
                        subject=triplet.subject,
                        predicate=triplet.predicate,
                        object=triplet.object,
                        recorded_at=triplet.timestamp or None,
                    )
                ]
        # Maintain the inverse index. `setdefault` + add keeps this idempotent,
        # so re-adding a note id (e.g. an UPDATE that re-inserts) cannot leave a
        # stale note id behind in an entity's bucket.
        for key in entity_keys:
            self._entity_to_notes.setdefault(key, set()).add(note.id)

        created_links: List[LinkRecord] = []

        if self._graph is not None:
            self._graph.add_node(note.id, type="fact", date=note.session_date or "")
            for key in sorted(entity_keys):
                self._entity_nodes.add(key)
                self._graph.add_node(key, type="entity")
                self._graph.add_edge(note.id, key, relation="entity_cooccurrence", weight=1.0)

        # 1. same-entity peers (the note-to-note projection of the entity nodes)
        for peer_id, _sim in self._entity_peers(note, entity_keys):
            created_links.append(LinkRecord(target_id=peer_id, relation=_REL_ENTITY))
            self._link_back(peer_id, note.id, _REL_ENTITY)
            if self._graph is not None:
                self._graph.add_edge(note.id, peer_id, relation=_REL_ENTITY, weight=1.0)

        # 2. Deterministic temporal versioning
        #
        # Versioning answers "this fact changed" — the signal behind
        # knowledge-update questions and contradiction resolution. TWO keys are
        # needed, because there are two distinct ways a fact is superseded:
        #
        #   (a) RESTATEMENT — the same (subject, predicate, object) seen again
        #       later. "Caroline lives in Boston" then "Caroline lives in
        #       Boston now" is a re-assertion of one slot.
        #   (b) REPLACEMENT — a SINGLE-VALUED predicate whose object changed.
        #       "lives in Brighton" -> "lives in Seattle": the subject can only
        #       live in one place, so the new value invalidates the old one.
        #
        # Keying on (subject, predicate) alone conflates (b) with unrelated
        # facts that merely share a verb — "bought a car" and "bought a house"
        # are not versions of each other — so the object must discriminate.
        # Keying on (subject, predicate, object) alone misses (b) entirely and
        # versioning stops firing (measured: zero superseded_by edges once the
        # object was added naively).
        #
        # So: an exact key for (a), and a predicate-slot key for (b) restricted
        # to predicates that actually denote a single-valued attribute.
        if triplet and triplet.subject and triplet.predicate:
            subj_key = self._norm(triplet.subject)
            pred_key = self._norm(triplet.predicate)
            obj_key = self._norm(triplet.object)
            exact_key = (subj_key, pred_key, obj_key)
            # The SLOT key uses the canonical slot form, not the raw phrase. Two
            # states of the same subject must share ONE slot while keeping
            # distinct values: `_slot_key_for_predicate` returns `state:happy`
            # vs `state:sad` (distinct slots, so the second is a REPLACEMENT that
            # can supersede) and `state:happy` for both `is happy` and
            # `was happy` (one slot, so a copula change is a re-confirmation).
            slot_key = (subj_key, self._slot_key_for_predicate(pred_key) or pred_key)

            old_id = self._subj_pred_index.get(exact_key)
            replacement = False
            if old_id is None and self._is_single_valued_predicate(pred_key):
                old_id = self._subj_pred_index_slot.get(slot_key)
                replacement = old_id is not None
            # GB2e ablation: when supersession is disabled the indices are still
            # MAINTAINED (below), so flipping the flag back on for a later run
            # behaves identically — only the emitted edge is suppressed.
            if not self.enable_supersession:
                old_id = None
            # A fact must not supersede itself, and an already-known id must
            # still exist in this graph.
            if old_id == note.id:
                old_id = None

            # A REPLACEMENT additionally requires the object to have actually
            # CHANGED. Without this guard the slot key fires on any pair sharing
            # (subject, head-verb): audited on the real bank, `has` produced a
            # supersession between "...has a dog and a cat" and "...recently got
            # another cat named Bailey" — two statements that are simultaneously
            # TRUE. `_objects_are_distinct` also rejects a paraphrase of the SAME
            # value (a restatement, which belongs in the same-entity/semantic
            # channel), and rejects pairs where the object is unspecified,
            # because marking evidence obsolete on missing data is the worse
            # error.
            if old_id and old_id in self._fact_nodes and replacement:
                old_note_for_obj = self._fact_nodes[old_id]
                old_obj = ""
                # Match on the SLOT KEY, not the raw predicate, so a stored
                # "lives in" is still found when the new triplet says "lives".
                want_slot = self._slot_key_for_predicate(pred_key)
                for t in getattr(old_note_for_obj, "triplets", None) or []:
                    if self._slot_key_for_predicate(self._norm(t.predicate)) == want_slot:
                        old_obj = t.object
                        # For a STATE the adjective IS the value, and it lives
                        # in the predicate ("is happy"), not necessarily in the
                        # object field. Prefer the adjective when present so
                        # `is happy -> is sad` compares happy vs sad rather than
                        # two empty objects (which the guard would reject as
                        # "unspecified" and silently suppress the edge).
                        old_state = self._state_value(t.predicate)
                        if old_state:
                            old_obj = old_state
                        break
                new_obj = self._state_value(pred_key) or triplet.object
                if not self._objects_are_distinct(old_obj, new_obj):
                    old_id = None

            if old_id and old_id in self._fact_nodes:
                old_note = self._fact_nodes[old_id]
                # Close the OLD fact's validity at the new fact's timestamp:
                # "this was true from when it was stated until it was replaced".
                # The new fact opens an interval from its own date. Without this
                # the pair is only 'superseded', which loses WHEN the change
                # happened — the question `knowledge-update` items ask.
                new_from = note.timestamp_iso or (
                    note.t.strftime("%Y-%m-%dT%H:%M:%SZ") if note.t else None
                )
                old_from = old_note.timestamp_iso or (
                    old_note.t.strftime("%Y-%m-%dT%H:%M:%SZ") if old_note.t else None
                )

                # A SAME-TIMESTAMP pair cannot have a positive-length interval:
                # all notes extracted from one session share that session's
                # timestamp, so closing the old fact "at" the new fact's time
                # wrote `valid_to == valid_from` and the renderer printed "true
                # until <the very moment it became true>". A zero-length window
                # is never meaningful, so it is suppressed rather than published:
                # the pair is still recorded as superseding, it simply carries no
                # interval. (Audited on the real bank: 1 of 5 pairs had
                # valid_to == valid_from.)
                closes = bool(new_from and old_from and new_from > old_from)

                created_links.append(
                    LinkRecord(
                        target_id=old_id,
                        relation=_REL_SUPERSEDED,
                        valid_from=new_from,
                        valid_to=None,
                    )
                )
                # Mirror edge on the older note, recording where it ENDED.
                self._set_interval_link(
                    old_id, note.id, _REL_SUPERSEDED,
                    valid_from=old_from,
                    valid_to=(new_from if closes else None),
                )
                if self._graph is not None:
                    self._graph.add_edge(
                        note.id, old_id, relation=_REL_SUPERSEDED, weight=1.5
                    )
                _log.debug("THG versioning | {} supersedes {}", note.id[:8], old_id[:8])
            # Record the new note as the latest for BOTH keys, so the next
            # restatement or replacement resolves against it.
            self._subj_pred_index[exact_key] = note.id
            if self._is_single_valued_predicate(pred_key):
                self._subj_pred_index_slot[slot_key] = note.id

        # 3. Semantic similarity (top matches above tau, capped)
        for other_id, sim in self._semantic_peers(note):
            created_links.append(LinkRecord(target_id=other_id, relation=_REL_SEMANTIC))
            self._link_back(other_id, note.id, _REL_SEMANTIC)
            if self._graph is not None:
                self._graph.add_edge(note.id, other_id, relation=_REL_SEMANTIC, weight=sim)

        return created_links

    def add_temporal_sequence(self, note_ids: List[str]) -> int:
        """Symmetric chronological edges between consecutive notes of a session.

        Returns the number of edges created.
        """
        created = 0
        for i in range(len(note_ids) - 1):
            a_id, b_id = note_ids[i], note_ids[i + 1]
            if a_id not in self._fact_nodes or b_id not in self._fact_nodes:
                continue
            self._link_both(a_id, b_id, _REL_TEMPORAL)
            if self._graph is not None:
                self._graph.add_edge(a_id, b_id, relation=_REL_TEMPORAL, weight=1.2)
            created += 1
        return created

    def pop_touched_peers(self) -> List[Note]:
        """Peer notes whose ``L`` was extended; the caller re-persists them."""
        peers = [self._fact_nodes[i] for i in self._touched if i in self._fact_nodes]
        self._touched = set()
        return peers

    def hydrate(self, notes: Iterable[Note]) -> int:
        """Load already-persisted notes into the graph, rebuilding ALL indices.

        The hyper-graph lives only in memory, but the notes it links live in
        SQLite. A resumed ingest (one that continues an interrupted conversation
        instead of rebuilding it from scratch) opens a bank that already holds
        thousands of notes, and every new note would otherwise be linked only to
        other new notes — silently producing a disconnected graph.

        Hydrating first restores the ``same-entity``, ``semantic`` AND
        ``superseded_by`` channels across the resume boundary. The first two are
        computed by scanning ``_fact_nodes`` / ``_entity_index``; versioning
        needs the triplet indices, which is why the triplet is now PERSISTED on
        the note (``Note.triplets``, GB1).

        Before that, this method rebuilt no versioning index at all — the
        triplet was not stored anywhere — so a hydrated (pre-existing) note could
        never be superseded by a newly ingested one. Since the graph object is
        constructed per system instance while the bank is on disk, that meant
        versioning knowledge did not survive the process boundary at all.

        Notes from a pre-GB1 bank carry no triplets; they are hydrated exactly as
        before (no versioning index entries), which is the correct reading —
        nothing is known about their (subject, predicate, object), so nothing may
        be claimed superseded on the basis of them.

        Returns the number of notes hydrated.
        """
        count = 0
        for note in notes:
            self._fact_nodes[note.id] = note
            entity_keys = self._entity_keys(note)
            self._entity_index[note.id] = entity_keys
            # (GB3) keep the inverse index in step, or `_entity_peers` on a
            # hydrated bank would consult an index that does not know about the
            # notes it just loaded.
            for key in entity_keys:
                self._entity_to_notes.setdefault(key, set()).add(note.id)
            if self._graph is not None:
                self._graph.add_node(note.id, type="fact", date=note.session_date or "")
                for key in sorted(entity_keys):
                    self._entity_nodes.add(key)
                    self._graph.add_node(key, type="entity")
                    self._graph.add_edge(
                        note.id, key, relation="entity_cooccurrence", weight=1.0
                    )
            # (GB1) rebuild the versioning indices from the persisted triplets,
            # mirroring exactly what add_note records for a live note — including
            # the CANONICAL-HEAD slot key, so a hydrated note and a freshly
            # ingested one land in the same slot (they otherwise key differently
            # and versioning silently stops crossing the hydrate boundary).
            for triplet in getattr(note, "triplets", None) or []:
                if not triplet.is_usable:
                    continue
                pred_key = self._norm(triplet.predicate)
                exact_key = (
                    self._norm(triplet.subject),
                    pred_key,
                    self._norm(triplet.object),
                )
                self._subj_pred_index[exact_key] = note.id
                if self._is_single_valued_predicate(pred_key):
                    self._subj_pred_index_slot[
                        (
                            self._norm(triplet.subject),
                            self._slot_key_for_predicate(pred_key) or pred_key,
                        )
                    ] = note.id
            count += 1
        return count

    def get_note(self, note_id: str) -> Optional[Note]:
        return self._fact_nodes.get(note_id)

    def invalidate(self, note_id: str) -> bool:
        """Drop one note from the graph and every index. Returns whether it existed.

        The mirror-edge path (:meth:`_link_back`) appends edges to PEER notes in
        place, and those peers are re-persisted — but a peer that merely SHARES an
        entity never had its entity view invalidated, and nothing removed a note
        from the graph when the bank dropped it (deleted, or superseded out by a
        memory-manager DELETE). A stale ``_fact_nodes`` entry keeps the note
        reachable as an entity/semantic peer forever and keeps its entities in
        the df counts, so a hub never decays even after its notes are gone.

        This is the counterpart to :meth:`add_note` and must be called whenever a
        note leaves the backing bank.
        """
        note = self._fact_nodes.pop(note_id, None)
        if note is None:
            self._touched.discard(note_id)
            return False

        # Entity indices, both directions.
        for key in self._entity_index.pop(note_id, set()) or set():
            bucket = self._entity_to_notes.get(key)
            if bucket is not None:
                bucket.discard(note_id)
                if not bucket:
                    # Keep the dict from accumulating empty buckets, and let the
                    # nx graph drop the now-unreferenced entity node below.
                    self._entity_to_notes.pop(key, None)
                    self._entity_nodes.discard(key)

        # Triplet indices. Only clear an entry that still POINTS AT this note —
        # a later note may already have claimed the slot, and dropping that would
        # silently disable versioning against the current holder.
        for key, holder in list(self._subj_pred_index.items()):
            if holder == note_id:
                self._subj_pred_index.pop(key, None)
        for key, holder in list(self._subj_pred_index_slot.items()):
            if holder == note_id:
                self._subj_pred_index_slot.pop(key, None)

        if self._graph is not None:
            if self._graph.has_node(note_id):
                self._graph.remove_node(note_id)
            # Entity nodes left with no fact attached are no longer meaningful.
            for key in self._entity_nodes - set(self._entity_to_notes):
                if self._graph.has_node(key):
                    self._graph.remove_node(key)

        self._touched.discard(note_id)
        return True

    def clear(self) -> None:
        """Forget every node and edge (used when the backing bank is reset)."""
        self._graph = nx.Graph() if nx is not None else None
        self._fact_nodes = {}
        self._entity_nodes = set()
        self._entity_index = {}
        self._entity_to_notes = {}
        self._subj_pred_index = {}
        self._subj_pred_index_slot = {}
        self._touched = set()

    def __len__(self) -> int:
        return len(self._fact_nodes)

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    # HEADS of predicates that are FUNCTIONAL by nature: the subject can hold
    # only ONE value of this slot at a time, so a second value REPLACES the
    # first. This list is deliberately SMALL and defensible.
    #
    # The direction of this fix was INVERTED after auditing the real graph. The
    # first revision assumed the bug was that the list was too NARROW to be
    # corpus-reachable (literal multi-word phrases vs terse one-word predicates).
    # An audit of all 5 `superseded_by` pairs in the frozen 365-note LoCoMo bank
    # showed the opposite: the list was simultaneously too narrow in FORM and
    # far too WIDE in SEMANTICS. Every one of the 5 pairs was semantically wrong,
    # and 4 of 5 traced to `has`/`owns` or to the wildcard copula rule below:
    #
    #   "...has a dog and a cat" -> "...recently got another cat named Bailey"
    #     ADDITIVE, both true — caused by `has` being listed as single-valued.
    #   "reacted to Caroline's necklace, ... grandmother gave her a necklace
    #    from Sweden" -> "reacted to Caroline's necklace, saying it is gorgeous"
    #     a RESTATEMENT, and the NEWER note is the lossy one.
    #
    # So the rule is: only genuinely functional attributes — residence,
    # employment, relationship status, and a small closed set of states —
    # may supersede. Everything else must NOT, because the cost of a false
    # positive is silently deleting evidence that was never obsolete.
    _SINGLE_VALUED_PREDICATES = frozenset({
        # ── residence / location: one place at a time ───────────────────
        "live", "lives", "lived", "living",
        "reside", "resides", "resided", "residing",
        "move", "moves", "moved", "moving",
        "relocate", "relocates", "relocated",
        # ── employment / role: one job at a time ────────────────────────
        "work", "works", "worked", "working",
        "employed", "employ",
        # ── relationship status: one status at a time ───────────────────
        "married", "marry", "marries", "engaged", "engage",
        "dating", "divorced", "divorce", "widowed",
        # ── health state ────────────────────────────────────────────────
        "allergic", "diagnosed", "pregnant",
        # ── explicit currency markers ───────────────────────────────────
        "current", "currently",
        # ── legacy multi-word phrases, kept for backward compatibility ──
        # A caller may still pass a historical entry verbatim. These are the
        # FORM-variants of the functional verbs above, not new semantics.
        "lives in", "live in", "lived in", "living in", "resides in",
        "moved to", "works at", "works for", "worked at", "worked for",
        "employed by", "is married to", "married to", "is dating",
        "is engaged to", "is allergic to",
    })

    # CLOSED list of genuinely single-valued STATES (the state-adjective form).
    #
    # This replaces the previous rule "any multi-word predicate starting with
    # is/was". That wildcard was an unbounded noise source: it matched
    # "is gorgeous", "is doing well", "is really lucky", "is great to have a
    # supporter" — every one of which is a description, not a slot. 4 of the 5
    # audited bad pairs came from it directly or through the copula. An explicit
    # list keeps the intended behaviour (`is happy` -> `is sad`) with no wildcard.
    _SINGLE_VALUED_STATES = frozenset({
        "happy", "sad", "tired", "exhausted", "angry", "upset", "anxious",
        "stressed", "depressed", "lonely", "unhappy", "content", "excited",
        "nervous", "worried", "relieved", "grateful", "pregnant", "sick",
        "ill", "healthy", "single", "married", "engaged", "divorced",
        "employed", "unemployed", "retired",
    })

    # State verbs that introduce a `_SINGLE_VALUED_STATES` adjective.
    _STATE_VERBS = frozenset({"is", "was", "were", "feels", "feel", "felt"})

    # An object shorter than this many characters is treated as UNSPECIFIED, and
    # an unspecified object can never justify a supersession (see
    # `_objects_are_distinct`): marking evidence obsolete on missing data is the
    # worse error, so the conservative direction is to decline.
    _MIN_OBJECT_CHARS = 2

    @classmethod
    def _predicate_key(cls, predicate: str) -> str:
        """Reduce a predicate phrase to its canonical slot key (the head verb).

        The allow-list has to be reachable from what the extractor ACTUALLY
        emits, which is terse and inconsistent — ``lives``, ``lives in``,
        ``is living`` are the same slot. Matching a literal phrase list fails on
        any phrasing the corpus happens not to use, so reduction is on the HEAD.

        Rules, in order:

        1. lowercase / strip / collapse whitespace;
        2. drop leading HELPER tokens only (auxiliaries, articles, the
           infinitive marker): ``has been reading`` -> ``reading``;
        3. return the first remaining token as the head. A copula ("is", "was")
           is retained on purpose so the state form keeps its slot meaning.

        An object glued onto the predicate by the extractor (``"lives in
        Boston"``) is dropped by rule 3, which is what makes the
        light-verb + noun form reduce to its head.

        The head is then matched against the closed lists above. An unknown head
        is MULTI-valued, so an unrecognised verb never invalidates a fact.
        """
        text = " ".join(str(predicate or "").strip().lower().split())
        if not text:
            return ""
        tokens = text.split(" ")
        i = 0
        while i < len(tokens) and tokens[i] in cls._PREDICATE_HELPERS:
            i += 1
        if i >= len(tokens):
            return tokens[0] if tokens else ""
        return tokens[i]

    @classmethod
    def _is_single_valued_predicate(cls, predicate: str) -> bool:
        """True when `predicate` denotes an attribute with only one value.

        Consulted to decide whether a changed OBJECT should invalidate a
        previous fact. A predicate qualifies only when it is FUNCTIONAL: the
        subject holds one value of that slot at a time (residence, employment,
        relationship status, one of the closed states).

        Explicitly NOT qualifying, because they are multi-valued and a changed
        object is a new fact rather than a new value: `bought`, `attended`,
        `visited`, `studied`, `has`, `had`, `owns`, `needs`, `reacted_to`,
        `appreciates`, `gave`, `said`, `wants`, `likes`, `loves`. The audit of
        the real bank had `has` producing a wrong supersession between two
        simultaneously-true statements, which is exactly the failure this list
        prevents.
        """
        pred = cls._norm(predicate)
        if not pred:
            return False
        # Legacy exact-phrase match first, so a caller passing a historical
        # multi-word entry verbatim keeps working.
        if pred in cls._SINGLE_VALUED_PREDICATES:
            return True
        head = cls._predicate_key(pred)
        if not head:
            return False
        if head in cls._SINGLE_VALUED_PREDICATES:
            return True
        # State-adjective form, CLOSED list only: "is happy" / "was tired".
        # No prefix wildcard — the previous `head in {"is","was"} and
        # len(split) > 1` matched every descriptive clause in the corpus.
        if head in cls._STATE_VERBS:
            tokens = pred.split(" ")
            if len(tokens) > 1 and tokens[-1] in cls._SINGLE_VALUED_STATES:
                return True
        return False

    @classmethod
    def _objects_are_distinct(cls, old_object: str, new_object: str) -> bool:
        """True when two objects differ enough to count as a REPLACEMENT.

        A supersession requires the object to have actually CHANGED. Two guards,
        both from real bad pairs:

        * **Unspecified object.** If either side is empty (or a placeholder like
          "-" / "n/a"), nothing can be concluded. The audit's pair 3 matched with
          no usable object comparison at all; marking evidence obsolete on
          missing data is the worse error, so this returns False.
        * **Restatement.** If the two objects are near-identical strings
          ("Boston" vs "boston", or "a dog and a cat" vs "a dog and a cat"),
          they describe the SAME value — that is a restatement, which belongs in
          the `same-entity`/`semantic` channel, not a versioning edge. The
          audited pair 1 is exactly this case, and there the NEWER note was the
          lossy one, so superseding would have deleted the better evidence.
        """
        new = cls._norm(new_object)
        old = cls._norm(old_object)
        if len(new) < cls._MIN_OBJECT_CHARS or len(old) < cls._MIN_OBJECT_CHARS:
            return False
        if old == new:
            return False
        return True

    # Words that carry no slot information and are dropped while looking for
    # the head verb: articles, possessives, and the infinitive marker.
    #
    # NOTE the copulas are deliberately ABSENT from this set. "is"/"was" are
    # HEADS, not helpers: the state-adjective form ("is happy", "was tired")
    # derives its meaning from the copula plus the adjective, so stripping the
    # copula would reduce "was happy" to the bare adjective "happy" — which is
    # not a slot the allow-list can match, silently disabling state
    # supersession. Helper-dropping is only correct for genuine auxiliaries
    # ("has been reading" -> "reading").
    _PREDICATE_HELPERS = frozenset({
        "a", "an", "the", "to", "of", "in", "at", "on", "for", "with",
        "my", "his", "her", "their", "our", "its", "your",
        "been", "being", "having",
        "do", "does", "did", "will", "would", "shall", "should",
        "can", "could", "may", "might", "must", "going",
        "currently", "now", "just", "still", "really",
    })

    @classmethod
    def _slot_key_for_predicate(cls, predicate: str) -> str:
        """The SLOT a predicate occupies: what the subject holds one value of.

        For an ordinary functional predicate the slot is the head verb, so
        ``lives`` / ``lives in`` / ``is living`` all resolve to ``lives`` and a
        changed object replaces the old value.

        The STATE-ADJECTIVE form is the exception, and the correct key is neither
        the bare copula nor the full phrase:

        * keying the SLOT on the bare head ``is`` made ``is happy`` and ``is sad``
          share one key — but then the second write OVERWROTE the first instead
          of finding it as a differing value, so no ``superseded_by`` edge was
          emitted (a real bug: the helper returned the right BOOLEAN while
          deriving the wrong KEY);
        * keying it on the FULL phrase (``is happy`` vs ``is sad``) gives every
          state its own slot, so the two never collide and still no edge fires
          (the opposite failure, found by tracing rather than by inspection).

        A state is ONE slot — a person's mood — whose VALUES are the adjectives.
        So every state predicate for a subject shares the single key
        ``"state"``, and the adjective travels in the triplet's OBJECT, which is
        exactly what ``_objects_are_distinct`` compares. That is what makes
        ``is happy -> is sad`` a replacement (same slot, different value) while
        ``is happy -> was happy`` is a re-confirmation (same slot, same value).
        """
        text = cls._norm(predicate)
        if not text:
            return ""
        head = cls._predicate_key(text)
        tokens = text.split(" ")
        if head in cls._STATE_VERBS and len(tokens) > 1 and tokens[-1] in cls._SINGLE_VALUED_STATES:
            return "state"
        return head or text

    @classmethod
    def _state_value(cls, predicate: str) -> str:
        """The state adjective carried by a state predicate (else '')."""
        text = cls._norm(predicate)
        tokens = text.split(" ")
        if not tokens:
            return ""
        if cls._predicate_key(text) in cls._STATE_VERBS and tokens[-1] in cls._SINGLE_VALUED_STATES:
            return tokens[-1]
        return ""

    def _entity_peers(self, note: Note, entity_keys: Set[str]) -> List[Tuple[str, float]]:
        """Existing notes sharing >= 1 entity, best weighted match first.

        Two corrections live here.

        **Cost.** The candidate set is read from ``_entity_to_notes`` (the
        inverse index) instead of scanning every note in the bank and testing
        set intersection inline. The old scan made one ``add_note`` O(N) and a
        whole build O(N^2); now the work is proportional to the notes that
        actually share an entity.

        **Hub suppression (IDF).** The old score was "any shared key is an equal
        hit, then rank by embedding similarity". That is measuring the wrong
        thing when an entity is a hub: a corpus that tags every turn
        ``[User] ...`` puts the literal token ``user`` on nearly every note, so
        nearly every pair "shares an entity", the ``max_entity_links`` budget is
        spent reaching the same hub-neighbours, and ``same-entity`` stops
        discriminating while still consuming the link budget. (Measured on the
        LongMemEval bank: one role token on 18/27 notes produced ~92% of the
        whole entity channel, i.e. 174 edges on 27 notes.)

        So a shared key is weighted by its inverse document frequency across the
        graph's own notes, ``log(N / df_key)``, and the weights are SUMMED. A
        key on every note scores ``log(1) == 0`` and contributes nothing; a key
        on two notes scores ``log(N/2)`` and dominates. Ranking is by
        ``weighted_cooccurrence * similarity``, so a rare shared proper noun
        beats a bare hub token even when the hub token's embedding match is
        slightly better.

        ``entity_idf_floor`` drops keys at or below a weight, which is how a
        caller can suppress a global hub outright.
        """
        if not entity_keys:
            return []
        n_docs = len(self._fact_nodes)
        if n_docs <= 1:
            return []

        # Candidate discovery over the inverse index. Only notes reachable
        # through a shared key are ever visited — this is what makes the method
        # O(matching peers) instead of the previous O(N) scan of the bank.
        candidates: Set[str] = set()
        for key in entity_keys:
            bucket = self._entity_to_notes.get(key)
            if not bucket:
                continue
            if self._entity_weight(key) <= 0.0:
                # Key below the configured floor: the caller has explicitly
                # asked for it to be ignored.
                continue
            for other_id in bucket:
                if other_id != note.id:
                    candidates.add(other_id)

        if not candidates:
            return []

        ranked: List[Tuple[str, float]] = []
        for other_id in candidates:
            other = self._fact_nodes.get(other_id)
            if other is None:
                continue
            sim = self._cosine(note.e, other.e)
            # `weight` is the SUM of the shared keys' idf, but a SUM still lets
            # a hub win on candidate volume: `caroline` (df=238) gives an
            # enormous number of candidate peers whose summed weight beats a
            # rare entity that only two notes carry. Measured on the frozen
            # 365-note LoCoMo bank with sum-only ranking: 1769 of 1793 peer
            # slots were filled via hub-only entities and just 24 via a rare
            # entity — the weighting was present (rarities held 41.7% of the
            # weighted mass) but it never changed WHO was picked.
            #
            # So the shared-key signal enters as a MULTIPLIER on similarity,
            # normalised by the strongest key the pair shares rather than
            # accumulated. A pair sharing only a hub still groups together
            # (good — hub peers are weak but real corroboration), while a pair
            # sharing a rare entity gets a markedly higher multiplier and wins
            # the slot. `1 + idf` keeps a hub-only pair at a positive score
            # instead of zero, so the channel degrades gracefully.
            strength = 1.0 + max(
                (
                    self._entity_weight(k)
                    for k in (self._entity_index.get(other_id) or set())
                    & entity_keys
                ),
                default=0.0,
            )
            ranked.append((other_id, strength * sim))
        # Ties broken by the raw id so peer selection is STABLE across runs:
        # `sorted` alone leaves equal scores in dict-insertion order, which made
        # the chosen peers depend on iteration order.
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[: max(0, self.max_entity_links)]

    # Smallest idf weight any entity key can carry. Present so a key on every
    # note (`df == N`, raw idf exactly 0) is still weak-positive rather than
    # excluded — the difference between down-weighting a hub and severing the
    # entity channel. See `_entity_weight`.
    _MIN_ENTITY_WEIGHT = 0.05

    def _entity_weight(self, key: str) -> float:
        """IDF-style weight of one entity key over this graph's notes.

        Formula: ``max(_MIN_ENTITY_WEIGHT, log((N + 1) / (df + 1)))``.

        The ``+1`` smoothing alone is NOT enough, and this is worth recording
        because the naive version silently broke the channel: with
        ``log((N+1)/(df+1))`` a key present on EVERY note still scores exactly
        ``log(1) == 0``, so a note whose only entity is the conversation's
        dominant token ended up with **no** ``same-entity`` neighbours. Two notes
        that share a speaker name are weakly related, not unrelated; the goal is
        to stop a hub DOMINATING the ranking, not to delete the corroboration it
        genuinely carries. Hence an explicit positive floor, so the channel
        degrades to weak-but-present instead of empty.

        On the frozen fixture (N=365): ``caroline`` (df=238) -> log(366/239) =
        0.426, ``oliver`` (df=3) -> log(366/4) = 4.52 — a 10.6x separation, which
        is what reorders peers while keeping the rare-entity tail connected.

        ``entity_idf_floor`` (default -inf: no filtering) is the explicit lever
        for a caller that wants keys below a weight dropped outright.
        """
        n_docs = len(self._fact_nodes)
        if n_docs <= 1:
            return 0.0
        df = len(self._entity_to_notes.get(key) or ())
        if df <= 0:
            return 0.0
        weight = max(self._MIN_ENTITY_WEIGHT, math.log((n_docs + 1.0) / (df + 1.0)))
        return weight if weight > self.entity_idf_floor else 0.0

    def _semantic_peers(self, note: Note) -> List[Tuple[str, float]]:
        """Existing notes at/above ``semantic_tau``, most similar first.

        KNOWN REMAINING O(N) TERM, deliberately left in place for WS1. This is a
        relevance test against every note's embedding, so it cannot be answered
        from an index without either an ANN structure over the graph's notes or a
        change to which notes qualify — both of which would alter retrieval. It
        is therefore a BUILD-TIME cost (query-time `retrieve()` does not call it),
        and it is measured rather than assumed: at N=2000 it is ~51 ms of a
        ~141 ms `add_note` (36%, by cProfile), which after GB3 removed the entity
        scan is the dominant N-proportional term. Pinned by
        `tests/test_hyper_graph_v2.py::test_semantic_peers_remains_an_on_scan` so
        it cannot change silently.

        The `max_semantic_links == 0` early-out is NOT just an optimisation: the
        cap previously trimmed the RESULT list while still computing a cosine
        against every note, so disabling the channel did not disable its cost —
        a graph configured with `max_semantic_links=0` still paid the full O(N)
        scan. Callers that turn the channel off now genuinely turn it off.
        """
        if note.e is None or self.max_semantic_links <= 0:
            return []
        similar: List[Tuple[str, float]] = []
        for other_id, other in self._fact_nodes.items():
            if other_id == note.id or other.e is None:
                continue
            sim = self._cosine(note.e, other.e)
            if sim >= self.semantic_tau:
                similar.append((other_id, sim))
        similar.sort(key=lambda item: item[1], reverse=True)
        return similar[: self.max_semantic_links]

    # ------------------------------------------------------------------
    # Link bookkeeping
    # ------------------------------------------------------------------

    def _link_back(self, peer_id: str, src_id: str, relation: str) -> None:
        """Apply the mirror edge ``peer -> src`` in place, and mark the peer."""
        peer = self._fact_nodes.get(peer_id)
        if peer is None:
            return
        if not any(
            link.target_id == src_id and link.relation == relation for link in peer.L
        ):
            peer.L.append(LinkRecord(target_id=src_id, relation=relation))
        self._touched.add(peer_id)

    def _set_interval_link(
        self,
        peer_id: str,
        src_id: str,
        relation: str,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> None:
        """Add/refresh a mirror edge carrying a validity interval.

        Like :meth:`_link_back`, but the edge is interval-bearing. An existing
        edge with the same ``(target, relation)`` is REPLACED so a later
        supersession can extend/close the interval rather than being ignored on
        the "already linked" check.
        """
        peer = self._fact_nodes.get(peer_id)
        if peer is None:
            return
        peer.L = [
            link
            for link in peer.L
            if not (link.target_id == src_id and link.relation == relation)
        ]
        peer.L.append(
            LinkRecord(
                target_id=src_id,
                relation=relation,
                valid_from=valid_from,
                valid_to=valid_to,
            )
        )
        self._touched.add(peer_id)

    def _link_both(self, a_id: str, b_id: str, relation: str) -> None:
        """Symmetric edge between two notes already known to the graph."""
        a = self._fact_nodes.get(a_id)
        if a is None:
            return
        if not any(
            link.target_id == b_id and link.relation == relation for link in a.L
        ):
            a.L.append(LinkRecord(target_id=b_id, relation=relation))
        self._touched.add(a_id)
        self._link_back(b_id, a_id, relation)

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _entity_keys(note: Note) -> Set[str]:
        """Identifiers for the ENTITY channel: named things, not topic keywords.

        Only ``entities`` feeds this channel. It used to also union ``K``
        (keywords), which conflated two different signals: a shared *entity*
        ("Caroline", "Hawaii") is evidence that two notes are about the same
        thing, whereas a shared *keyword* ("shopping", "travel", "tired") only
        means they are on a similar topic — which the semantic channel already
        measures with embeddings, and which the BM25 channel already measures
        lexically.

        Folding keywords in inflated the entity fan-out badly: with
        ``max_entity_links=5`` every note grabbed the 5 highest-similarity peers
        *among any note sharing any keyword*, so the entity graph drifted toward
        a clique and ``same-entity`` stopped discriminating. Keywords remain
        available to retrieval through the keyword/BM25 channel.

        ROLE tokens are dropped on top of that. ``user`` / ``assistant`` are not
        entities — they are speaker roles that LongMemEval writes on every turn
        (``[User] ...``), so indexing them puts one literal string on nearly
        every note of the bank. A token that appears on everything carries no
        information about which two notes are *about the same thing*, but it
        does consume the entity link budget. The IDF weighting in
        :meth:`_entity_peers` suppresses such a hub statistically; dropping the
        unambiguous role words here removes it at the source, and leaves genuine
        speaker NAMES (which share their own idf treatment) to the statistics.
        """
        keys = {
            TemporalHyperGraph._norm(e)
            for e in (note.entities or [])
        }
        keys.discard("")
        return keys - TemporalHyperGraph._ROLE_TOKENS

    # Speaker/role words that a dialogue corpus writes into the `entities` field
    # but which name no entity. Kept deliberately small and generic — dataset
    # proper nouns must NOT be hardcoded here (that would break LoCoMo), and a
    # name that IS a hub is handled by IDF weighting instead.
    _ROLE_TOKENS = frozenset({
        "user", "assistant", "system", "ai", "bot", "agent",
        "speaker", "participant", "person", "someone", "anyone",
        "they", "them", "you", "i", "me", "we", "us", "he", "she", "it",
    })

    @staticmethod
    def _norm(text: str) -> str:
        return str(text or "").strip().lower()

    @staticmethod
    def _cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        a = np.asarray(a, dtype="float32").reshape(-1)
        b = np.asarray(b, dtype="float32").reshape(-1)
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _directed_from_links(self) -> Any:
        """A DiGraph whose edges follow the PERSISTED link direction.

        ``personalized_pagerank`` needs a notion of direction to support the
        reverse walk (GB4), but ``self._graph`` is a plain undirected
        ``networkx.Graph`` — and ``Graph`` has no ``reverse()``, so the first
        implementation raised ``AttributeError``. More fundamentally, on an
        undirected graph a "reversed" walk is identical to the forward one and
        GB4 would be a no-op.

        Direction has to be recovered from somewhere, and it IS available: the
        note-to-note projection persists ``Note.L`` as "X links to Y", which is a
        directed assertion. So this builds ``X -> Y`` for every such record. The
        reverse walk then flows from the seeds back along those assertions,
        which is exactly "what does this evidence rest on" — the premise case
        that an outward-only walk cannot reach.

        Entity-cooccurrence and semantic edges carry no inherent direction, so
        they are added both ways and are unaffected by the reversal.
        """
        if nx is None:
            return None
        directed = nx.DiGraph()
        for note_id, note in self._fact_nodes.items():
            directed.add_node(note_id, type="fact", date=note.session_date or "")
        for note_id, note in self._fact_nodes.items():
            for link in note.L or []:
                if link.target_id in self._fact_nodes:
                    directed.add_edge(
                        note_id, link.target_id,
                        relation=link.relation, weight=1.0,
                    )
            for key in self._entity_keys(note):
                directed.add_node(key, type="entity")
                directed.add_edge(note_id, key, relation="entity_cooccurrence", weight=1.0)
        return directed

    def _resolve_ppr_graph(self) -> Optional[Any]:
        """Return the nx graph the FORWARD walk uses.

        ``_graph`` is populated by ``add_note``/``hydrate``. A graph built by
        neither — one constructed and then handed a bank only through
        ``hydrate``, or a caller that populated ``_fact_nodes`` directly — would
        make every PPR query return nothing, so the structure is rebuilt from the
        notes the graph already holds in that case.

        ``Note.L`` records that are not yet in ``_graph`` are also folded in.
        They matter on a RESUMED bank: the notes arrive from SQLite carrying
        their persisted links, and a walk that ignored them would see a
        disconnected graph even though the edges exist. Adding them is
        conservative (edges are only ever added, never removed), and it keeps the
        forward and reverse walks operating on the same topology.
        """
        graph = self._graph
        if graph is None or graph.number_of_nodes() == 0:
            if nx is None or not self._fact_nodes:
                return None
            graph = nx.Graph()
            for note_id, note in self._fact_nodes.items():
                graph.add_node(note_id, type="fact", date=note.session_date or "")
                for key in self._entity_keys(note):
                    graph.add_node(key, type="entity")
                    graph.add_edge(note_id, key, relation="entity_cooccurrence", weight=1.0)
            self._graph = graph

        # Fold in any persisted note-to-note link the nx graph does not know yet.
        for note_id, note in self._fact_nodes.items():
            if not graph.has_node(note_id):
                graph.add_node(note_id, type="fact", date=note.session_date or "")
            for link in note.L or []:
                target = link.target_id
                if target in self._fact_nodes and not graph.has_edge(note_id, target):
                    graph.add_edge(
                        note_id, target, relation=link.relation, weight=1.0
                    )
        return graph

    def personalized_pagerank(
        self,
        seed_note_ids: List[str],
        top_k: int = 10,
        directions: str = "out",
        reverse: bool = False,
    ) -> List[Tuple[str, float]]:
        """Personalized PageRank over the hyper-graph, seeded on ``seed_note_ids``.

        ``directions`` selects the walk:
          * ``"out"`` (default) — the historical behaviour: the walk flows from
            the seeds along the graph's edges.
          * ``"in"`` — the reversed-edge walk: rank notes that POINT AT the
            seeds. This is what surfaces a *premise* the seed depends on, which
            an outward-only walk provably cannot reach at any rank.
          * ``"both"`` — the two rankings merged by max.

        ``reverse=True`` is a convenience alias for ``directions="in"`` (GB4), so
        a caller that only wants "give me the notes feeding INTO these seeds"
        does not need to know about the ``directions`` vocabulary.

        Returns ``[(note_id, score), ...]`` sorted by score, fact nodes only.

        The empty-result contract matters as much as the walk. This used to
        return ``[(nid, 1.0) for nid in seeds]`` when no seed was in the graph —
        a UNIFORM score. A uniform score is worse than no score: the retriever
        min-max normalises this channel, so every note ties and Phase D receives
        a flat term that dilutes the real ranking signal. It now rebuilds the
        graph from the notes it holds and, failing that, returns ``[]`` so the
        caller can fall through to its other channels.
        """
        graph = self._resolve_ppr_graph()
        if graph is None or graph.number_of_nodes() == 0:
            return []

        valid_seeds = [nid for nid in seed_note_ids if graph.has_node(nid)]
        if not valid_seeds:
            return []

        # A seed with NO incident edges cannot propagate any of its
        # personalization mass, so the walk degenerates to "the seed keeps all
        # of it": every seed scores 1.0 and every other note 0.0. That is exactly
        # the UNIFORM-score failure mode this method must not produce — the
        # retriever min-max normalises the channel, so a constant ranking
        # injects a flat term into Phase D. Return [] and let the caller fall
        # through to its other channels (the caller treats a missing score as
        # "below the ranked set", which is the correct reading of "no evidence").
        #
        # This is checked against the graph that will actually be WALKED, not
        # against `_graph`: the reverse walk runs over the directed projection of
        # `Note.L`, where a seed can have incoming edges that the undirected
        # `_graph` does not know about. Testing `_graph` here silently returned []
        # for exactly the premise case GB4 exists to serve.
        want = str(directions).lower()
        reverse_walk = bool(reverse) or want == "in"
        if reverse_walk or want == "both":
            probe_graph = self._directed_from_links()
        else:
            probe_graph = graph
        if probe_graph is None or not any(
            probe_graph.has_node(nid) and probe_graph.degree(nid) > 0
            for nid in valid_seeds
        ):
            return []

        personalization = {nid: (1.0 / len(valid_seeds)) for nid in valid_seeds}

        # The forward walk uses the undirected graph (unchanged from before GB4,
        # so `directions="out"` is byte-identical to the historical behaviour).
        # The reverse walk needs DIRECTION, which an undirected Graph cannot
        # express, so it runs over the DiGraph built from the persisted link
        # records — see `_directed_from_links`.
        try:
            if reverse_walk or want == "both":
                directed = self._directed_from_links()
                if directed is None or not directed.has_node(valid_seeds[0]):
                    return []
                rev_graph = directed.reverse(copy=False)
                rev_scores = nx.pagerank(
                    rev_graph, alpha=0.85, personalization=personalization, max_iter=100
                )
                if want == "both" or not reverse_walk:
                    fwd_scores = nx.pagerank(
                        graph, alpha=0.85, personalization=personalization, max_iter=100
                    )
                    pr_scores = {
                        nid: max(float(s), float(rev_scores.get(nid, 0.0)))
                        for nid, s in fwd_scores.items()
                    }
                else:
                    pr_scores = rev_scores
            else:
                pr_scores = nx.pagerank(
                    graph, alpha=0.85, personalization=personalization, max_iter=100
                )
            fact_scores = [
                (nid, float(score))
                for nid, score in pr_scores.items()
                if nid in self._fact_nodes
            ]
            # Ties broken by id so the ranking is deterministic across runs
            # (dict iteration order otherwise decides equal scores).
            fact_scores.sort(key=lambda x: (-x[1], x[0]))
            return fact_scores[:top_k]
        except Exception as exc:
            _log.warning("PPR computation fallback: {}", exc)
            return []

    def bidirectional_pagerank(
        self, seeds: List[str], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Rank notes by BOTH the forward and reverse personalization walks.

        ``personalized_pagerank`` walks outward from the seeds, so a note that
        merely POINTS AT the seed set — a premise the seed depends on — could
        never be surfaced, no matter how strong that dependency is. This runs the
        forward walk and the reverse walk and merges them by MAX, giving one
        ranking that admits both "what follows from the evidence" and "what the
        evidence rests on".

        Returned scores are normalised to their own maximum so the two walks,
        whose raw PageRank mass differs, are comparable before merging.
        """
        forward = self.personalized_pagerank(seeds, top_k=top_k * 2, directions="out")
        reverse = self.personalized_pagerank(seeds, top_k=top_k * 2, directions="in")

        def _norm(rows: List[Tuple[str, float]]) -> Dict[str, float]:
            if not rows:
                return {}
            hi = max(score for _, score in rows)
            if hi <= 0.0:
                return {nid: 0.0 for nid, _ in rows}
            return {nid: score / hi for nid, score in rows}

        merged: Dict[str, float] = {}
        for nid, score in _norm(forward).items():
            merged[nid] = max(merged.get(nid, 0.0), score)
        for nid, score in _norm(reverse).items():
            merged[nid] = max(merged.get(nid, 0.0), score)

        ranked = sorted(merged.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

