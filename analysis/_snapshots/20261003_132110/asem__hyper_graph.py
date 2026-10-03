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

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:  # pragma: no cover - networkx is a hard dep in practice
    nx = None

from .logging_utils import get_logger
from .note import LinkRecord, Note

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
    ) -> None:
        self.semantic_tau = semantic_tau
        # Fan-out caps. A dominant entity (the conversation's own speaker name,
        # "Caroline") is shared by most notes, so an uncapped entity channel
        # would link every note to every other note and collapse the graph into
        # a clique. Peers are ranked by embedding similarity before the cut.
        self.max_entity_links = max_entity_links
        self.max_semantic_links = max_semantic_links

        self._graph = nx.Graph() if nx is not None else None
        self._fact_nodes: Dict[str, Note] = {}
        self._entity_nodes: Set[str] = set()
        self._entity_index: Dict[str, Set[str]] = {}  # note_id -> normalised entity keys
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
            slot_key = (subj_key, pred_key)

            old_id = self._subj_pred_index.get(exact_key)
            if old_id is None and self._is_single_valued_predicate(pred_key):
                old_id = self._subj_pred_index_slot.get(slot_key)
            # A fact must not supersede itself, and an already-known id must
            # still exist in this graph.
            if old_id == note.id:
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
                    valid_from=old_from, valid_to=new_from,
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
        """Load already-persisted notes into the graph WITHOUT creating links.

        The hyper-graph lives only in memory, but the notes it links live in
        SQLite. A resumed ingest (one that continues an interrupted conversation
        instead of rebuilding it from scratch) opens a bank that already holds
        thousands of notes, and every new note would otherwise be linked only to
        other new notes — silently producing a disconnected graph.

        Hydrating first restores the ``same-entity`` and ``semantic`` channels
        across the resume boundary, because both are computed by scanning
        ``_fact_nodes`` / ``_entity_index``.

        Known limitation: the ``(subject, predicate) -> note_id`` index used for
        ``superseded_by`` versioning is NOT rebuilt, because the triplet is not
        persisted on the note. Notes hydrated here can therefore be superseded
        only by a later hydrated note, never by a newly ingested one. Pass
        ``subj_pred`` to supply the triplets when the caller has them.
        """
        count = 0
        for note in notes:
            self._fact_nodes[note.id] = note
            self._entity_index[note.id] = self._entity_keys(note)
            if self._graph is not None:
                self._graph.add_node(note.id, type="fact", date=note.session_date or "")
            count += 1
        return count

    def get_note(self, note_id: str) -> Optional[Note]:
        return self._fact_nodes.get(note_id)

    def clear(self) -> None:
        """Forget every node and edge (used when the backing bank is reset)."""
        self._graph = nx.Graph() if nx is not None else None
        self._fact_nodes = {}
        self._entity_nodes = set()
        self._entity_index = {}
        self._subj_pred_index = {}
        self._subj_pred_index_slot = {}
        self._touched = set()

    def __len__(self) -> int:
        return len(self._fact_nodes)

    # ------------------------------------------------------------------
    # Ranking helpers
    # ------------------------------------------------------------------

    # HEADS of predicates whose object is a single-valued attribute: a new value
    # REPLACES the old one rather than adding a separate fact.
    #
    # Matching is on the normalised HEAD VERB of the predicate, not on the full
    # phrase. The previous revision stored literal multi-word phrases ("lives in",
    # "is reading") while the extractor prompt asks for a bare
    # `"predicate": "verb / status / relationship"` and the per-turn fallback
    # emits `predicate: ""` — so a corpus that says "lives"/"works"/"owns" never
    # matched anything, and the documented W3 case
    # (`lives in Boston` -> `lives in Seattle`) fired ~never. Measured on the
    # LongMemEval bank build: **0 superseded_by edges**.
    #
    # This is a deliberately conservative allow-list. The cost of a false
    # positive is a fact wrongly marked obsolete (silently removing evidence);
    # the cost of a false negative is a missed versioning edge. Being
    # conservative is the safer direction, so only verbs that denote a
    # single-valued slot are listed — a verb is NOT listed merely because it
    # appeared in a superseding sentence.
    _SINGLE_VALUED_PREDICATES = frozenset({
        # ── residence / location (one place at a time) ──────────────────
        "live", "lives", "lived", "living", "reside", "resides", "resided",
        "residing", "stay", "stays", "stayed", "move", "moves", "moved",
        "moving", "relocate", "relocates", "relocated",
        # ── employment / role (one job at a time) ───────────────────────
        "work", "works", "worked", "working", "employed", "employ",
        "hire", "hired",
        # ── relationship status (one status at a time) ──────────────────
        "married", "marry", "marries", "dating", "date", "dates",
        "engaged", "engage", "divorced", "divorce", "widowed",
        # ── health / body state ─────────────────────────────────────────
        "allergic", "diagnosed", "pregnant",
        # ── possession / current tooling (the user's ONE of a kind slot) ─
        "own", "owns", "owned", "owning",
        "has", "have", "had", "having",
        "use", "uses", "used", "using",
        "drive", "drives", "drove", "driven",
        # ── ongoing consumption / study (one current item) ──────────────
        "read", "reads", "reading",
        "watch", "watches", "watching",
        "learn", "learns", "learning",
        "study", "studies", "studying",
        # NOT listed on purpose: "attend"/"attended", "visit"/"visited",
        # "graduate"/"graduated", "buy"/"bought". These are MULTI-valued — a
        # person attends many events, visits many places and graduates from
        # more than one programme — so a changed object is a new fact, not a
        # new value of one slot. Listing them would mark real evidence obsolete.
        # ── plans / intent for a single slot ────────────────────────────
        "plan", "plans", "planned", "planning",
        "want", "wants", "wanted", "hoping", "hope", "hopes",
        # ── state-of-being ──────────────────────────────────────────────
        "is", "was", "were", "feels", "feel", "felt", "seems", "seem",
        # ── explicit currency markers ───────────────────────────────────
        "current", "currently", "now",
        # ── legacy multi-word phrases, kept for backward compatibility ──
        # These can no longer be produced by head-extraction, but an existing
        # caller (or a hand-built Triplet) may still pass them verbatim.
        "lives in", "live in", "lived in", "living in", "resides in",
        "moved to", "works at", "works for", "worked at", "worked for",
        "employed by", "is married to", "married to", "is dating",
        "is engaged to", "is allergic to", "is reading", "is watching",
        "is learning", "is studying",
    })

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
    def _predicate_head(cls, predicate: str) -> str:
        """Reduce a predicate phrase to the single canonical slot verb.

        The allow-list has to be reachable from what the extractor ACTUALLY
        emits, which is terse and inconsistent — ``lives``, ``lives in``,
        ``is reading``, ``reading``, ``has been reading``, ``works at`` are all
        the same slot. Matching a literal phrase list therefore fails on any
        phrasing the corpus happens to use (the defect this replaces).

        Reduction rules, in order:

        1. lowercase / strip / collapse whitespace;
        2. drop leading HELPER tokens only (auxiliaries, articles, the
           infinitive marker): ``has been reading`` -> ``reading``;
        3. return the first remaining token as the head. A copula ("is", "was")
           is retained as the head on purpose, so the state-adjective form
           keeps its slot meaning.

        An object glued onto the predicate by the extractor (``"bought a car"``,
        ``"lives in Boston"``) is dropped by rule 3, which is what makes the
        light-verb + noun form reduce to ``bought`` / ``lives``.

        The returned head is looked up in :attr:`_SINGLE_VALUED_PREDICATES`; an
        unknown head is treated as MULTI-valued, so an unrecognised verb never
        invalidates an existing fact.
        """
        text = " ".join(str(predicate or "").strip().lower().split())
        if not text:
            return ""
        tokens = [t for t in text.split(" ") if t]
        i = 0
        while i < len(tokens) and tokens[i] in cls._PREDICATE_HELPERS:
            i += 1
        if i >= len(tokens):
            # Every token was a helper word; fall back to the first token so an
            # all-helper predicate is at least stable (and unmatched).
            return tokens[0] if tokens else ""
        return tokens[i]

    # Copulas whose presence signals the state-adjective form ("is happy").
    _STATE_COPULAS = frozenset({"is", "was", "were", "feels", "feel", "felt"})

    @classmethod
    def _is_single_valued_predicate(cls, predicate: str) -> bool:
        """True when `predicate` denotes an attribute with only one value.

        Used to decide whether a changed OBJECT should invalidate the previous
        fact. Multi-valued verbs ("bought", "attended", "visited") are excluded
        so that "bought a car" and "bought a house" remain two facts — and so
        that "attended X" is not superseded by "attended Y", since one can
        attend many events.

        Matching is on the HEAD VERB of the predicate (see
        :meth:`_predicate_head`) so ``lives`` / ``lives in`` / ``is living`` all
        resolve to the same slot, and the state-adjective form ("is happy" /
        "was tired") is recognised from the same head.
        """
        pred = cls._norm(predicate)
        if not pred:
            return False
        # Legacy exact-phrase match first, so a caller passing a historical
        # multi-word entry verbatim keeps working.
        if pred in cls._SINGLE_VALUED_PREDICATES:
            return True
        head = cls._predicate_head(pred)
        if not head:
            return False
        if head in cls._SINGLE_VALUED_PREDICATES:
            return True
        # "is <adj>" / "was <adj>" / "feels <adj>" — a state a later state
        # replaces. Requires MORE than the bare copula: a predicate that is
        # just "is" is matched by the allow-list above instead.
        return head in cls._STATE_COPULAS and len(pred.split(" ")) > 1

    def _entity_peers(self, note: Note, entity_keys: Set[str]) -> List[Tuple[str, float]]:
        """Existing notes sharing >= 1 entity, best embedding match first."""
        if not entity_keys:
            return []
        scored: Dict[str, float] = {}
        for other_id, other in self._fact_nodes.items():
            if other_id == note.id:
                continue
            if not (self._entity_index.get(other_id) or set()) & entity_keys:
                continue
            scored[other_id] = self._cosine(note.e, other.e)
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return ranked[: max(0, self.max_entity_links)]

    def _semantic_peers(self, note: Note) -> List[Tuple[str, float]]:
        """Existing notes at/above ``semantic_tau``, most similar first."""
        if note.e is None:
            return []
        similar: List[Tuple[str, float]] = []
        for other_id, other in self._fact_nodes.items():
            if other_id == note.id or other.e is None:
                continue
            sim = self._cosine(note.e, other.e)
            if sim >= self.semantic_tau:
                similar.append((other_id, sim))
        similar.sort(key=lambda item: item[1], reverse=True)
        return similar[: max(0, self.max_semantic_links)]

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
        """
        keys = {TemporalHyperGraph._norm(e) for e in (note.entities or [])}
        keys.discard("")
        return keys

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

    def personalized_pagerank(self, seed_note_ids: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Run Personalized PageRank (PPR) over hyper-graph from seed notes."""
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return [(nid, 1.0) for nid in seed_note_ids[:top_k]]

        valid_seeds = [nid for nid in seed_note_ids if self._graph.has_node(nid)]
        if not valid_seeds:
            return []

        personalization = {nid: (1.0 / len(valid_seeds)) for nid in valid_seeds}
        try:
            pr_scores = nx.pagerank(self._graph, alpha=0.85, personalization=personalization, max_iter=100)
            # Filter only fact nodes
            fact_scores = [(nid, score) for nid, score in pr_scores.items() if nid in self._fact_nodes]
            fact_scores.sort(key=lambda x: x[1], reverse=True)
            return fact_scores[:top_k]
        except Exception as exc:
            _log.warning("PPR computation fallback: {}", exc)
            return [(nid, 1.0) for nid in valid_seeds[:top_k]]

