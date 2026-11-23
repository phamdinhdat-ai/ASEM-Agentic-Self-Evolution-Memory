"""Layered memory graph over one conversation history, built without an LLM.

Layers (all nodes in one sparse graph):
    L0 turns         edges: next (adjacent turns of a session), semantic (k-NN by embedding)
    L1 entities      capitalized tokens that occur in few turns (IDF filter); edge: mention
    L2 sessions      edge: member
    L3 communities   Louvain communities of the L0 turn graph; edge: member

Query = personalized PageRank (PPR) from dense seed turns over the whole graph.
The PPR mass of a turn is its query-specific *influence*. Rankings:
    ppr       turns by PPR mass
    mix       lambda * z(dense) + (1 - lambda) * z(PPR)
    community best turns from the communities closest to the query, round-robin
All randomness (Louvain) is seeded. Deterministic given the embeddings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re

import numpy as np

_CAP = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_STOP = {"The", "This", "That", "What", "When", "Where", "Which", "Who", "How", "Why", "Yes",
         "Yeah", "Thanks", "Thank", "Hey", "Hi", "Hello", "Wow", "Sure", "Also", "And", "But",
         "Just", "Well", "Okay", "Great", "Good", "Nice", "Here", "There", "They", "You", "Your",
         "She", "Her", "His", "Him", "We", "Our", "It", "Its", "Let", "Can", "Could", "Would",
         "Should", "Do", "Did", "Does", "Have", "Has", "Had", "Are", "Was", "Were", "Is", "If",
         "So", "Oh", "Not", "No", "All", "Some", "For", "With", "From", "Into", "Image", "Caption"}


@dataclass(frozen=True)
class GraphConfig:
    knn: int = 5
    w_next: float = 1.0
    w_sem: float = 1.0
    w_entity: float = 0.5
    w_session: float = 0.3
    w_community: float = 0.3
    entity_max_df: float = 0.05     # an entity may appear in at most 5% of turns
    seeds: int = 5
    alpha: float = 0.5              # restart probability
    iterations: int = 40
    louvain_seed: int = 20261117


class LayeredMemoryGraph:
    def __init__(self, turns, turn_vectors: np.ndarray, config: GraphConfig = GraphConfig()):
        self.turns = list(turns)
        self.cfg = config
        n = len(self.turns)
        self.vectors = turn_vectors.astype(np.float32)
        self.n_turns = n
        nodes = [("turn", t.source_id) for t in self.turns]
        edges: list[tuple[int, int, float]] = []

        # L0: next edges inside a session
        for i in range(n - 1):
            if self.turns[i].session_id == self.turns[i + 1].session_id:
                edges.append((i, i + 1, config.w_next))
        # L0: semantic k-NN edges
        sims = self.vectors @ self.vectors.T
        np.fill_diagonal(sims, -np.inf)
        k = min(config.knn, max(n - 1, 0))
        self.sem_pairs = set()
        for i in range(n):
            for j in np.argsort(-sims[i])[:k]:
                j = int(j)
                pair = (min(i, j), max(i, j))
                if pair not in self.sem_pairs and np.isfinite(sims[i, j]):
                    self.sem_pairs.add(pair)
                    edges.append((pair[0], pair[1], config.w_sem * max(float(sims[i, j]), 0.0)))

        # L3 communities over the turn graph (computed before adding other layers)
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        g = nx.Graph()
        g.add_nodes_from(range(n))
        for a, b, w in edges:
            if w > 0:
                g.add_edge(a, b, weight=w)
        self.communities = [sorted(c) for c in louvain_communities(g, weight="weight",
                                                                  seed=config.louvain_seed)]
        self.community_of = {i: ci for ci, members in enumerate(self.communities) for i in members}

        # L1 entities
        mentions = [set(m for m in _CAP.findall(t.text) if m not in _STOP) for t in self.turns]
        df = Counter(e for ms in mentions for e in ms)
        limit = max(2, int(config.entity_max_df * n))
        entity_index = {}
        for e, c in sorted(df.items()):
            if 2 <= c <= limit:
                entity_index[e] = len(nodes)
                nodes.append(("entity", e))
        for i, ms in enumerate(mentions):
            for e in ms:
                if e in entity_index:
                    edges.append((i, entity_index[e], config.w_entity))
        self.n_entities = len(entity_index)

        # L2 sessions
        session_index = {}
        for i, t in enumerate(self.turns):
            if t.session_id not in session_index:
                session_index[t.session_id] = len(nodes)
                nodes.append(("session", t.session_id))
            edges.append((i, session_index[t.session_id], config.w_session))

        # L3 community nodes
        for ci, members in enumerate(self.communities):
            idx = len(nodes)
            nodes.append(("community", str(ci)))
            for i in members:
                edges.append((i, idx, config.w_community))

        self.nodes = nodes
        m = len(nodes)
        adj = np.zeros((m, m), dtype=np.float64)       # histories have < ~1,000 nodes; dense is fine
        for a, b, w in edges:
            adj[a, b] += w
            adj[b, a] += w
        degree = adj.sum(axis=1)
        dangling = degree == 0
        degree[dangling] = 1.0
        self.P = adj / degree[:, None]                  # row-stochastic (dangling rows handled below)
        self.dangling = dangling
        self.n_edges = int(np.count_nonzero(np.triu(adj)))
        self.centroids = np.stack([self.vectors[c].mean(axis=0) for c in self.communities])

    # --- queries -------------------------------------------------------------------------
    def dense_scores(self, q: np.ndarray) -> np.ndarray:
        return self.vectors @ q.astype(np.float32)

    def ppr(self, q: np.ndarray) -> np.ndarray:
        """PPR mass on every node from the top-`seeds` dense turns (uniform restart)."""
        dense = self.dense_scores(q)
        seeds = np.argsort(-dense)[: self.cfg.seeds]
        s = np.zeros(len(self.nodes))
        s[seeds] = 1.0 / len(seeds)
        pi = s.copy()
        a = self.cfg.alpha
        for _ in range(self.cfg.iterations):
            step = self.P.T @ pi
            step += s * pi[self.dangling].sum()                # dangling mass returns to the seeds
            pi = a * s + (1 - a) * step
        return pi

    def rank(self, q: np.ndarray, method: str, k: int, lam: float = 0.5) -> list[int]:
        dense = self.dense_scores(q)
        if method == "dense":
            score = dense
        elif method in ("ppr", "mix"):
            influence = self.ppr(q)[: self.n_turns]
            if method == "ppr":
                score = influence
            else:
                z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
                score = lam * z(dense) + (1 - lam) * z(influence)
        elif method == "community":
            return self._community_rank(q, dense, k)
        else:
            raise ValueError(method)
        return [int(i) for i in sorted(range(self.n_turns), key=lambda i: (-float(score[i]), i))[:k]]

    def _community_rank(self, q: np.ndarray, dense: np.ndarray, k: int) -> list[int]:
        order = np.argsort(-(self.centroids @ q.astype(np.float32)))
        queues = [sorted(self.communities[int(c)], key=lambda i: (-float(dense[i]), i)) for c in order]
        out: list[int] = []
        depth = 0
        while len(out) < k and any(depth < len(qu) for qu in queues):
            for qu in queues[:max(1, min(len(queues), 4))]:   # round-robin over the 4 closest communities
                if depth < len(qu) and qu[depth] not in out:
                    out.append(qu[depth])
                    if len(out) == k:
                        break
            depth += 1
        for i in sorted(range(self.n_turns), key=lambda i: (-float(dense[i]), i)):   # fill if needed
            if len(out) == k:
                break
            if i not in out:
                out.append(i)
        return out

    def stats(self) -> dict:
        return {"turns": self.n_turns, "entities": self.n_entities,
                "sessions": sum(1 for kind, _ in self.nodes if kind == "session"),
                "communities": len(self.communities), "semantic_edges": len(self.sem_pairs),
                "nodes": len(self.nodes), "edges": self.n_edges}
