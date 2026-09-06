# ASEMv2 → FastASEM (ASEM-v3): Technical Upgrade

**Scope:** This document describes, at the code level, how the ASEM memory system was
upgraded from **ASEMv2** (batch LLM ingestion + enhanced hybrid retrieval) to
**FastASEM / ASEM-v3** (session-level atomic-fact ingestion with zero-loss temporal
grounding, deterministic gating, multi-channel RRF retrieval, and direct temporal QA).

**Primary source files**

| Concern | ASEMv2 | FastASEM |
|---|---|---|
| Ingestion | `asem/batch_ingestion.py` → `BatchIngestor` | `asem/fast_ingest.py` → `FastSessionIngestor` (SLAFI) |
| System wrapper | `eval/systems.py` → `ASEMSystemV2` | `eval/systems.py` → `FastASEMSystem` |
| Builder | `build_asem_v2_system()` | `build_fast_asem_system()` |
| Retriever | `asem/enhanced_retriever.py` → `EnhancedHybridRetriever` | `asem/retriever.py` → `HybridRetriever` (RRF mode) |
| Answer prompt | `_RETRIEVAL_PROMPT` (distil mode) | `data/prompts/P_temporal_qa.txt` (direct mode) |
| Note schema | base 9 fields | base 9 + `session_id`, `session_date`, `timestamp_iso`, `entities`, `speaker` |
| Config | generic | `configs/presets/sota_benchmark.yaml` |

---

## 1. TL;DR — what changed and why

ASEMv2 was **correct but slow and temporally blind**. FastASEM keeps the same five-stage
pipeline skeleton but replaces the three most expensive / lossy stages:

1. **Ingestion:** 3 LLM calls/session → **1 LLM call/session** + deterministic (<1 ms)
   gating and graph weaving.
2. **Temporal grounding:** notes now carry the **session date** and the extraction prompt
   **resolves relative time to absolute dates** ("last week" → "7 May 2023"). This is the
   single biggest accuracy win.
3. **Retrieval:** single dense-similarity + utility blend → **multi-channel RRF**
   (dense + BM25 + entity + temporal-boost) with link traversal.
4. **Answering:** distillation mode → **direct mode** with a temporal-aware QA prompt and
   `include_dates=True`.

Net effect on LoCoMo conv-26 (117 Qs, same metric functions): overall EM **17.9 → 29.1**,
ROUGE-L **19.3 → 43.6**, and Temporal-Reasoning EM **2.7 → 43.2** (see
`fairplay_locomo10_conv26_analysis.md`).

---

## 2. Pipeline-level architecture

Both systems use the same `ASEMPipeline` five-stage backbone
(`asem/pipeline.py`):

```
Content → [S1 NoteConstruction] → [S2 MemoryManager] → [S3 LinkEvolver]
Query   → [S4 Retriever → AnswerAgent] → [S5 UtilityUpdater]
```

The difference is **which concrete component is injected** at each stage via constructor
injection, and **how ingestion is driven**:

| Stage | ASEMv2 | FastASEM |
|---|---|---|
| S1 Note construction | `NoteConstructor` (batch prompt P1_batch) | `NoteConstructor` (same) — but facts come from SLAFI |
| S2 Memory ops | **LLM** batch ADD/UPDATE/DELETE/NOOP | **Deterministic** cosine gating (no LLM) |
| S3 Link evolution | **LLM** batch link generation | **Deterministic** entity + temporal weaving (no LLM) |
| S4 Retrieval | `EnhancedHybridRetriever` | `HybridRetriever` (RRF + BM25 + entity + temporal) |
| S4 Answer | `AnswerAgent` distil mode | `AnswerAgent` **direct mode** + temporal QA prompt |
| S5 Utility | `UtilityUpdater` (EMA) | `UtilityUpdater` (EMA, unchanged) |
| Write gate | none | `WriteGate` (tau_high / tau_redund) |

The two system wrappers differ in their ingestion entry point:

```python
# ASEMv2 — flat turn list, NO session dates
class ASEMSystemV2:
    def ingest_conversation(self, dialogue_turns: List[str]) -> int: ...

# FastASEM — list of sessions WITH explicit dates
class FastASEMSystem:
    def ingest_conversation(self, sessions: List[Dict]) -> int:
        # each session = {"turns": [...], "date": "8 May 2023", "session_id": "sess_3"}
```

This API change is what makes temporal grounding possible: FastASEM receives the
**session timestamp** at ingestion time, whereas ASEMv2 discards it.

---

## 3. Ingestion: `BatchIngestor` → `FastSessionIngestor` (SLAFI)

This is the core of the upgrade. "SLAFI" = **S**ession-**L**evel **A**tomic **F**act
**I**ngestion.

### 3.1 LLM call budget per session

| Step | ASEMv2 (`BatchIngestor`) | FastASEM (`FastSessionIngestor`) |
|---|---|---|
| Note extraction | 1 LLM call (P4 batch extraction) | 1 LLM call (SLAFI extraction prompt) |
| Memory ops (ADD/UPDATE/DELETE/NOOP) | **1 LLM call** (P5) | **0** — deterministic cosine gate |
| Link generation | **1 LLM call** (P6) | **0** — deterministic weaving |
| **Total** | **3 LLM calls/session** | **1 LLM call/session** |

For conv-26 (19 sessions) this is ~57 → ~19 LLM calls for ingestion, a ~3× reduction in
ingestion latency and API cost.

#### Step-by-step flow

**ASEMv2 `BatchIngestor.ingest_conversation(turns, bank)`** (6 steps, 3 LLM calls):

1. **Extract** — one LLM call (P4) over the whole session dialogue → list of
   `{content, keywords, tags, description}`.
2. **Embed** — `_embed_notes()` builds `Note` objects; `e = embed(c+K+G+X)`,
   `z = embed(c)`, `t = datetime.utcnow()`.
3. **Memory ops** — one LLM call (P5) over new notes + first 20 existing notes →
   `{index, op, target_id}` per note.
4. **Execute** — `_execute_ops()` applies ADD / UPDATE(overwrite) / DELETE / NOOP.
5. **Link** — one LLM call (P6) over added notes + ≤20 ANN neighbors → free-form
   `{source, target, relation}` triples, applied bidirectionally.
6. **Rebuild** — `memory_bank._rebuild_index()` (FAISS) once.

**FastASEM `FastSessionIngestor.ingest_session(turns, session_date_str, session_id, bank)`**
(4 steps, 1 LLM call):

1. **Parse date** — `parse_session_datetime(session_date_str)` → `(dt_obj, iso_str)`.
2. **Extract** — one LLM call (SLAFI prompt) → list of
   `{fact, entities, keywords, tags, speaker}` with absolute dates already resolved.
   On JSON parse failure, `_fallback_extract()` produces per-line facts stamped with
   `(Date: {session_date})`.
3. **Build + gate** — for each fact, build the `Note` (with `session_date`,
   `timestamp_iso`, `entities`, `speaker`) and run the deterministic cosine gate
   (§3.3) to decide ADD / UPDATE / NOOP. No LLM.
4. **Weave** — `_weave_graph_links()` adds deterministic entity + temporal/semantic edges
   (§3.4). No LLM.

The net effect: the two LLM calls that ASEMv2 spends on *deciding* memory operations and
*inventing* links are replaced by deterministic, sub-millisecond logic, and the one LLM
call that remains is spent entirely on *extracting well-grounded atomic facts*.

### 3.2 Temporal grounding (the key accuracy fix)

This is the single most impactful change. It has three cooperating parts:
(a) parse the session timestamp into a real `datetime`, (b) instruct the LLM to rewrite
relative time into absolute dates, and (c) stamp every note with the session date so
retrieval and answering can use it.

#### (a) The ASEMv2 defect — wall-clock stamping, no session date

ASEMv2's `BatchIngestor` never receives a session date. Its `ingest_conversation()` takes a
flat `List[str]` of turns and `_embed_notes()` stamps each note with the *current*
wall-clock time:

```python
# asem/batch_ingestion.py — _embed_notes()
note = Note(
    id=str(uuid.uuid4()),
    c=c,
    t=datetime.utcnow(),          # <-- wall-clock "now", NOT the session date
    K=K, G=G, X=X,
    e=e_vec, L=[], z=z_vec, q=self._q0,
    # no session_id / session_date / timestamp_iso / entities / speaker
)
```

Consequences:
- A turn "I went to the support group **last week**" (from a session dated 8 May 2023) is
  stored with `t = <run date>` and the relative phrase "last week" intact in `c`.
- There is no `session_date` field, so the retriever has nothing to boost on and the answer
  agent has no absolute date to surface. It echoes "last week", which fails against the
  LoCoMo reference "7 May 2023".

#### (b) FastASEM — parse the session timestamp

`parse_session_datetime()` converts the human-readable LoCoMo timestamp into a real
`datetime` plus an ISO-8601 string. It tries ISO first, then a set of `strptime` formats
for the "1:56 pm on 8 May, 2023" shape, and falls back to current-UTC:

```python
# asem/fast_ingest.py
def parse_session_datetime(timestamp_str: str) -> Tuple[datetime, Optional[str]]:
    # "1:56 pm on 8 May, 2023" -> (datetime(2023,5,8,13,56), "2023-05-08T13:56:00Z")
    raw = timestamp_str.strip()
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt, dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        pass
    parts = raw.lower().split(" on ")
    if len(parts) == 2:
        datetime_str = f"{parts[1].strip().replace(',','')} {parts[0].strip()}"
        for fmt in ["%d %B %Y %I:%M %p", "%d %b %Y %I:%M %p",
                    "%B %d %Y %I:%M %p", "%b %d %Y %I:%M %p",
                    "%Y-%m-%d %I:%M %p"]:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                return dt, dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue
    now = datetime.now(timezone.utc)
    return now, now.strftime("%Y-%m-%dT%H:%M:%SZ")
```

#### (c) The extraction prompt resolves relative → absolute

The SLAFI system prompt (`_EXTRACTION_SYSTEM_PROMPT`) carries five CRITICAL RULES. Rule 2
is the temporal one:

> **RESOLVE RELATIVE TIME TO ABSOLUTE DATES:** The session timestamp is given below.
> Convert every relative time expression into an absolute date using it:
> - "yesterday" → the day before the session date
> - "last week" / "a few days ago" → an approximate earlier date in the same month
> - "next month" / "next week" → the following month/week
> - "last year" → the previous calendar year
> Always state the resolved date explicitly in the fact
> (e.g., "On 7 May 2023, Caroline went to the LGBTQ support group.").

The user prompt injects the session date so the model has the anchor:

```python
_EXTRACTION_USER_PROMPT = """Session Timestamp: {session_date}
Dialogue:
{dialogue}

Extract all key factual statements from this session:"""
```

**Worked example.** Session dated `8 May 2023`, turn:
`[Caroline] I went to the LGBTQ support group last week, it was really helpful.`

- ASEMv2 stores: `c = "Caroline went to the LGBTQ support group last week"`,
  `t = <run date>`, no `session_date`.
- FastASEM stores: `c = "On 1 May 2023, Caroline went to the LGBTQ support group."`,
  `t = 2023-05-08`, `session_date = "8 May 2023"`,
  `timestamp_iso = "2023-05-08T13:56:00Z"`, `entities = ["Caroline", "LGBTQ support group"]`,
  `speaker = "Caroline"`.

The other four rules reinforce this:
1. **RESOLVE PRONOUNS** — "I went there" → "Caroline went to Hawaii" (so facts are
   self-contained and entity-searchable).
3. **KEEP FACTS CONCRETE AND ATOMIC** — one event = one note; a past event and a future
   plan are two notes; relationship/status facts ("I'm single") get their own note.
4. **PRESERVE SPECIFICITY** — "went to a LGBTQ support group" ≠ "volunteers at a LGBTQ
   youth center".
5. **Extract entities/keywords/tags** per note (feeds the entity channel and the gate).

The note is then built with the session date stamped in:

```python
# asem/fast_ingest.py — ingest_session(), step 2
note = Note(
    id=str(uuid.uuid4()),
    c=fact,
    t=dt_obj,                     # <-- the SESSION date, not wall-clock
    K=keywords, G=tags, X=fact,
    e=e_vec,                      # joint embed of fact+K+G+entities
    L=[], z=z_vec, q=self._q0,    # z = embed(fact) (intent embedding)
    session_id=session_id,
    session_date=session_date_str,
    timestamp_iso=iso_str,
    entities=entities,
    speaker=speaker,
)
```

### 3.3 Deterministic memory gating (replaces LLM memory ops)

#### ASEMv2: LLM-decided batch ops

ASEMv2's `_batch_memory_ops()` serializes the new notes **plus only the first 20 existing
notes** into a prompt and asks the LLM to return one `{index, op, target_id}` per new note,
where `op ∈ {ADD, UPDATE, DELETE, NOOP}`:

```python
# asem/batch_ingestion.py — _batch_memory_ops()
existing_payloads = [
    {"id": n.id, "keywords": n.K, "tags": n.G, "description": n.X}
    for n in existing[:20]  # cap at 20 for prompt size
]
prompt = self._memory_ops_prompt.format(
    new_notes=json.dumps(new_payloads),
    existing_memory=json.dumps(existing_payloads) if existing_payloads else "[]",
)
```

Problems with this design:
- **Context cap.** Only 20 existing notes are visible, so as the bank grows the LLM cannot
  see the true duplicate and tends to ADD.
- **Non-deterministic.** Same input can yield different ops across runs (temperature,
  sampling), so the bank is not reproducible.
- **UPDATE overwrites.** `_execute_ops()` replaces the target's `c/K/G/X/e` with the new
  note's values, **losing the original fact**:

```python
# asem/batch_ingestion.py — _execute_ops(), UPDATE branch
merged = Note(
    id=target.id,
    c=note.c,        # <-- original target.c is discarded
    t=note.t, K=note.K, G=note.G, X=note.X, e=note.e,
    L=target.L, z=note.z, q=target.q,
)
```

- **DELETE is destructive** and LLM-judged — a wrong call removes a note permanently.

#### FastASEM: deterministic cosine gate

FastASEM replaces the LLM with a sub-millisecond gate against the top-3 ANN neighbors of
the new note's **intent embedding `z`** (the embedding of the fact text alone, not the
joint embedding):

```python
# asem/fast_ingest.py — ingest_session(), step 3
existing = memory_bank.ann_search(note.z, k=3)
if not existing:
    memory_bank.add(note); added_notes.append(note); continue

top_ex  = max(existing, key=lambda ex: float(self._cosine(note.z, ex.z)))
max_sim = float(self._cosine(note.z, top_ex.z))

if max_sim >= self._tau_redund:            # 0.90 -> NOOP
    continue
elif max_sim >= self._tau_novel:           # 0.75 -> UPDATE (conditional)
    speaker_names = {note.speaker.lower(), top_ex.speaker.lower()} - {None}
    overlap = {e for e in (set(note.entities) & set(top_ex.entities))
               if e.lower() not in speaker_names}
    if overlap:
        ...merge (append) into top_ex...
        continue
memory_bank.add(note); added_notes.append(note)   # default -> ADD
```

Decision table (thresholds from `sota_benchmark.yaml`: `tau_high=0.75`, `tau_redund=0.90`):

| Condition | Action | Rationale |
|---|---|---|
| `max_sim ≥ 0.90` | **NOOP** | near-duplicate; skip |
| `0.75 ≤ max_sim < 0.90` **and** shared non-speaker entity | **UPDATE** (append) | same event, new detail |
| `0.75 ≤ max_sim < 0.90` **and** only speaker overlap | **ADD** | distinct facts about same person |
| `max_sim < 0.75` | **ADD** | novel fact |

Two design choices matter:

- **Speaker-excluded entity overlap.** Two facts that share only the dominant speaker
  (e.g. both about "Caroline") are *not* merged — this preserves atomic-fact granularity.
  This is why `tau_high` was raised from 0.40 to **0.75** (see config comment): at 0.40,
  LoCoMo's 2-speaker structure made entity overlap almost always non-empty and distinct
  facts were being collapsed into one note.
- **Append, don't overwrite.** On UPDATE the new fact is *appended* to the existing note's
  description rather than replacing the headline `c`, and the joint embedding `e` is
  recomputed from **both** facts so Phase-A retrieval reflects the merge:

```python
# asem/fast_ingest.py — UPDATE branch
base_desc = top_ex.X if (top_ex.X and top_ex.X != top_ex.c) else top_ex.c
merged_desc = f"{base_desc} | {note.c}" if note.c not in base_desc else base_desc
e_text = " ".join([top_ex.c, note.c,
                   " ".join(merged_keywords), " ".join(merged_tags),
                   " ".join(merged_entities)])
merged_e = self._backend.embed(e_text)
updated_note = Note(
    id=top_ex.id, c=top_ex.c, t=top_ex.t,   # original headline + timestamp preserved
    K=merged_keywords, G=merged_tags, X=merged_desc, e=merged_e,
    L=top_ex.L, z=top_ex.z, q=top_ex.q,
    session_id=top_ex.session_id, session_date=top_ex.session_date,
    timestamp_iso=top_ex.timestamp_iso,
    entities=merged_entities,
    speaker=top_ex.speaker or note.speaker,
)
```

Note the asymmetry vs ASEMv2: the original `c`, `t`, `session_date`, and `q` are all
preserved; only `X` (description) grows and `e` is recomputed. No fact is ever lost, and
there is no destructive DELETE path.

### 3.4 Deterministic graph weaving (replaces LLM link generation)

#### ASEMv2: LLM free-form pairwise relations

ASEMv2's `_batch_link()` gathers each added note's top-10 ANN neighbors (capped at 20
neighbor payloads for the prompt) and asks the LLM to emit free-form
`{source, target, relation}` triples. The relation vocabulary is open-ended
(`validate_link_array(..., allow_unknown_relations=True)`), so the LLM may invent
semantic/causal/temporal relations. It is then applied bidirectionally and cross-session
links are counted. This is a third LLM call per session, is non-deterministic, and is
bounded by the 20-neighbor prompt cap.

#### FastASEM: two deterministic edge rules

`_weave_graph_links()` replaces the LLM with two cheap, reproducible rules:

```python
# asem/fast_ingest.py — _weave_graph_links()
for n in new_notes:
    # 1. Entity co-occurrence
    if n.entities:
        for match in memory_bank.search_by_entities(n.entities, k=5):
            if match.id != n.id:
                self._add_edge(n, match, relation="same-entity", bank=memory_bank)
    # 2. Dense semantic / temporal adjacency
    for neighbor in memory_bank.ann_search(n.e, k=4):
        if neighbor.id != n.id:
            sim = self._cosine(n.e, neighbor.e)
            if sim >= 0.55:
                rel = "temporal" if n.session_id == neighbor.session_id else "semantic"
                self._add_edge(n, neighbor, relation=rel, bank=memory_bank)
```

| Rule | Trigger | Relation label |
|---|---|---|
| Entity co-occurrence | shares a named entity (top-5 entity search) | `same-entity` |
| Dense adjacency | joint-embedding cosine ≥ 0.55 (top-4 ANN) | `temporal` (same session) / `semantic` (cross-session) |

`_add_edge()` is idempotent and bidirectional — it adds a `LinkRecord` to both notes and
persists via `bank.update()`, skipping if the edge already exists:

```python
@staticmethod
def _add_edge(n1, n2, relation, bank):
    if not any(l.target_id == n2.id for l in n1.L):
        n1.L.append(LinkRecord(target_id=n2.id, relation=relation))
        bank.update(n1.id, {"L": n1.L})
    if not any(l.target_id == n1.id for l in n2.L):
        n2.L.append(LinkRecord(target_id=n1.id, relation=relation))
        bank.update(n2.id, {"L": n2.L})
```

The closed relation vocabulary (`same-entity`, `temporal`, `semantic`) is a deliberate
trade-off: it loses the LLM's open-ended semantic/causal relations but gains determinism,
zero latency, and no prompt-size cap. The `temporal` label (same-session adjacency) is what
feeds the retriever's temporal-boost channel downstream.

### 3.5 Graceful fallback

If the LLM extraction JSON fails to parse, FastASEM falls back to a heuristic per-line
extractor that still stamps each fact with the session date
(`f"{content} (Date: {session_date})"`), so temporal grounding degrades gracefully instead
of producing an empty bank.

---

## 4. Note schema enrichment

`asem/note.py` gained five optional fields (lines 118–122) that FastASEM populates and
ASEMv2 leaves `None`:

```python
session_id:    Optional[str] = None   # which session the note came from
session_date:  Optional[str] = None   # human-readable session date
timestamp_iso: Optional[str] = None   # ISO-8601 absolute timestamp
entities:      List[str] = field(default_factory=list)  # named entities
speaker:       Optional[str] = None   # utterance speaker
```

These are serialized/deserialized in `to_dict()` / `from_dict()` (lines 136–160) and are
what enable the four downstream capabilities. The table below maps each new field to the
code that consumes it:

| Field | Populated by | Consumed by | Effect |
|---|---|---|---|
| `session_id` | SLAFI step 2 | `_weave_graph_links` (temporal vs semantic label), `HybridRetriever` | same-session adjacency → `temporal` edges |
| `session_date` | SLAFI step 2 | `HybridRetriever` temporal-boost, `AnswerAgent.direct_answer` date prefix | temporal-boost channel; `[8 May 2023]` prefix in QA context |
| `timestamp_iso` | `parse_session_datetime` | `HybridRetriever` temporal-boost (fallback when `session_date` empty) | absolute-time retrieval signal |
| `entities` | SLAFI rule 5 | `search_by_entities` (gate + retrieval + weaving) | entity channel; speaker-excluded dedup; `same-entity` edges |
| `speaker` | SLAFI rule 1 | gate (speaker-excluded overlap), `direct_answer` | preserves atomicity; pronoun-resolved facts |

Because the fields are optional with `None`/empty defaults, ASEMv2 banks (which never set
them) remain fully backward-compatible — the new retrieval channels simply find nothing to
boost when the fields are absent, which is exactly why ASEMv2's temporal performance is
weak.

---

## 5. Retrieval: `EnhancedHybridRetriever` → `HybridRetriever`

> **Important nuance.** `EnhancedHybridRetriever` *subclasses* `HybridRetriever` and the
> ASEMv2 builder does **not** set `use_rrf=False`, so **both** systems run the *same*
> multi-channel RRF Phase A (dense + BM25 + entity + temporal-boost) and the same
> z-scored Phase B re-rank. The genuine differences are the **post-RRF phases**:
> ASEMv2 adds a true N-hop BFS expansion and a global-graph (community + PageRank)
> re-rank; FastASEM stops at a 1-hop link traversal. The temporal advantage of FastASEM
> therefore comes mainly from **better notes** (absolute dates, §3.2) feeding the shared
> temporal-boost channel, not from a different Phase A.

| Phase | ASEMv2 (`EnhancedHybridRetriever`) | FastASEM (`HybridRetriever`) |
|---|---|---|
| **A — recall** | RRF: dense + BM25 + entity + temporal-boost (inherited) | RRF: dense + BM25 + entity + temporal-boost |
| **B — re-rank** | z-score `(1−λ)·sim + λ·q`, top `k2` (inherited) | z-score `(1−λ)·sim + λ·q`, top `k2` |
| **C — link expand** | **N-hop BFS** (`max_hops=2`, `hop_decay=0.7`, relation-type weights, intent-grounded Q), top 5 | **1-hop** `_traverse_links` (`score = sim·(0.5+0.5·q)`), top 3 |
| **D — global re-rank** | **Louvain community + PageRank**, `hybrid = α·local + β·global + γ·intent-Q` | *(none)* |

The RRF formulation is robust to the different scales of the four channels (dense cosine,
BM25 TF-IDF, entity overlap, temporal proximity) — each channel produces a ranked list and
RRF combines them by `Σ 1/(rrf_k + rank)`, avoiding the magnitude-dominance problem that a
raw weighted sum has. The **temporal-boost** channel directly rewards notes whose
`timestamp_iso` is close to the time frame the query is about. Because FastASEM's notes
carry absolute dates (§3.2) while ASEMv2's do not, the *same* temporal-boost term is far
more effective on FastASEM's bank — which is what makes Temporal-Reasoning retrieval (and
hence answering) dramatically better.

#### How `HybridRetriever.retrieve()` actually works

The method is two-phase. **Phase A** builds a candidate pool from up to four channels and
fuses them with RRF; **Phase B** re-ranks the top `k1` by a z-scored blend of RRF score and
utility `q`, then expands via link traversal.

```python
# asem/retriever.py — retrieve()
# Phase A: multi-channel recall
dense_candidates = M.ann_search(e_q, k=self.k1)                 # channel 1
dense_ranks = {n.id: r for r, n in enumerate(dense_candidates)}
all_pool_map = {n.id: n for n in dense_candidates}

if self.use_bm25:                                               # channel 2
    for r, (_, n) in enumerate(M.bm25_search(query, k=self.k1)):
        bm25_ranks[n.id] = r; all_pool_map[n.id] = n

if self.use_entity_filter:                                      # channel 3
    query_entities = re.findall(r"\b[A-Z][a-z0-9_-]+\b", query)
    if query_entities:
        for r, n in enumerate(M.search_by_entities(query_entities, k=self.k1)):
            entity_ranks[n.id] = r; all_pool_map[n.id] = n

is_temp = bool(re.search(
    r"\b(when|before|after|date|time|year|month|day|during|first|last)\b",
    query, re.I))

# RRF fusion + temporal boost
for nid, note in all_pool_map.items():
    score = 0.0
    if nid in dense_ranks:  score += self.dense_weight    / (self.rrf_k + dense_ranks[nid])
    if nid in bm25_ranks:   score += self.bm25_weight     / (self.rrf_k + bm25_ranks[nid])
    if nid in entity_ranks: score += self.entity_weight   / (self.rrf_k + entity_ranks[nid])
    if self.use_temporal_boost and is_temp and (note.session_date or note.timestamp_iso):
        score += self.temporal_weight / (self.rrf_k + 0)   # flat boost for temporal notes
    sim = self._cosine(e_q, note.e)
    if sim >= self.delta or nid in bm25_ranks or nid in entity_ranks:
        rrf_scores.append((note, score))
```

Key details:
- **Temporal boost is a flat rank-0 term.** For a temporal query, any note carrying a
  `session_date`/`timestamp_iso` gets `temporal_weight / rrf_k` added regardless of its
  dense rank — this is what pulls date-bearing notes up for "when" questions.
- **Entity channel** extracts capitalized tokens from the query and matches them against
  note `entities` (populated by SLAFI rule 5). This is why pronoun-resolved, entity-tagged
  notes are retrieved for name-based questions.
- **BM25 channel** catches exact keyword matches that dense embeddings miss (e.g. a
  specific place or number).
- **Phase B** then z-scores the RRF scores and the `q` values within the candidate pool and
  blends `(1−λ)·sim_norm + λ·q_norm` to pick the top `k2`, finally appending notes reached
  by `_traverse_links()` (multi-hop expansion, `max_hops=2`, `hop_decay=0.75`).

By contrast, ASEMv2's `EnhancedHybridRetriever` runs the *same* RRF Phase A but then adds
two extra phases: a **true N-hop BFS** (`_multi_hop_expand`, up to 2 hops with per-hop
decay and relation-type edge weights) and a **global-graph re-rank** (Louvain community
membership + PageRank centrality, blended as `α·local + β·global + γ·intent-Q`). These
help multi-hop / inference questions but do nothing for "when" questions, because the
temporal-boost channel they share is fed by notes that (in ASEMv2) carry no absolute date.

Builder wiring (`build_fast_asem_system`):

```python
retriever = HybridRetriever(
    backend=backend,
    k1=hp.k1, k2=hp.k2,
    delta=hp.delta, lambda_weight=hp.lambda_weight,
    use_rrf=True,
    use_bm25=rt_cfg.use_bm25,
    use_entity_filter=rt_cfg.use_entity_filter,
    use_temporal_boost=rt_cfg.use_temporal_boost,
    dense_weight=rt_cfg.dense_weight,
    bm25_weight=rt_cfg.bm25_weight,
    entity_weight=rt_cfg.entity_weight,
    temporal_weight=rt_cfg.temporal_weight,
    rrf_k=rt_cfg.rrf_k,
    max_link_hops=rt_cfg.max_hops,
    enable_link_traversal=True,
)
```

---

## 6. Answering: distil mode → direct mode + temporal QA prompt

| Aspect | ASEMv2 | FastASEM |
|---|---|---|
| Mode | distillation (select + rewrite candidates) | **`direct_mode=True`** |
| Baseline prompt | `_RETRIEVAL_PROMPT` (generic) | `data/prompts/P_temporal_qa.txt` |
| Dates | not emphasized | `include_dates=True` |
| Context notes | up to `k2` | `max_context_notes=8` |
| Temperature | 0.1 | **0.0** (deterministic) |

The temporal QA prompt (`P_temporal_qa.txt`) is tuned for the benchmark's terse reference
format:

> 1. Answer concisely: a few words or one short sentence. No preamble, no quotes.
> 2. Rely strictly on the facts in the memory notes.
> 3. **For time/date questions, provide the exact date, month, year, or relative
>    timeframe mentioned in the notes.**
> 4. If no note is relevant, answer "I don't know".

#### How `direct_mode` builds the context

In `AnswerAgent.distil_and_answer()`, `direct_mode=True` short-circuits the JSON
distillation step and calls `direct_answer()`, which formats the retrieved notes into a
chronologically-sorted, date-prefixed context:

```python
# asem/answer_agent.py — direct_answer()
sorted_notes = sorted(candidates, key=lambda n: n.t if n.t else datetime.min)
for n in sorted_notes:
    date_prefix = f"[{n.session_date}] " if n.session_date \
                  else f"[{n.t.strftime('%d %B %Y')}] "
    entities_str = f" (Entities: {', '.join(n.entities)})" if n.entities else ""
    keywords_str = f" (Keywords: {', '.join(n.K[:12])})" if n.K else ""
    desc_str     = f" (Description: {n.X})" if (n.X and n.X != n.c) else ""
    context_items.append(f"- {date_prefix}{n.c}{entities_str}{keywords_str}{desc_str}")
context = "\n".join(context_items)
prompt = self.baseline_prompt_template.format(query=query, context=context)
return self._generate_resilient(prompt).strip()
```

Three things make this temporally effective:
- **Chronological sort** by `n.t` (the session date) gives the model a coherent timeline.
- **Date prefix** `[8 May 2023]` on every note makes the absolute date explicit in context.
- **Merged-fact visibility.** Because SLAFI UPDATE *appends* facts into `X` (description)
  while `c` keeps a single headline, `direct_answer()` also surfaces `K` and `X` — so facts
  folded into a merged note are still visible to the model. ASEMv2's overwrite-UPDATE would
  have hidden the original fact entirely.

By contrast, ASEMv2 runs **distillation mode**: it first asks the LLM to select a subset of
candidate notes (a JSON `{selected_ids, answer}`), then answers. That extra LLM call is
slower and, with relative-time notes, the distilled context still contains "last week" —
so the final answer echoes the relative phrase.

Because the notes now contain **absolute dates** (from §3.2) and the prompt is instructed to
surface them, FastASEM emits "7 May 2023" where ASEMv2 emitted "last week" — the direct
cause of the Temporal-Reasoning EM jump (2.7 → 43.2).

---

## 7. Write gate

FastASEM introduces a `WriteGate` (`asem/write_gate.py`), a **Tier-0 deterministic
short-circuit** that sits in front of the S2 Memory-Manager LLM on the *per-turn*
`write_path` (`asem/pipeline.py`, lines 52 and 151). It computes novelty from the raw
content embeddings `z` and only lets the **ambiguous band** fall through to the LLM:

```python
# asem/write_gate.py — propose()
max_sim = max(self._cosine(note.z, c.z) for c in candidates)
novelty = 1.0 - max_sim
if novelty >= self.tau_high:      # clearly new topic
    return Op.ADD, max_sim        # LLM skipped
if max_sim >= self.tau_redund:    # near-verbatim duplicate
    return Op.NOOP, max_sim       # LLM skipped
return None, max_sim              # ambiguous -> Memory Manager LLM decides
```

```python
# asem/pipeline.py — write_path()
gate_op, _ = self.write_gate.propose(note, existing)
if gate_op is not None:
    op, target = gate_op, None                 # deterministic, LLM skipped
else:
    op, target = self.memory_manager.select_op(content, existing)
    self.write_gate.record_ambiguous_llm(op)   # audit the LLM's choice
```

Key properties:
- **Conservative by design.** Only ADD and NOOP are gated; **UPDATE/DELETE always stay with
  the LLM** (contradictions and evolution need semantic judgment).
- **Auditable.** `stats` counts `gate_add`, `gate_noop`, `ambiguous`, and the LLM's verdicts
  inside the ambiguous band (`amb_add/update/delete/noop`), so the thresholds can be
  calibrated from data rather than guessed.
- **Cost saving.** The S2 LLM is the most expensive per-turn stage; the gate removes it for
  the unambiguous majority of turns.

**Relationship to the SLAFI gate (§3.3).** The two gates are related but distinct:
- The **SLAFI ingestor gate** (§3.3) is *fully deterministic* — it has no LLM fallback and
  makes the final ADD/UPDATE/NOOP decision itself (with the speaker-excluded entity-overlap
  rule). It is what actually runs during the LoCoMo benchmark ingestion.
- The **`WriteGate`** is a *Tier-0 pre-filter* on the per-turn path that defers the
  ambiguous band to the Memory-Manager LLM. It shares the same `tau_high`/`tau_redund`
  config values (`0.75`/`0.90` in the preset) so the ADD/NOOP boundary is consistent across
  both paths.

ASEMv2 had neither gate — every write decision was made by the LLM.

---

## 8. Configuration: `configs/presets/sota_benchmark.yaml`

The FastASEM preset introduces sections ASEMv2 did not use:

```yaml
write_gate:
  enabled: true
  tau_high: 0.75      # UPDATE threshold (was 0.40 -> over-merged; see §3.3)
  tau_redund: 0.90    # NOOP threshold

retriever:
  mode: "rrf"
  use_bm25: true
  use_entity_filter: true
  use_temporal_boost: true
  max_hops: 2
  hop_decay: 0.75
  rrf_k: 60
  dense_weight: 1.0
  bm25_weight: 0.9
  entity_weight: 0.7
  temporal_weight: 0.6

answer:
  direct_mode: true
  include_dates: true
  max_context_notes: 8
  max_tokens: 350
  temperature: 0.0

ingestion:
  mode: "session_batch"
  lazy_evolution: true
  link_tau: 0.30
  max_notes_per_session: 20
```

Shared hyperparameters (k1=30, k2=8, k=6, δ=0.25, λ=0.35, α=0.10, q0=0.50) and the backend
(langchain / openai / `gpt-5.4`, temp 0.1, embedder `all-MiniLM-L6-v2`) are unchanged.

---

## 9. Measured impact (LoCoMo conv-26, 117 Q, identical metric functions)

| Metric | ASEMv2 | FastASEM | Δ |
|---|:---:|:---:|:---:|
| Overall EM% | 17.9 | **29.1** | +11.2 |
| Overall ROUGE-L% | 19.3 | **43.6** | +24.3 |
| Overall BERT-F1% | 82.9 | **88.2** | +5.3 |
| Overall Judge% | 65.8 | 70.1 | +4.3 |
| **Temporal EM%** | 2.7 | **43.2** | **+40.5** |
| Temporal ROUGE-L% | 8.7 | **60.6** | +51.9 |

The upgrade's benefit is **concentrated in Temporal Reasoning** — exactly the capability the
temporal-grounding + temporal-boost + temporal-QA-prompt changes target. On
Conversational/Single-Hop, ASEMv2 and FastASEM are closer (and both trail FullContext),
because those categories reward fluent paraphrase over precise date matching.

#### Per-category EM% (ASEMv2 → FastASEM)

| Category (n) | ASEMv2 | FastASEM | Δ | Driver |
|---|:---:|:---:|:---:|---|
| Temporal (37) | 2.7 | **43.2** | **+40.5** | absolute-date notes + temporal-boost + temporal QA prompt |
| Conversational (35) | 42.9 | 31.4 | −11.5 | ASEMv2's LLM distillation paraphrases better here |
| Single-Hop (32) | 12.5 | 18.8 | +6.3 | entity channel + atomic facts |
| Multi-Hop (13) | 7.7 | 7.7 | 0.0 | both weak; needs deeper link traversal |

The Conversational regression is the main cost of the upgrade: ASEMv2's LLM distillation
step is better at producing fluent, reference-matching paraphrases, whereas FastASEM's
direct mode favors terse, date-anchored answers. This is a deliberate bias toward the
benchmark's exact-match scoring rather than a defect.

---

## 10. Trade-offs / what was given up

- **LLM-driven memory ops removed.** ASEMv2's LLM could in principle perform *semantic*
  DELETE/UPDATE that a cosine gate cannot. FastASEM trades that expressiveness for speed,
  determinism, and reproducibility. (Mitigation: append-not-overwrite UPDATE preserves
  information; DELETE is no longer performed at ingest.)
- **LLM link semantics removed.** Free-form semantic/causal/temporal relations from the LLM
  are replaced by entity + dense-adjacency edges. Richer relation types are lost, but the
  graph is now deterministic and cheap.
- **Atomicity bias.** The speaker-excluded overlap rule + high `tau_high` deliberately keep
  facts separate; this can grow the bank (more notes) versus aggressive merging.
- **Conversational regression.** Direct mode + terse QA prompt trades some conversational
  fluency for date precision (−11.5 EM on Conversational, §9).
- **Single-conversation evidence.** The numbers above are from conv-26 only; the temporal
  gain should be validated across all 10 LoCoMo conversations before claiming it generalizes.

#### What was gained (for balance)

- **~3× faster ingestion** (1 vs 3 LLM calls/session) and lower API cost.
- **Reproducibility** — deterministic gating and weaving mean the same input yields the
  same bank, which ASEMv2's LLM ops could not guarantee.
- **No information loss on UPDATE** — append-not-overwrite keeps the original fact visible.
- **Temporal capability** — the single largest accuracy gain (Temporal EM 2.7 → 43.2).

---

## 11. Pseudocode of the core algorithms

Notation: `M` = memory bank; `embed(x)` = dense embedding; `cos(a,b)` = cosine similarity;
`ANN(v,k)` = top-k approximate-nearest-neighbors of vector `v`; `LLM(prompt)` = one model
call. Each algorithm is given for both systems.

### 11.1 Ingestion — ASEMv2 (`BatchIngestor.ingest_conversation`)

```
ALGORITHM ASEMv2-INGEST(turns, M)
  dialogue ← join(turns)

  # ---- Step 1: batch note extraction (LLM call #1) ----
  facts ← LLM( P4_extraction_prompt(dialogue) )        # [{content,keywords,tags,description}]
  if facts = ∅: facts ← fallback_per_line(dialogue)

  # ---- Step 2: embed notes (NO session date) ----
  notes ← []
  for f in facts:
      c ← f.content
      e ← embed( c + f.keywords + f.tags + f.description )   # joint embedding
      z ← embed( c )                                          # intent embedding
      notes.append( Note(c, t=NOW_UTC, K=f.keywords, G=f.tags,
                         X=f.description, e, z, q=q0) )      # t = wall-clock "now"

  # ---- Step 3: batch memory ops (LLM call #2) ----
  existing_payload ← first 20 notes of M
  ops ← LLM( P5_memory_ops_prompt(notes, existing_payload) ) # [{index,op,target_id}]
  # op ∈ {ADD, UPDATE, DELETE, NOOP}; on parse failure → ADD all

  # ---- Step 4: execute ops ----
  added ← []
  for (i, op) in ops:
      n ← notes[i]
      if op = ADD:            M.add(n); added.append(n)
      elif op = UPDATE:       M.add( overwrite(M.get(target_id), n) )   # loses original c
      elif op = DELETE:       M.delete(target_id)
      elif op = NOOP:         pass

  # ---- Step 5: batch link generation (LLM call #3) ----
  neighbors ← top-10 ANN of each added note (≤20 payloads)
  rels ← LLM( P6_link_prompt(added, neighbors) )            # [{source,target,relation}]
  for (s,t,rel) in rels: add_bidirectional_edge(s, t, rel)

  # ---- Step 6: rebuild FAISS index once ----
  M.rebuild_index()
  return added
```

**Cost:** 3 LLM calls/session; non-deterministic; UPDATE overwrites; DELETE destructive.

### 11.2 Ingestion — FastASEM (`FastSessionIngestor.ingest_session`)

```
ALGORITHM FastASEM-INGEST(turns, session_date_str, session_id, M)
  (dt, iso) ← parse_session_datetime(session_date_str)   # "8 May 2023" → datetime + ISO

  # ---- Step 1: atomic-fact extraction (LLM call #1, the ONLY LLM call) ----
  facts ← LLM( SLAFI_prompt(session_date_str, join(turns)) )
  #   rules: resolve pronouns; resolve relative→absolute dates; atomic; entities
  if facts = ∅: facts ← fallback_per_line(turns, session_date)   # "(Date: ...)" stamped

  # ---- Step 2: build candidate notes (session date stamped) ----
  raw ← []
  for f in facts:
      e ← embed( f.fact + f.keywords + f.tags + f.entities )     # joint embedding
      z ← embed( f.fact )                                        # intent embedding
      raw.append( Note(f.fact, t=dt, K=f.keywords, G=f.tags, X=f.fact,
                       e, z, q=q0,
                       session_id, session_date=session_date_str,
                       timestamp_iso=iso, entities=f.entities,
                       speaker=f.speaker) )

  # ---- Step 3: deterministic gate (NO LLM) ----
  added ← []
  for n in raw:
      top3 ← ANN(n.z, k=3)
      if top3 = ∅: M.add(n); added.append(n); continue
      (top_ex, max_sim) ← argmax_{ex∈top3} cos(n.z, ex.z)

      if max_sim ≥ τ_redund:            # 0.90 → NOOP
          continue
      elif max_sim ≥ τ_novel:           # 0.75 → UPDATE (conditional)
          speakers ← {n.speaker, top_ex.speaker}
          overlap ← (n.entities ∩ top_ex.entities) − speakers
          if overlap ≠ ∅:
              M.add( append_merge(top_ex, n) )   # X grows, e recomputed, c/t preserved
              added.append(top_ex); continue
      M.add(n); added.append(n)            # default → ADD

  # ---- Step 4: deterministic graph weaving (NO LLM) ----
  for n in added:
      for m in M.search_by_entities(n.entities, k=5):
          if m ≠ n: add_edge(n, m, "same-entity")
      for nb in ANN(n.e, k=4):
          if nb ≠ n and cos(n.e, nb.e) ≥ 0.55:
              add_edge(n, nb, "temporal" if same_session else "semantic")
  return added
```

**Cost:** 1 LLM call/session; deterministic; UPDATE appends (no loss); no DELETE.

### 11.3 Retrieval — ASEMv2 (`EnhancedHybridRetriever.retrieve`)

```
ALGORITHM ASEMv2-RETRIEVE(query, M)
  e_q ← embed(query)

  # ---- Phase A: multi-channel RRF recall (inherited from HybridRetriever) ----
  pool ← ANN(e_q, k1)
  dense_rank ← rank(pool)
  if use_bm25:    bm25_rank ← rank( M.bm25_search(query, k1) )
  if use_entity:  ent_rank  ← rank( M.search_by_entities(capital_tokens(query), k1) )
  is_temp ← query matches /when|before|after|date|time|year|month|day|.../

  for note in union(pool, bm25, ent):
      s ← 0
      if note in dense_rank: s += w_dense / (rrf_k + dense_rank[note])
      if note in bm25_rank:  s += w_bm25  / (rrf_k + bm25_rank[note])
      if note in ent_rank:   s += w_ent   / (rrf_k + ent_rank[note])
      if is_temp and note has session_date: s += w_temp / rrf_k
      if cos(e_q, note.e) ≥ δ or note in bm25/ent: keep (note, s)
  candidates ← top-k1 by s

  # ---- Phase B: z-scored composite re-rank (inherited) ----
  lam ← adaptive_lambda(query)            # factual 0.25 / reasoning 0.60 / default 0.40
  sim_norm ← zscore([cos(e_q, n.e) for n in candidates])
  q_norm   ← zscore([n.q for n in candidates])
  result   ← top-k2 by (1−lam)·sim_norm + lam·q_norm

  # ---- Phase C: TRUE N-hop BFS expansion (ASEMv2-specific) ----
  frontier ← ids of top `link_traversal_topn` of result
  visited  ← ids(result); found ← []
  for hop in 1..max_hops:                 # max_hops = 2
      decay ← hop_decay ** hop           # 0.7 ** hop
      next_frontier ← ∅
      for note in M.get_notes(frontier):
          for (nb_id, rel) in note.L:
              if nb_id in visited: continue
              visited.add(nb_id); next_frontier.add(nb_id)
              q_eff ← nb.q · cos(e_q, nb.z)        # intent-grounded Q
              found.append( ( cos(e_q, nb.e) · (0.5+0.5·q_eff) · decay
                              · relation_weight[rel], nb ) )
      frontier ← next_frontier
      if frontier = ∅: break
  result += top-5 of found

  # ---- Phase D: GLOBAL-GRAPH re-rank (ASEMv2-specific) ----
  refresh_community_and_pagerank(M)       # Louvain + PageRank, cached by bank hash
  for note in result:
      local   ← cos(e_q, note.e)
      q_eff   ← note.q · cos(e_q, note.z)  # intent-grounded
      global  ← community_boost·[same community as top note] + pagerank[note]
      hybrid  ← α·norm(local) + β·global + γ·norm(q_eff)   # α.35 β.25 γ.40
  result ← re-sort by hybrid
  return result
```

**Cost:** RRF Phase A + N-hop BFS (2 hops, relation-weighted) + global-graph re-rank
(Louvain + PageRank). Strong on multi-hop/inference; no temporal signal beyond the shared
RRF boost.

### 11.4 Retrieval — FastASEM (`HybridRetriever.retrieve`)

```
ALGORITHM FastASEM-RETRIEVE(query, M)
  e_q ← embed(query)

  # ---- Phase A: multi-channel RRF recall (identical to ASEMv2 Phase A) ----
  pool ← ANN(e_q, k1)
  dense_rank ← rank(pool)
  if use_bm25:    bm25_rank ← rank( M.bm25_search(query, k1) )
  if use_entity:  ent_rank  ← rank( M.search_by_entities(capital_tokens(query), k1) )
  is_temp ← query matches /when|before|after|date|time|year|month|day|.../

  for note in union(pool, bm25, ent):
      s ← 0
      if note in dense_rank: s += w_dense / (rrf_k + dense_rank[note])
      if note in bm25_rank:  s += w_bm25  / (rrf_k + bm25_rank[note])
      if note in ent_rank:   s += w_ent   / (rrf_k + ent_rank[note])
      if is_temp and note has session_date: s += w_temp / rrf_k   # ← effective here
      if cos(e_q, note.e) ≥ δ or note in bm25/ent: keep (note, s)
  candidates ← top-k1 by s

  # ---- Phase B: z-scored composite re-rank (identical to ASEMv2 Phase B) ----
  lam ← adaptive_lambda(query)
  sim_norm ← zscore([cos(e_q, n.e) for n in candidates])
  q_norm   ← zscore([n.q for n in candidates])
  result   ← top-k2 by (1−lam)·sim_norm + lam·q_norm

  # ---- Phase C: 1-hop link traversal (simpler than ASEMv2's N-hop) ----
  for seed in result:
      for nb in M.get_notes([l.target_id for l in seed.L]):
          if nb not in result:
              score(nb) ← cos(e_q, nb.e) · (0.5 + 0.5·nb.q)
  result += top-3 by score
  return result
```

**Cost:** RRF Phase A + 1-hop traversal only. No N-hop BFS, no global-graph re-rank. The
temporal-boost term in Phase A is what makes "when" questions work, and it is effective
*because* FastASEM's notes carry absolute dates (§3.2).

### 11.5 Side-by-side summary

| | ASEMv2 | FastASEM |
|---|---|---|
| Ingest LLM calls/session | 3 (extract, ops, links) | 1 (extract only) |
| Note timestamp | `NOW_UTC` (wall-clock) | parsed session date + ISO |
| Memory op decision | LLM (20-note cap) | deterministic cosine gate |
| UPDATE semantics | overwrite (loses original) | append (preserves original) |
| DELETE | LLM-judged, destructive | none |
| Link generation | LLM free-form | deterministic entity + adjacency |
| Retrieval Phase A | RRF (dense+BM25+entity+temporal) | RRF (dense+BM25+entity+temporal) |
| Retrieval Phase C | N-hop BFS (2 hops, relation-weighted) | 1-hop traversal |
| Retrieval Phase D | Louvain community + PageRank re-rank | none |
| Answer mode | distillation (select + rewrite) | direct (date-prefixed context) |

---

## 12. Where to look (file map)

- `asem/fast_ingest.py` — `FastSessionIngestor`, `parse_session_datetime`,
  `_EXTRACTION_SYSTEM_PROMPT`, `_weave_graph_links`, deterministic gate.
- `asem/batch_ingestion.py` — `BatchIngestor` (ASEMv2 path, for reference).
- `asem/note.py` (lines 118–160) — new note fields + (de)serialization.
- `asem/retriever.py` — `HybridRetriever` (RRF / BM25 / entity / temporal).
- `asem/enhanced_retriever.py` — `EnhancedHybridRetriever` (ASEMv2 path).
- `asem/answer_agent.py` — `direct_mode` / `distil_and_answer` / `_generate_resilient`.
- `asem/write_gate.py` — `WriteGate`.
- `eval/systems.py` — `ASEMSystemV2` (line 168), `FastASEMSystem` (line 214),
  `build_asem_v2_system` (line 405), `build_fast_asem_system` (line 608).
- `data/prompts/P_temporal_qa.txt` — temporal QA prompt.
- `configs/presets/sota_benchmark.yaml` — FastASEM preset.
