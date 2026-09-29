# ASEM-THG vs. ASEM (v1) vs. FastASEM — Prompt & Ingestion Analysis

Scope: the three ingestion architectures registered in `eval/systems.py`, compared at the
level of (a) the extraction prompt, (b) the prompt-driven pipeline stages, and (c) the
deterministic maintenance that produces the stored bank.

> **Status.** The defects found in this analysis were subsequently fixed (see
> §5 and §6). Where a section describes the *original* behaviour it is marked
> "(was)"; the corrected behaviour is described alongside it. The remaining open
> item is the PPR wiring in §3.2.

| System | Builder | Ingest code | Prompt(s) |
|---|---|---|---|
| **ASEM (v1)** | `build_asem_system` | `asem/pipeline.py::write_batch` → `NoteConstructor.build_batch` + `MemoryManager` + `LinkEvolver` | `P1_batch_note_construction.txt`, `P5_batch_memory_ops.txt`, `P6_batch_link_generation.txt`, `P3_batch_evolution.txt` |
| **FastASEM** | `build_fastasem_system` | `asem/fast_ingest.py::FastSessionIngestor` | inline `_EXTRACTION_SYSTEM_PROMPT` + `_EXTRACTION_USER_PROMPT` |
| **ASEM-THG** | `build_asem_thg_system` | `asem/single_pass_ingest.py::SinglePassSessionIngestor` | inline `_SINGLE_PASS_PROMPT_TEMPLATE` |

---

## 1. Prompt-level comparison

### 1.1 Requested output schema

| Field | ASEM v1 (`P1_batch`) | FastASEM | ASEM-THG |
|---|---|---|---|
| raw content `c` | the turn text | `fact` | `fact` |
| `K` keywords | ✅ `keywords` (3–8) | ✅ `keywords` | ✅ `keywords` |
| `G` tags | ✅ `tags` (2–5 from an allowed list) | ✅ `tags` | ✅ `tags` (was: hardcoded `["atomic_fact"]`) |
| `X` description | ✅ `description` (≤45 words, rewritten by S3) | `= fact` | `= fact` |
| `entities` | ✅ | ✅ | ✅ |
| `speaker` | ✅ | ✅ | ✅ |
| **triplet `(subject, predicate, object)`** | ❌ | ❌ | ✅ **unique** |

ASEM-THG is the only prompt that asks for a `(subject, predicate, object)` triple.
That is the enabling signal for deterministic temporal versioning
(`TemporalHyperGraph._subj_pred_index` → `superseded_by`), which neither of the others has.

It used to be the only prompt that **dropped tags** (`G=["atomic_fact"]` for every note,
which also kept them out of the joint embedding `e = concat(c, K, G, X)`). Tags are now
extracted and fed into `e` like the other two systems.

### 1.2 Granularity of extraction

| | Rule |
|---|---|
| ASEM v1 | "Output **EXACTLY ONE object per turn**, in the same order as the turns." → 1 note per dialogue turn, greetings included. |
| FastASEM | "Extract **EVERY** distinct concrete event, activity, plan, intention, preference, relationship, and status as its **OWN** separate note. **Do NOT merge** … Do NOT drop a fact just because it sounds minor." + explicit past-event vs. future-plan example. |
| ASEM-THG | "extract clear, standalone, **atomic** factual notes" — atomicity is stated but never operationalised. |

ASEM-THG's rule was the weakest of the three: it had neither the "one per turn" contract that
makes ASEM v1's output size predictable, nor FastASEM's explicit enumeration of what counts as
a separate note. Output size was effectively delegated to the model's own judgement.

**(now)** the prompt carries an explicit "ONE FACT PER NOTE" rule with the same
past-event/future-plan disambiguation FastASEM uses, so it no longer depends on the model's
implicit judgement.

### 1.3 Relative-time / pronoun grounding

| | Guidance |
|---|---|
| ASEM v1 | 1 rule + 1 worked example (`"last month" → April 2023`); `{session_date}` injected as `%d %B %Y`. |
| FastASEM | 6 explicit mappings (`yesterday`, `last week`, `next month`, `last year`, …) + "**Always** state the resolved date explicitly in the fact". |
| ASEM-THG | 5 explicit mappings (`yesterday`, `last week`, `last Saturday`, `next week/month`, `last year`) + "state the resolved absolute date inside the fact". (was: 1 rule line, no example, no enumeration) |

(The `{session_date}` placeholder in ASEM-THG was also missing from the template until the
recent fix — see §5.1.)

### 1.4 Prompt scaffolding

| | Task | Rules | Reasoning steps | Few-shot examples | Common mistakes | Length cap |
|---|---|---|---|---|---|---|
| ASEM v1 | ✅ | ✅ | ✅ | ✅ (2) | ✅ | ✅ 45 words |
| FastASEM | ✅ | ✅ (5, dense) | ❌ | ❌ | ❌ | ✅ 45 words |
| ASEM-THG | ✅ | ✅ (4) | ❌ | ❌ | ❌ | ✅ 45 words |

**(was)** ASEM-THG had the thinnest prompt of the three: no examples, no reasoning steps, no
counter-examples and no output-length bound. It is now closest to FastASEM's shape — dense
rules (including the 6 relative-date mappings FastASEM enumerates) plus a hard length cap —
while keeping the triplet field.

---

## 2. LLM call cost per session

Let `N` = turns in a session, `M` = facts extracted.

| Stage | ASEM v1 | FastASEM | ASEM-THG |
|---|---|---|---|
| S1 extraction | 1 (batched over turns) | 1 | 1 |
| S2 write decision | `0…M` (`WriteGate` short-circuits clear ADD/NOOP; ambiguous → LLM) | 0 (deterministic gate) | 0 (deterministic gate) |
| S3 link generation | `M × 1` (per written note) | 0 | 0 |
| S3 memory evolution | up to `M × 1` (batched, sparse gate) | 0 | 0 |
| post-pass | `cross_chunk_link_evolve` over unlinked notes | — | — |
| **Total** | **≈ 1 + 2M … 1 + 3M** | **1** | **1** |

FastASEM and ASEM-THG are **equal** at 1 LLM call per session. ASEM-THG's saving over ASEM v1
is real and large (the "75%+" figure holds because `M` per session is ~20–30).

---

## 3. Deterministic graph maintenance

| | Link generation | Relation labels | Versioning | Evolution |
|---|---|---|---|---|
| ASEM v1 | LLM, pairwise new↔neighbours | 6 LLM labels: `contradicts, extends, causal, same-topic, temporal, semantic` | ❌ | ✅ LLM rewrites neighbour `X` |
| FastASEM | deterministic | `same-entity`, `temporal`, `semantic` | ❌ | ❌ |
| ASEM-THG | deterministic | `semantic`, `superseded_by` | ✅ `superseded_by` (new→old) | ❌ |

### 3.1 Which edges actually reach the bank

This is the most important structural difference, and it is easy to miss.

**(was)** `TemporalHyperGraph.add_note()` added **four** edge types but returned only **two**
as `LinkRecord`s, so only those two were written back to `note.L`:

| Edge type | Added to nx graph | Returned → persisted in `note.L` (was) |
|---|---|---|
| entity co-occurrence | ✅ | ❌ (graph-only) |
| temporal sequence (`add_temporal_sequence`) | ✅ | ❌ (graph-only) |
| semantic similarity (cosine ≥ 0.70) | ✅ | ✅ |
| versioning `superseded_by` | ✅ | ✅ |

FastASEM persists `same-entity` + `temporal` + `semantic`, so the stored ASEM-THG bank had
*fewer* usable edges than FastASEM's despite the richer in-memory model.

**(now)** all four edge types are emitted as `LinkRecord`s and persisted:

| Edge type | nx graph | Persisted in `note.L` | Relation label |
|---|---|---|---|
| shared-entity peers | ✅ (via entity nodes) | ✅ | `same-entity` |
| temporal sequence | ✅ | ✅ | `temporal` |
| semantic similarity | ✅ | ✅ | `semantic` |
| versioning | ✅ | ✅ | `superseded_by` |

Two further fixes came with it:

* **Symmetric edges.** The retriever only follows *outgoing* links from a seed
  (`HybridRetriever._traverse_links`), so a one-directional edge is invisible whenever the
  seed is the older endpoint. Every edge is now written on both endpoints. Peers that
  pre-date the ingest call exist only in the bank, so the ingestor re-persists them
  (`SinglePassSessionIngestor._persist` + `TemporalHyperGraph.pop_touched_peers`).
* **Fan-out caps.** A dominant entity (the conversation's own speaker name) is shared by most
  notes; an uncapped entity channel would link every note to every other note. Peers are
  ranked by embedding similarity and cut at `max_entity_links` / `max_semantic_links`.
* `temporal` and `same-entity` deliberately reuse FastASEM's labels, so the shared retriever
  weights (`_RELATION_BOOST`, `relation_weights`) apply to both systems.

### 3.2 The hyper-graph itself is never used at retrieval

`TemporalHyperGraph.personalized_pagerank()` is implemented and unit-tested
(`tests/test_hyper_graph.py`) but **nothing calls it**. Grep shows `hyper_graph` /
`TemporalHyperGraph` appear only in `asem/hyper_graph.py`, `asem/single_pass_ingest.py`, and
the test.

- `build_asem_thg_system` constructs `SinglePassSessionIngestor(backend=…, q0=…, max_retries=…)`
  with **no `hyper_graph=`**, so the ingestor owns a private graph.
- `EnhancedHybridRetriever` (the THG retriever) has **no hypergraph parameter**. It rebuilds
  its traversal graph from `note.L` (`_refresh_graph`).

Consequence: the entity-node graph and the PPR multi-hop retrieval described in the design
proposal **do not run** in the static-eval workflow. This is structural, not incidental —
Phase A (ingest) and Phase B (eval) are separate processes, so any in-memory graph is
discarded at the end of ingest regardless.

**Status: still open**, but mitigated. Because the entity channel is now projected onto
`note.L` as `same-entity` peer edges (§3.1), the *retriever's existing* link traversal
reaches the same neighbours it would have reached through an entity node. PPR would add a
graded, multi-hop weighting on top; wiring it needs the retriever to rebuild
`TemporalHyperGraph.from_bank(M)` at query time.

### 3.3 Relation-label coverage in the retriever

`EnhancedHybridRetriever.relation_weights` has no entry for `superseded_by`, so it falls back
to the default `0.5` — the same weight as an unknown/legacy label. `temporal` (0.7) and the
strong relations (1.1–1.2) are unreachable for ASEM-THG because those labels are never stored.

---

## 4. Metadata written onto each note

| | `session_id` | `session_date` | `timestamp_iso` | `entities` | `speaker` |
|---|---|---|---|---|---|
| ASEM v1 | `session_N` / label | raw header | ✅ | ✅ | ✅ |
| FastASEM | `s{N}` | raw | ✅ | ✅ | ✅ |
| ASEM-THG | `s{N}` | raw | ✅ | ✅ | ✅ |

All three are temporally grounded identically (shared `asem/temporal.py`).

---

## 5. Concrete defects found in ASEM-THG

### 5.1 Extraction prompt never contained the dialogue (FIXED)

`_SINGLE_PASS_PROMPT_TEMPLATE` only had `{session_date}`; the
`.format(session_date=…, dialogue=dialogue_text)` call silently discarded the extra kwarg.
The model saw no conversation, returned `[]`, and every conversation recorded
`produced 0 notes — recorded as an error`. Fixed by adding a `DIALOGUE:\n{dialogue}` block, plus
a fallback when the parse is empty. Regression tests added.

### 5.2 `null` list fields crashed the ingestor (FIXED)

`asem/single_pass_ingest.py` used to read:

```python
entities = [str(e).strip() for e in item.get("entities", []) if str(e).strip()]
```

`item.get("entities", [])` returns `None` when the key exists with a JSON `null` value, and
iterating `None` raises `TypeError`. The safe form — `(item.get("entities") or [])` — is now
used for `entities`, `keywords`, `tags`, `speaker`, `subject`, `predicate` and `object`.
Regression test: `test_null_list_fields_tags_and_caps_are_handled`.

### 5.3 No write-path caps (FIXED)

`cap_description` (600 chars) and `cap_keywords` (≤12) are now applied on the ASEM-THG write
path, matching ASEM v1, FastASEM and `batch_ingestion`.

### 5.4 Tags dropped (FIXED)

Tags are extracted by the prompt and included in the joint embedding `e`; `_DEFAULT_TAG`
(`"fact"`) is the fallback when the extractor returns none.

### 5.5 Weaker fallback (FIXED)

`_fallback_extract` now strips the `[Speaker]` prefix off the fact text (storing the speaker
in its own field) and skips `[Session …]` header lines, matching FastASEM's fallback shape.

### 5.6 Stale graph on reset (FIXED)

`ASEMTHGSystem.reset()` cleared the bank but kept the hyper-graph's `_fact_nodes`, so
`superseded_by` versioning would have compared new notes against a cleared bank.
`TemporalHyperGraph.clear()` is now called too.

---

## 6. Summary judgement

| Dimension | ASEM v1 | FastASEM | ASEM-THG |
|---|---|---|---|
| Prompt richness | ★★★★ | ★★★ | ★★★ (was ★★) |
| Extraction schema | descriptive triple | `fact` + tags | **+ `(s,p,o)` triplet** |
| Tags captured | ✅ | ✅ | ✅ (was ❌) |
| LLM calls / session | 1 + 2M…3M | **1** | **1** |
| Deterministic linking | ❌ (LLM) | ✅ | ✅ |
| Temporal versioning | ❌ | ❌ | ✅ |
| LLM-free evolution | ❌ (LLM rewrite) | ✅ | ✅ |
| Edges persisted | 6 labels | 3 labels | ✅ 4 labels, symmetric (was 2) |
| Graph used at retrieval | ✅ (`_traverse_links`) | ✅ (`_refresh_graph`) | ✅ link traversal (PPR still unwired) |
| Robustness | mature | mature | ✅ null-safe, capped, dim-checked |

**Bottom line.** ASEM-THG's *architecture* (single-pass extraction + deterministic hyper-graph
+ `superseded_by` versioning + no LLM rewriting) is the strongest of the three and is the right
direction for cost and for adversarial robustness. The implementation gaps found in this
analysis have been closed: the prompt is now FastASEM-dense and capped, tags are captured, all
four edge types are persisted symmetrically with fan-out caps, and the two failure modes the
codebase had already fixed (`null` list fields, missing write-path caps) no longer apply.

**Applied fixes**

1. ✅ Defensive `or []` on every list-valued LLM field (§5.2).
2. ✅ Persist all four relation types into `note.L`, symmetrically, with fan-out caps (§3.1).
3. ✅ Extract `tags` and include them in the joint embedding `e` (§5.4).
4. ✅ Apply `cap_description` / `cap_keywords` on the write path (§5.3).
5. ✅ Bring the extraction prompt up to FastASEM's rule density — explicit "one fact per note",
   the relative-date enumeration, and a 45-word cap (§1.2–1.4).
6. ✅ Reset the hyper-graph with the bank; register THG's relation labels on the retriever (§5.6).

**Still open**

7. Wire the entity-node graph into retrieval so `personalized_pagerank` actually runs (§3.2).
   Needs `TemporalHyperGraph.from_bank(M)` plus an opt-in PPR stage in `EnhancedHybridRetriever`.
   Mitigated for now: the `same-entity` peer edges persist, so the retriever's existing link
   traversal reaches the same neighbours through the note-to-note projection.

**Verification.** `tests/test_hyper_graph.py`, `tests/test_single_pass_ingest.py`,
`tests/test_asem_thg_system.py`, `tests/test_enhanced_retriever.py`, `tests/test_pipeline.py`
all green; full suite 169 passed (the 3 `tests/test_backends.py` HF failures are a pre-existing
missing-`accelerate` environment issue, unrelated to these changes).
