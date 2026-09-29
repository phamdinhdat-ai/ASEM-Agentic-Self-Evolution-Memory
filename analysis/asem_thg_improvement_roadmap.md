# ASEM-THG Improvement Roadmap

**Scope:** concrete, code-level improvements to ASEM-THG, organized by LoCoMo category,
grounded in the retrieval tests (`scratch_diag/retriever_comparison.md`,
`scratch_diag/asem_thg_retrieve.md`), the ingestion demo (`scratch_diag/asem_thg_demo.md`),
and the prompt/graph analysis (`analysis/asem_thg_vs_asem_vs_fastasem_prompts.md`).

---

## 1. Where ASEM-THG stands today

| Category | Status | Evidence |
|---|---|---|
| Temporal | ✅ strong | retrieval rank 1 (7 May 2023); absolute dates; `superseded_by` versioning exists |
| Single-hop | 🟡 mixed | bare terms rank 1; but generic note out-ranks the specific one |
| Multi-hop / attribution | 🔴 weak | "Melanie an ally?" → no Melanie-support premise retrieved |
| Conversational / recency | 🔴 weak | returns the later trip plan; gold is the earlier adoption plan |
| Adversarial | ⚪ unmeasured | masked-query channel + `same-entity` peers exist but untested |

Three structural reasons, all verifiable in code:

1. **The hyper-graph is never consulted at retrieval.** `personalized_pagerank()` is called
   nowhere outside `tests/test_hyper_graph.py`; `from_bank` does not exist. The entity-node
   graph — THG's defining feature — is dead weight at query time.
2. **THG's unique edges are mis-weighted.** `relation_weights` (`enhanced_retriever.py`) and
   `_RELATION_BOOST` (`retriever.py`) have **no `superseded_by`** and **no `same-entity`**, so
   those edges fall back to 0.5/1.0 and cannot influence ranking.
3. **THG answers in distillation mode.** `AnswerAgent.direct_mode` defaults to `False` and
   `build_asem_thg_system` never sets it, so THG paraphrases through an extra LLM call instead
   of using FastASEM's direct, date-prefixed QA context.

---

## 2. P0 — small, high-impact (do first)

### P0-1. Direct, date-prefixed answering for THG

Mirror `build_fast_asem_system` in `build_asem_thg_system`:

```python
qa_prompt_path = os.path.join("data", "prompts", "P_temporal_qa.txt")
qa_prompt = _load_text(qa_prompt_path) if os.path.exists(qa_prompt_path) else _RETRIEVAL_PROMPT
...
answer_agent = AnswerAgent(
    backend=backend,
    prompt_template=distil_prompt,
    baseline_prompt_template=qa_prompt,          # was _RETRIEVAL_PROMPT
    direct_mode=bool(ans_cfg.get("direct_mode", True)),   # was: default False
    max_retries=max_retries,
    max_tokens=int(ans_cfg.get("max_tokens") or 0) or None,
    context_window=int(ans_cfg.get("context_window") or 0) or None,
    max_context_notes=int(ans_cfg.get("max_context_notes") or 0) or None,
    content_char_limit=int(ans_cfg.get("content_char_limit", 200)),
)
```

*Why:* `direct_answer()` builds a chronology-sorted, `[<session date>]`-prefixed context, which
is exactly what produced FastASEM's temporal/EM win, and it removes one LLM call per question.
*Targets:* Temporal, Single-hop, EM format.
*Verify:* re-run `run_static_eval` with `--fresh`; compare per-category EM/ROUGE-L.

### P0-2. Register THG's edge types in both weight tables

```python
# asem/enhanced_retriever.py  -> relation_weights
"superseded_by": 1.3,   # versioning is strong evidence (a newer fact replaces an older one)
"same-entity":   0.9,   # corroboration, weaker than a typed relation

# asem/retriever.py -> _RELATION_BOOST  (same two entries)
```

*Why:* versioning is THG's unique signal and currently scores as *unknown*; `temporal` (0.7)
never applies to THG edges either.
*Targets:* Temporal, multi-hop.
*Verify:* the retrieval comparison's `traversed_relations` stats should now show
`superseded_by` and `same-entity`.

---

## 3. P1 — architectural (the multi-hop lever)

### P1-1. Entity-node hygiene at weave time

`TemporalHyperGraph._entity_keys` unions **`entities` and `keywords`**:

```python
keys = {norm(e) for e in note.entities} | {norm(k) for k in note.K}
```

`K` contains verbs/adjectives (`accepted`, `bonding`, `cherish`, `congratulations`), so the
entity nodes are polluted and `same-entity` edges form a near-clique (the demo showed 684
same-entity edges on 141 notes).

**Fix:** index **named entities only** (`note.entities`), or keep `K` under a separate node
type with a lower edge weight. *Targets:* multi-hop, adversarial (cleaner entity traversal).
*Expected:* fewer, more meaningful `same-entity` edges; less clique collapse.

### P1-2. Wire Personalized PageRank (PPR) into retrieval

This is the single biggest multi-hop improvement and completes THG's design.

**Design** (`asem/hyper_graph.py` + `asem/enhanced_retriever.py`):

```python
# 1) Rebuild the bipartite fact<->entity graph from the persisted bank (no ingest state).
def build_entity_graph(M):
    g = nx.Graph()
    for n in M.list_notes():
        g.add_node(n.id, kind="fact")
        for e in {norm(x) for x in (n.entities or [])}:
            g.add_node(f"ent:{e}", kind="entity")
            g.add_edge(n.id, f"ent:{e}", weight=1.0)
        for l in n.L:                          # note-to-note typed edges
            g.add_edge(n.id, l.target_id, weight=relation_weights.get(l.relation, 0.5))
    return g

# 2) Opt-in PPR stage in EnhancedHybridRetriever.retrieve(), seeded by the Phase-B top notes.
if self.enable_ppr and self._ppr_scores is None:
    g = build_entity_graph(M)                          # cached by bank hash
    seeds = [n.id for n in base_results[: self.ppr_seed_topn]]
    self._ppr_scores = nx.pagerank(g, alpha=0.85, personalization=seeds)

# 3) Blend as a 4th term
hybrid = alpha*norm(local) + beta*global_ + gamma*norm(q_eff) + delta_ppr*norm(ppr[note.id])
```

*Why:* the entity-node graph is rebuilt at query time from `note.entities` (persisted) + `note.L`
(persisted), so Phase A / Phase B separation is respected. PPR spreads relevance through shared
entities, which is exactly the "Melanie + transgender + support" multi-hop bridge that plain
1-hop traversal misses.
*Cost:* O(N·|entities|) to build + PPR; cache by bank hash (the retriever already caches
Louvain/PageRank this way). Cheap at N≈337.
*Config:* `enable_ppr` (default False until measured), `ppr_seed_topn`, `delta_ppr`.
*Verify:* the multi-hop query "Would Melanie be an ally?" should now retrieve a Melanie-support
note into the top-k.

### P1-3. Object-aware versioning key

`TemporalHyperGraph.add_note` versions on `(subject, predicate)` only, so unrelated facts
supersede each other (demo: 52 spurious `superseded_by` edges; e.g. *"Caroline thinks painting is
a great outlet"* ← *"Caroline thinks the colors blend"*).

**Fix (pick one):**
- key on `(subject, predicate, object_key)` where `object_key = norm(object)` bucketed by
  embedding similarity — versioning fires only when the object is the *same slot*; or
- keep `(subject, predicate)` but require the predicate to be **single-valued**
  (`has been married for`, `lives in`, `is`, `status`) and the objects to be non-disjoint.

*Why:* it makes `superseded_by` a trustworthy signal (and P0-2 gives it weight 1.3), instead of
noise that promotes arbitrary notes.
*Targets:* Temporal (correct versioning), Conversational (recency).

---

## 4. P2 — answer quality

### P2-1. Multi-note aggregation for list / inference queries

Single-hop list questions ("What do Melanie's kids like?" → nature **and** dinosaurs) and
multi-hop inference ("Would she be an ally?") need the answer to combine notes, not read one.
For query types `list` / `inference`, pass the **top-k notes** and instruct union/reasoning
before allowing "I don't know".
*Targets:* Multi-hop (currently ~3/13 judge), Single-hop sets.

### P2-2. Tighten extraction

The demo produced 32–63 notes/session (337 for conv-26) including conversational filler
(*"Melanie believed Caroline had guts"*) and photo captions (*"shared a photo of a man and a
little girl in front of a waterfall"*). Add to `_SINGLE_PASS_PROMPT_TEMPLATE`:
- **DROP** greetings, acknowledgements, compliments, and pure photo descriptions (unless the
  photo is the *only* source of a fact).
- Prefer facts with a named entity **and** a predicate.
*Why:* smaller bank → less retrieval noise, cheaper context, higher precision.
*Risk:* over-filtering loses facts — measure per-category EM both ways.

---

## 5. Measurement plan

The static workflow is ingest-once, evaluate-many, and **resumes from cached predictions**.
To get a clean before/after, back up the baseline before re-running.

```bash
conda activate memory-r1

# Baseline (bank: static/memory_banks/locomo10/ds_thg, all 199 conv-26 QA)
python scripts/run_static_eval.py --tag ds_thg --config configs/models/deepseek_openai.yaml \
    --systems ASEM-THG --metrics em rougeL bertscore_f1 judge --conversations locomo_0000

# apply P0-1 + P0-2, then re-answer from scratch:
python scripts/run_static_eval.py --tag ds_thg --config configs/models/deepseek_openai.yaml \
    --systems ASEM-THG --metrics em rougeL bertscore_f1 judge --conversations locomo_0000 --fresh
```

Results: `data/benchmarks/results/static/locomo10/ds_thg__deepseek_v4_flash.{json,md}` —
the JSON carries the per-category breakdown.

**Tracking table** (fill as runs land):

| Category (n) | Baseline EM | After P0 EM | Δ | Baseline ROUGE-L | After ROUGE-L | Δ |
|---|--:|--:|--:|--:|--:|--:|
| Temporal | | | | | | |
| Single-hop | | | | | | |
| Multi-hop | | | | | | |
| Open-domain / adversarial | | | | | | |

> Note: the static eval scores **all 199** conv-26 QA pairs, whereas the fair-play comparison
> used FastASEM's canonical **117**-question subset — so don't mix the two tables.

---

## 6. Priority order

1. **P0-1 direct mode** (1 edit) — likely the largest EM/temporal move.
2. **P0-2 relation weights** (1 edit) — frees THG's own signal.
3. **P1-2 PPR** — the multi-hop fix (medium effort).
4. **P1-1 entity hygiene** + **P1-3 object-aware versioning** — precision of the graph.
5. **P2-1 aggregation** / **P2-2 extraction tightening** — quality polish.
