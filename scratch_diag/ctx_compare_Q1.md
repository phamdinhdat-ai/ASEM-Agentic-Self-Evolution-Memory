# Context Comparison — Q: "When did Caroline go to the LGBTQ support group?"

**Gold:** `7 May 2023` · **FastASEM pred:** `7 May 2023` ✓ · **FullContext pred:** `Yesterday` ✗
(Same question, same conversation conv-26 — but radically different contexts.)

---

## FastASEM — what the answer agent sees (11 retrieved notes)

Sourced from `logs/dump_ctx.log` (`dump_fastasem_context.py --only 1,2,3`). Compact, dated,
resolved-absolute facts, chronologically sorted, ~11 notes / ~1.2 KB:

```
- [1:56 pm on 8 May, 2023] On 7 May 2023, Caroline went to an LGBTQ support group.   <-- GOLD, date already resolved
- [7:55 pm on 9 June, 2023] On 2 June 2023, Caroline encouraged students to get involved in the LGBTQ community.
- [1:36 pm on 3 July, 2023] Caroline felt that the LGBTQ+ community had grown significantly.
- [4:33 pm on 12 July, 2023] On 10 July 2023, Caroline went to an LGBTQ conference. (Description: … met people … felt accepted …)
- [1:51 pm on 15 July, 2023] Caroline felt comforted by knowing that Caroline was not alone …
- [1:51 pm on 15 July, 2023] Caroline feels supported by people who embrace and back Caroline up.
- [8:56 pm on 20 July, 2023] The name of Caroline's group is Connected LGBTQ Activists.
- [8:56 pm on 20 July, 2023] On 18 July 2023, Caroline joined a new LGBTQ activist group. (Description: …)
- [2:24 pm on 14 August, 2023] Caroline found the pride parade inspiring and it pushed her to keep fighting for LGBTQ rights.
- [3:31 pm on 23 August, 2023] Caroline supports LGBTQ rights.
- [3:19 pm on 28 August, 2023] Caroline talked with LGBTQ+ young people at the youth center.
```

**Why it answers correctly:** the first note is *already temporal-grounded* — ingestion turned the raw
"yesterday" (from the session dated 8 May 2023) into **"On 7 May 2023"**, stamped with its session
timestamp. The QA prompt's rule 3 ("for time/date questions give the exact date") makes the model
surface it.

---

## FullContext — what its answer agent sees (whole transcript)

Sourced from `extract_sessions_from_conv` + `_FULL_CONTEXT_PROMPT` (max_history_turns=0). The SAME
question receives **all 419 turns (~70 KB / ~17k tokens)**, of which only ~2 turns matter:

```
Use the conversation excerpts below to answer the question. Reply with only the answer — a few words or one sentence, no explanation.

Conversation:
[Caroline] Hey Mel! Good to see you! How have you been?
[Melanie] Hey Caroline! Good to see you! I'm swamped with the kids & work. …
[Caroline] I went to a LGBTQ support group yesterday and it was so powerful.   <-- the ONLY evidence
[Melanie] Wow, that's cool, Caroline! What happened that was so awesome? …
[Caroline] The transgender stories were so inspiring! …
… [Melanie] I painted that lake sunrise last year! …
… [419 turns total — the whole 9-month conversation] …

Question: When did Caroline go to the LGBTQ support group?

Answer:
```

**Why it answers wrong:** the evidence turn literally says *"…yesterday"*, and the model echoes it.
There is no absolute date anywhere in the transcript (the session timestamp `8 May 2023` lives only in
the dataset metadata, which FullContext never injects), so FullContext cannot resolve "yesterday" → "7 May 2023".

---

## Side-by-side

| Aspect | FastASEM | FullContext |
|---|---|---|
| Context size | **11 notes ≈ 1.2 KB** | **419 turns ≈ 70 KB / ~17k tokens** |
| Evidence for this Q | 1 pre-resolved absolute-date note | 1 raw "yesterday" turn buried in 419 |
| Temporal grounding | done at ingestion (note says 7 May 2023) | none — echoes raw "yesterday" |
| Structure | dated, entity/keyword/description-annotated, chronological | raw conversational turns, undated |
| Cost per Q | tiny + local retrieval | huge transcript on every call |
| Answer | **7 May 2023 ✓** | Yesterday ✗ |

**Takeaway:** the same information ("went to the group yesterday") is present in both — but FastASEM
pre-converts it to an absolute date at ingestion and puts it first in a tiny context, so it *answers*;
FullContext carries the entire conversation yet the decisive temporal fact is only expressible in the
relative form the model echoes back.
