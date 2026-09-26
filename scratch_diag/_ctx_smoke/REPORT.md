# Context samples — context thật mà từng phương pháp gửi cho model

- Tag bank: `ds_fixed` · model: `qwen_qwen3_4b_instruct_2507` · config: `configs/models/qwen3_4b_api.yaml`
- Lấy mẫu: **1 ca sai + 0 ca đúng** cho mỗi (hệ thống × loại câu hỏi), cộng các idx bắt buộc []. seed=0.
- Tổng: **2 ca** — {'ASEM': 1, 'FullContext': 1}
- Chẩn đoán: {'GENERATION': 1, 'FULL_CONTEXT': 1}

> Context được **replay bằng probe backend** (không gọi LLM): prompt ghi lại chính là prompt
> first-pass đã tạo ra `preds/*.jsonl`. Prompt đầy đủ nằm trong `contexts/<system>__<conv>__<idx>.txt`,
> metadata đầy đủ trong `index.jsonl`.

Chẩn đoán tự động: **INGESTION** = bank không có note mang gold · **RETRIEVAL** = có note nhưng
không vào context · **GENERATION** = note mang gold đã ở trong context mà vẫn trả lời sai ·
**FULL_CONTEXT** = baseline ngữ cảnh đầy đủ (kèm cờ `evidence_in_prompt`) · **NO_RETRIEVAL** = NoMemory.

## Tổng hợp

| System | Loại | Sai | Đúng | Bucket (sai) |
|---|---|---|---|---|
| ASEM | single_hop | 1 | 0 | GENERATION:1 |
| FullContext | single_hop | 1 | 0 | FULL_CONTEXT:1 |

---

# ASEM

## single_hop

### [✗ SAI] `#1362` What pets does Jolene have?

- gold: `snakes`
- pred (real run): `Jolene has a snake named Seraphim and cats, including Max.`
- judge=False error=contradiction f1=0.0000 em=0.0 · diagnosis=**GENERATION**
- context: 3339 token (13356 ký tự) · note ids: `['b6c85d65-282f-4596-9936-0dab3cbd36bd', '94250268-7f8b-450d-a41b-e948f4ad3173', '8c1c75ee-4894-4727-93c5-5c2e9acbbe06', 'a556a2c2-ec30-4bff-be00-fc6919af294e', 'b33d2bfe-105e-4686-9148-23d966ac1ef9', 'c008d2e9-8e60-4313-b4a2-e9dc0deb4e71', '09d0dbb5-1ef0-4201-9fa6-a824fb87e6d3', 'fe41832e-8469-490c-a70e-d9bfe1fe6361']` · gold-bearing ids: `['09d0dbb5-1ef0-4201-9fa6-a824fb87e6d3', '1c263cf5-3641-454c-bddf-7341ebf4e967', '715cc18f-2158-4896-9b62-00d6e6d155b3', 'e3963b85-cc5f-4c5e-ac94-4de36d3a4086', 'f83f1438-b86d-4ac2-b2e6-21b4b1728f3d']`
- prompt đầy đủ: `contexts/ASEM__locomo_0007__1362.txt`
- judge nói: The gold answer specifies that Jolene has only snakes, but the AI answer includes cats as well, which contradicts the gold answer by introducing additional pets not mentioned in the expected response.

```text
You are a memory retrieval agent. Given a question and a set of candidate memory notes from a memory graph, select the notes needed to answer and produce a concise factual answer.

Each note carries: id, speaker (who said it), entities (people/places named), keywords, description, session_date, and relations — typed edges to OTHER NOTES IN THIS SAME LIST, referenced by target_id. A note may also carry `also_linked`: a count of its remaining edges to notes that are NOT in this list (e.g. "same-topic:14, extends:3"). Those targets cannot be read, but the counts tell you whether a note is heavily corroborated, contradicted or extended elsewhere in the memory.

## Task
Output a JSON object with two fields:
- **selected_ids**: The ids of the candidate notes whose information is actually used in the answer. Empty list if no note is relevant.
- **answer**: The concise answer (a few words to one sentence). Answer from the notes even if they only PARTIALLY cover the question — say "I don't know" ONLY when no note contains any information related to the question. For time or date questions, give the exact date, month, or year using the notes' session_date and descriptions, resolving any relative timeframe into an absolute date.

## Decision procedure (think through these BEFORE outputting the JSON)
Step 1: Read the question and identify what fact it asks for.
Step 2: Skim each candidate note's description, keywords, entities, speaker, and session_date.
Step 3: Keep every note that contributes information related to the question. Follow `relations` (especially temporal / causal / cont

…[11157 chars omitted — full prompt in the .txt file]…

cats look like they are having a blast and asks how often she takes them out; Deborah takes her cats for a run in the park every morning and evening as part of her exercise and nature routine.", "session_date": "2023-07-09T19:37:00Z", "entities": ["Jolene", "Deborah"], "speaker": "Jolene", "relations": [{"relation": "same-topic", "target_id": "8c1c75ee-4894-4727-93c5-5c2e9acbbe06"}, {"relation": "same-topic", "target_id": "a556a2c2-ec30-4bff-be00-fc6919af294e"}], "also_linked": "semantic:2, extends:1", "content": "[Jolene] Looks like they're having a blast! How often do you take them out?"}]


```


# FullContext

## single_hop

### [✗ SAI] `#1915` Why did Dave start working on cars?

- gold: `Fascinated with how machines work`
- pred (real run): `Dave started working on cars ever since he was ten when he found an old car in a neighbor's garage and asked if he could fix it.`
- judge=False error=missing_info f1=0.0000 em=0.0 · diagnosis=**FULL_CONTEXT**
- context: 6512 token (26049 ký tự)
- prompt đầy đủ: `contexts/FullContext__locomo_0009__1915.txt`
- judge nói: The gold answer states that Dave was 'fascinated with how machines work', which explains his motivation. The AI answer provides a narrative about when he started working on cars but does not mention or convey the key reason—his fascination with machines. This key information is missing.
- evidence của dataset có trong prompt: **True**

```text
Use the conversation excerpts below to answer the question. Reply with only the answer — a few words or one sentence, no explanation.

Conversation:
[Session 1 — 11:53 am on 23 March, 2023]
[Calvin] Hey Dave! Nice to meet you! How's it going since we talked?
[Dave] Hey Calvin! Nice to meet you too! Things have been going well since we last talked. I attended this awesome event recently. It was like a car lover's paradise! They had so many classic cars on show, I was in awe. I even had the opportunity to speak with some of the owners and hear their fascinating stories. It was super inspiring. What about you? Anything exciting happening in your life lately?
[Calvin] That event sounds great! Something really exciting happened to me - I just had a big life change! Here's my new mansion - pretty cool huh? (image: a photo of a building with a sign on the front of it)
[Dave] Wow! Congrats on the big change! What inspired you to start this journey?
[Dave] Thanks, Calvin! Appreciate the kind words and support. Can't wait for your visit! Take care and keep creating amazing music! Check out pic of my garage, it looks stunning! (image: a photo of a car in a garage with a coca cola sign)
[Calvin] Thanks! I can't wait for your visit either. Take care and keep enjoying your hobbies!
[Dave] Sure thing! Thanks again for your help. Bye! Have a great day.
[Calvin] No problem! Always good chatting with you. Have an awesome day!
[Dave] Thanks, Calvin! Catch you later. Have a great day!
[Session 19 — 12:13 am on 15 September, 2023]
[Dave] Hey Calvin! Long time no talk! Got some cool news to shar

…[23850 chars omitted — full prompt in the .txt file]…

een the artist and the crowd is just amazing!
[Dave] Wow, it's amazing how that connection between artist and crowd can be indescribable. So glad you get to experience that!
[Calvin] Wow, Dave! It's a rush connecting with everyone. That feeling is unbeatable! Wishing you a harmonious day ahead, my friend!
[Dave] Yeah, I can imagine it's a rush being up on stage with all the fans cheering. Must be a unique experience. Wishing you many more electrifying moments in the spotlight! See you soon!

Question: Conversation between Calvin and Dave. Question: Why did Dave start working on cars?

Answer:

```

