# Báo cáo phân tích kết quả benchmark LoCoMo10

**Cấu hình:** frozen banks `static/memory_banks/locomo10/ds_fixed` · backbone + judge = Qwen3-4B-Instruct-2507
· 10 hội thoại · 1986 QA · 4 hệ thống (ASEM, FastASEM, FullContext, NoMemory)

**Ngày phân tích:** 2026-09-22
**Artifact nguồn:** `data/benchmarks/results/static/locomo10/ds_fixed__qwen_qwen3_4b_instruct_2507.{md,json}`,
`.../preds/`, `.../scores/`
**Phương pháp:** mọi con số dưới đây đọc trực tiếp từ artifact hoặc tính lại bằng script audit LLM-free
trong `scratch_diag/` (không gọi API, không cần GPU). Không có số liệu suy đoán.

---

## 0. TL;DR — kết luận và độ tin cậy

| # | Kết luận | Độ tin cậy | Bằng chứng |
|---|---|---|---|
| 1 | ASEM ≈ FullContext về **tổng thể** nhưng **không tương đương thống kê**: −2.6 pp, p = 0.047 | Cao | McNemar ghép cặp, n = 1986 |
| 2 | ASEM **thắng vững** ở adversarial (+16.6 pp) và temporal (+12.1 pp) | Cao | p < 1e-4 và p = 0.0009 |
| 3 | ASEM **thua** ở single_hop (−16.2 pp, 42% dữ liệu) và multi_hop (−8.9 pp) | Cao | p < 1e-4 và p = 0.018 |
| 4 | Nhãn loại câu hỏi trong harness **sai** với category 1, 3, 4 | Rất cao | evidence-turn: 3.13 / 2.08 / 1.07 |
| 5 | Ở nhóm FullContext thắng đậm, **100% evidence nằm nguyên văn ở cuối prompt** | Rất cao | replay `FullContext.answer`, n = 1986 |
| 6 | Lỗi single_hop của ASEM là **retrieval precision** (49%), không phải nén mất chi tiết | Cao | replay retriever thật, n = 162 |
| 7 | Lỗi multi_hop của ASEM là **ingestion/aggregation** (58%) | Cao | cùng phương pháp, n = 102 |
| 8 | FastASEM thua adversarial vì **từ chối** (29.6%, 0% được chấm đúng), không phải vì bịa | Cao | prompt guideline 3 vs 5 mâu thuẫn |
| 9 | Judge là **self-judge (cùng Qwen3-4B)** và có **2.4% verdict tự mâu thuẫn** | Cao | đọc `scores/*.jsonl` |
| 10 | Eval tĩnh **không đo được Stage 5 (RL/utility)**: `λ·z(q) ≡ 0` | Rất cao | `retriever._zscore` guard + `read_path` read-only |

---

## 1. Thiết lập thí nghiệm

| Thành phần | Giá trị |
|---|---|
| Bank | `static/memory_banks/locomo10/ds_fixed/` — ASEM re-ingest 2732 note; FastASEM copy từ tag `ds_nothink` 3504 note |
| Dataset | `datasets/locomo/locomo10.json`, 10 hội thoại, **1986 QA** |
| Backbone **và judge** | `configs/models/qwen3_4b_instruct_api.yaml` (không có trong repo; bản tương đương `configs/models/qwen3_4b_api.yaml`) |
| Ngân sách answer | `answer.context_window: 8192`, `answer.max_tokens: 512` → budget prompt 7616 token |
| Ngân sách mọi hệ thống | Dùng chung một công thức `prompt ≤ window − max_tokens − 64` (`asem/token_budget.py`) |
| Truy hồi (ASEM/FastASEM) | `k1=20`, `k2=5`, `δ=0.30`, RRF (dense 1.0 / BM25 0.8 / entity 0.6 / temporal 0.5), `rrf_k=60`, `max_context_notes: 8` |
| Metrics | `em, f1, rougeL, bertscore_f1, judge` |
| Thời gian chạy | 39 008 s (~10.8 h) cho 4 hệ thống × 1986 QA + 7944 judge call |
| Số hội thoại / hệ thống | 10/10, n mỗi hội thoại 105–260 (không còn tình trạng lệch n như các run trước) |

---

## 2. Kiến trúc hai hệ thống

### 2.1 Ba đường của pipeline

```
Ghi    : content → S1 NoteConstructor (P1_batch) → S2 MemoryManager (ADD/UPDATE/DELETE/NOOP)
                 → S3 LinkEvolver (P2 link + P3 evolve)                     [P1_batch_note_construction.txt]
Truy hồi: query → 3 kênh (dense ANN / BM25 / entity-match) → RRF → lọc δ
                 → Phase B (1−λ)·z(sim) + λ·z(q) → top-k2 → duyệt cạnh có nhãn → cap 8 note
Sinh    : 8 note → prompt (P_distil hoặc P_temporal_qa) → 1 LLM call → answer
```

### 2.2 Khác biệt ASEM v1 vs FastASEM (bản được đánh giá)

| | **ASEM** (v1) | **FastASEM** (v3) |
|---|---|---|
| Ingest | `ASEMPipeline.write_batch` → `build_batch` + `P1_batch_note_construction.txt`; link do `LinkEvolver` (có LLM) | `asem/fast_ingest.py`, prompt inline; link **không dùng LLM** (`_weave_graph_links`: trùng entity + kề thời gian + cosine ≥ 0.55) |
| Note | `X` = mô tả chưng cất, `c` = lượt thô, có `entities`/`speaker`/`session_date`/`L` | `X = c = fact` một câu |
| Prompt sinh | **`P_distil.txt`** — JSON `{selected_ids, answer}`, mô hình **tự chọn lại note** | **`P_temporal_qa.txt`** — trả lời trực tiếp, context là graph node `[n] date= speaker= entities= relations -> [k]` |
| Cơ chế | `AnswerAgent.distil_and_answer` + `ASEMPipeline._recover` (khi abstention: mở rộng k2/δ, hỏi lại 1 lần) | `AnswerAgent.direct_answer` |
| Trọng số link | `_RELATION_BOOST`: contradicts 1.35 > causal 1.25 > temporal 1.20 > extends 1.10 | cố định theo loại cạnh |

### 2.3 Ba lỗi config↔code ảnh hưởng trực tiếp tới tính công bằng

Đọc `eval/systems.py:build_asem_system` / `build_fast_asem_system`:

1. **`answer.direct_mode: true` không được truyền** cho `AnswerAgent` của ASEM v1 → ASEM chạy
   đường distil, FastASEM chạy direct. Hai hệ thống **không cùng giao thức sinh**.
2. **Khối `retriever:` không được truyền** → ASEM dùng default dataclass `max_link_hops=1` trong khi
   config ghi `max_hops: 2`; `link_traversal_topn=3` cũng lấy default.
3. Trong eval tĩnh, `read_path` là read-only → mọi note giữ `q0 = 0.5` → `_zscore` có guard
   `std == 0 → [0.0]*n` → **`λ·z(q) ≡ 0`**, tức λ (0.25 factual / 0.40 default / 0.60 reasoning)
   **không thể** thay đổi thứ hạng, và `_traverse_links` cũng nhân hệ số `(0.5 + 0.5·q)` không đổi.
   ⇒ Trong giao thức static-bank, phần "RL/value-aware" của ASEM **không hoạt động**.

---

## 3. Kết quả

### 3.1 Tổng thể

| System | n | EM | F1 | ROUGE-L | BERTScore-F1 | Judge | Errors |
|---|---|---|---|---|---|---|---|
| ASEM | 1986 | 0.0055 | 0.2372 | 0.2236 | 0.8563 | **0.6782** | 0 |
| FastASEM | 1986 | 0.0740 | 0.2821 | 0.2683 | 0.8688 | 0.5000 | 0 |
| FullContext | 1986 | 0.0745 | **0.3485** | **0.3361** | **0.8773** | **0.7039** | 0 |
| NoMemory | 1986 | 0.0035 | 0.0678 | 0.0625 | 0.8288 | 0.1843 | 0 |

- ASEM đạt **96.3%** mức judge của FullContext (0.6782 / 0.7039).
- Tính theo phần cải thiện so với NoMemory: (0.6782−0.1843)/(0.7039−0.1843) = **95.0%** "lợi ích của
  ngữ cảnh đầy đủ" — xác nhận lại con số 95% trong phân tích ban đầu.
- Tỉ lệ Judge/F1: ASEM 2.86, FullContext 2.02, FastASEM 1.77 → ASEM diễn giải lại nhiều nhất.

### 3.2 Theo loại câu hỏi (nhãn đã sửa — xem §4.1)

| Loại đúng (n) | System | EM | F1 | BERTScore | Judge |
|---|---|---|---|---|---|
| **adversarial** (446) | ASEM | 0.0045 | 0.2112 | 0.8577 | **0.5426** |
| | FastASEM | 0.0112 | 0.0847 | 0.8390 | 0.1166 |
| | FullContext | 0.0471 | 0.2268 | 0.8622 | 0.3767 |
| | NoMemory | 0.0045 | 0.0806 | 0.8412 | 0.2175 |
| **temporal** (321) | ASEM | 0.0280 | 0.2786 | 0.8417 | **0.7040** |
| | FastASEM | 0.0561 | **0.4182** | 0.8766 | 0.6106 |
| | FullContext | 0.0218 | 0.3030 | 0.8545 | 0.5826 |
| | NoMemory | 0.0000 | 0.0057 | 0.7753 | 0.0187 |
| **open_domain** (96) | ASEM | 0.0000 | 0.0959 | 0.8327 | 0.5000 |
| | FastASEM | 0.0104 | 0.1510 | 0.8515 | 0.4375 |
| | FullContext | 0.0104 | 0.1529 | 0.8483 | 0.5312 |
| | NoMemory | 0.0208 | 0.1238 | 0.8387 | 0.2708 |
| **multi_hop** (282) | ASEM | 0.0000 | 0.1886 | 0.8486 | 0.6170 |
| | FastASEM | 0.0426 | 0.2676 | 0.8685 | 0.4752 |
| | FullContext | 0.0390 | 0.2889 | 0.8684 | **0.7057** |
| | NoMemory | 0.0000 | 0.0504 | 0.8316 | 0.1028 |
| **single_hop** (841) | ASEM | 0.0000 | 0.2676 | 0.8663 | 0.7812 |
| | FastASEM | 0.1320 | 0.3547 | 0.8837 | 0.6766 |
| | FullContext | 0.1284 | **0.4727** | **0.9003** | **0.9429** |
| | NoMemory | 0.0036 | 0.0841 | 0.8407 | 0.2473 |

Chênh ASEM − FullContext (judge): adversarial **+0.166**, temporal **+0.121**,
open_domain −0.031, multi_hop −0.089, single_hop **−0.162**.

### 3.3 Theo hội thoại (judge)

| Conv | n | ASEM | FastASEM | FullContext | NoMemory | ASEM − FC |
|---|---|---|---|---|---|---|
| locomo_0000 | 199 | 0.7387 | 0.5327 | 0.7387 | 0.1709 | 0.000 |
| locomo_0001 | 105 | 0.6667 | 0.4667 | 0.7238 | 0.2095 | −0.057 |
| locomo_0002 | 193 | 0.7098 | 0.5285 | 0.7098 | 0.1503 | 0.000 |
| locomo_0003 | 260 | 0.6538 | 0.4385 | 0.6346 | 0.1923 | +0.019 |
| locomo_0004 | 242 | 0.7066 | 0.4587 | 0.6653 | 0.2025 | +0.041 |
| locomo_0005 | 158 | 0.5886 | 0.5443 | 0.6962 | 0.2405 | −0.108 |
| locomo_0006 | 190 | 0.6842 | 0.5421 | 0.7158 | 0.1053 | −0.032 |
| locomo_0007 | 239 | 0.6653 | 0.5314 | 0.7280 | 0.1590 | −0.063 |
| locomo_0008 | 196 | 0.6327 | 0.5204 | 0.7041 | 0.2755 | −0.071 |
| locomo_0009 | 204 | 0.7157 | 0.4559 | 0.7549 | 0.1569 | −0.039 |

- ASEM ổn định trong khoảng 0.589–0.739; ngang FullContext ở 0000/0002, hơn ở 0003/0004,
  kém rõ ở 0005 (−10.8 pp) và 0008 (−7.1 pp).
- FastASEM thấp đều ở **cả 10** hội thoại (0.4385–0.5443) → hạn chế mang tính hệ thống.
- n mỗi hội thoại 105–260 ⇒ sai số từng hội thoại ±6–9 pp; không nên diễn giải chênh lệch nhỏ.

### 3.4 Kiểm định ghép cặp (McNemar chính xác + bootstrap 2000 lần, ASEM vs FullContext)

| Tập | n | ASEM | FC | Chênh | b (chỉ ASEM đúng) | c (chỉ FC đúng) | p | 95% CI |
|---|---|---|---|---|---|---|---|---|
| Tất cả | 1986 | 0.678 | 0.704 | −2.6 pp | 292 | 343 | **0.0471** | [−5.0, −0.1] pp |
| adversarial | 446 | 0.543 | 0.377 | **+16.6 pp** | 126 | 52 | **<1e-4** | [+11.0, +22.4] |
| temporal | 321 | 0.704 | 0.583 | **+12.1 pp** | 86 | 47 | **0.0009** | [+5.6, +18.7] |
| single_hop | 841 | 0.781 | 0.943 | **−16.2 pp** | 27 | 163 | **<1e-4** | [−19.3, −13.1] |
| multi_hop | 282 | 0.617 | 0.706 | −8.9 pp | 39 | 64 | 0.0176 | [−16.0, −2.5] |
| open_domain | 96 | 0.500 | 0.531 | −3.1 pp | 14 | 17 | 0.7201 | [−13.5, +8.3] |
| Bỏ adversarial | 1540 | 0.718 | 0.799 | **−8.1 pp** | 166 | 291 | **<1e-4** | [−10.9, −5.5] |

**Đọc đúng:** khác biệt tổng thể tuy nhỏ nhưng **có thật**; không được phát biểu "ASEM ngang bằng
FullContext". Thành tích của ASEM phụ thuộc đáng kể vào nhóm adversarial: bỏ nhóm này, khoảng cách
tăng từ 2.6 lên 8.1 pp. Nhóm open_domain (n = 96) không đủ lực để kết luận theo cả hai hướng.

---

## 4. Phân tích nguyên nhân

### 4.1 Nhãn loại câu hỏi sai — đã kiểm chứng

`CATEGORY_NAMES` (`scripts/run_locomo10_experiments.py:85`) gán 1=`single_hop`,
3=`commonsense`, 4=`conversational`. Đếm evidence turn trung bình trong `locomo10.json`:

| category | nhãn harness | n | evidence turn (mean) | % câu cần > 1 turn |
|---|---|---|---|---|
| 1 | `single_hop` | 282 | **3.13** | **98%** |
| 2 | `temporal` | 321 | 1.17 | 12% |
| 3 | `commonsense` | 96 | **2.08** | 49% |
| 4 | `conversational` | 841 | **1.07** | **5%** |
| 5 | `adversarial` | 446 | 1.03 | 3% |

⇒ category 1 là **multi_hop**, category 4 là **single_hop**, category 3 là **open_domain** /
suy luận. Phân bố n (282 / 321 / 96 / 841 / 446) khớp phân bố chuẩn của LoCoMo10.

**Hệ quả:** mọi kết luận phải đọc lại — ASEM yếu ở **single_hop và multi_hop**, không phải ở
"hội thoại" nói chung. Cần sửa cả code (để các run sau tự đúng) và phần diễn giải trong báo cáo cũ.

### 4.2 FullContext trong harness này gần như được trao sẵn đáp án

`_build_history_for_qa` dựng history = **toàn bộ lượt từ session 1 đến session chứa evidence, và
bao gồm cả lượt evidence**. Replay đúng đường production (`asem/token_budget.py` +
`eval/baselines.py:FullContext.answer`, spy backend, ngân sách 8192/512):

| Loại | history (lượt, p50) | giữ lại (p50) | % bị cắt | prompt token p50 | **% giữ toàn bộ evidence** | vị trí evidence cuối |
|---|---|---|---|---|---|---|
| single_hop | 358 | 205 | 69.7% | 7588 | **100.0%** | **0.94** |
| adversarial | 351 | 202 | 69.1% | 7586 | 100.0% | 0.93 |
| temporal | 302 | 198 | 62.0% | 7585 | 96.0% | 0.92 |
| multi_hop | 418 | 208 | 81.6% | 7593 | **51.1%** | 0.91 |

Ở đúng nhóm FullContext thắng đậm nhất (841 câu, 42% dữ liệu), câu trả lời **nằm nguyên văn ở cuối
prompt** và FullContext chỉ cần chép lại. Bằng chứng độc lập: chênh F1 ở nhóm này là +0.205 trong
khi chênh judge chỉ +0.162 → lợi thế F1 là **sao chép**, không phải hiểu đúng hơn.

Ở nhóm cần ghép nhiều turn (multi_hop) evidence chỉ còn 51% và FullContext **thua** ASEM 8.9 pp.

**Hệ quả phương pháp luận:** đây không phải bài needle-in-a-haystack. Để đo khả năng đọc ngữ cảnh dài
thật, phải dựng biến thể history đóng ở session **trước** evidence (hoặc loại các session có evidence).

### 4.3 single_hop: lỗi là **retrieval precision**, không phải nén mất chi tiết

`scratch_diag/retrieval_recall_qwen.py` replay **đúng retriever production** trên bank `ds_fixed`
(embedder thật, BM25/entity cache trong RAM, không gọi LLM). Định nghĩa "note mang đáp án" = note có
các trường (`c`, `X`, `K`, `G`, `entities`, `session_date`) phủ ≥ 60% token của gold.

| Lát cắt | n | INGESTION (gold không có trong bank) | RETRIEVAL (có trong bank, không vào context) | GENERATION (có trong context vẫn sai) |
|---|---|---|---|---|
| Tất cả ASEM sai | 628 | 30% | 41% | 29% |
| single_hop: ASEM sai & FC đúng | 162 | 18% | **49%** | 33% |
| single_hop: ASEM **đúng** (đối chứng) | 653 | 8% | 8% | 84% |
| multi_hop: ASEM sai | 102 | **58%** | 29% | 13% |

Ví dụ lỗi retrieval (bank có note đúng, nhưng note khác được chọn):

| idx | Câu hỏi | Gold | ASEM | FullContext |
|---|---|---|---|---|
| 104 | What book did Caroline recommend to Melanie? | `"Becoming Nicole"` | "Caroline recommended \"Charlotte's Web\" to Melanie." | `"Becoming Nicole" by Amy Ellis Nutt` |
| 112 | What did Mel and her kids paint in July 2023? | `a sunset with a palm tree` | "…painted a dog-faced cup in their pottery workshop on 7 July 2023." | "…painted a sunset with a palm tree…" |
| 118 | What did Melanie and her family see while camping last year? | `Perseid meteor shower` | "…enjoyed marshmallow roasting, campfire stories…" | "…saw the Perseid meteor shower…" |
| 137 | What painting did Melanie show Caroline on October 13, 2023? | `A painting inspired by sunsets with a pink sky.` | "…her blue abstract painting on October 13, 2023." | "A painting of a sunset with a pink sky." |

Đối chứng: ở 653 câu single_hop mà ASEM trả lời **đúng**, tỉ lệ mất retrieval chỉ **8%** → retriever
làm đúng khi note mang đáp án được xếp hạng cao; vấn đề là **precision**, không phải năng lực ingest.

Ví dụ lỗi generation do mô tả note bị trừu tượng hoá:

| idx | Gold | ASEM |
|---|---|---|
| 113 | `many people wanting to create loving homes for children in need` | "Caroline found the council meeting for adoption inspiring and emotional." |
| 85 | `researching adoption agencies` | "Caroline plans a special summer trip just with Melanie to catch up and explore nature." |

### 4.4 multi_hop: lỗi là **ingestion / aggregation**

58% câu multi_hop mà ASEM trả lời sai thì gold **không có note nào** phủ đủ. Đây là giới hạn kiến
trúc: note là *fact per-turn*, còn gold của nhiều câu multi_hop là **hợp nhất nhiều phiên**:

| Câu hỏi | Gold | ASEM |
|---|---|---|
| What activities does Melanie partake in? | `pottery, camping, painting, swimming` | "Melanie participates in running, family activities, and painting." |
| What do Melanie's kids like? | `dinosaurs, nature` | "…pottery, playing outdoors at the park, attending a birthday concert…" |
| What books has Melanie read? | `"Nothing is Impossible", "Charlotte's Web"` | "Melanie's favorite childhood book was \"Charlotte's Web\"." |
| How many dogs has Maria adopted from the shelter she volunteers at? | `two` | "Maria has adopted one dog, named Shadow." |

### 4.5 adversarial: ASEM thắng vì trả lời fact gốc; FastASEM thua vì **từ chối**

| | n | ASEM | FastASEM | FullContext | NoMemory |
|---|---|---|---|---|---|
| Judge | 446 | 54.3% | 11.7% | 37.7% | 21.7% |
| % từ chối | 446 | 2.9% | **29.6%** | 9.2% | 40.1% |
| Judge trên câu bị từ chối | 446 | 8% | **0%** | 5% | 1% |
| Judge trên câu có trả lời | 446 | 56% | 17% | 41% | 36% |

132/446 câu adversarial của FastASEM bị từ chối, và **0% trong số đó được chấm đúng** vì gold tồn tại.
Ví dụ: gold `'Sweden'`, FastASEM → *"The memory notes do not mention Melanie's grandma or her country
of origin."*

Nguyên nhân nằm ngay trong `data/prompts/P_temporal_qa.txt`, hai guideline mâu thuẫn:

- guideline 3: *"If the question credits a fact to the wrong person, STILL GIVE THE FACT and name the
  correct speaker… A mismatch between the names in the question and the note's speaker is NOT a reason
  to refuse."*
- guideline 5: *"Only answer "I don't know" if NO note contains any information bearing on the question
  — **including the case where the fact belongs to a different person than the question names**"*

Cách diễn đạt của guideline 5 bị đọc thành "được phép từ chối khi fact thuộc người khác", và mô hình 4B
chọn từ chối. Lưu ý semantics của LoCoMo: gold của category 5 là **fact gốc** (ví dụ
`"What does Melanie's necklace symbolize?"` → `love, faith, and strength`), nên **từ chối = sai theo
thiết kế**.

⇒ Khuyến nghị trong bản phân tích ban đầu ("bổ sung bước kiểm tra bằng chứng/từ chối cho FastASEM")
là **ngược dấu**: phải **bỏ** quyền từ chối đó.

### 4.6 temporal: ASEM thắng, nhưng cẩn thận cách chấm ngày

ASEM 0.704 > FastASEM 0.611 > FullContext 0.583 (NoMemory 0.019). Hợp lý vì bộ nhớ có cấu trúc gắn
`session_date` cho từng note, trong khi FullContext phải tự tìm mốc thời gian trong ngữ cảnh dài.

Hai điều cần lưu ý khi diễn giải:

1. **Gold tương đối vs đáp án tuyệt đối.** Nhiều gold là mốc *tương đối* neo vào session hỏi, ví dụ
   `"The Friday before 15 July 2023"`, trong khi hệ thống trả về ngày tuyệt đối `"Friday 7 July 2023"`.
   Cả hai có thể đúng/lệch nhau tuỳ session neo; judge 4B xử lý rất không nhất quán.
2. **ASEM sai ở dạng "đúng chủ đề, sai mốc"**: ví dụ `#54` gold `The week before 23 August 2023` vs
   pred `mid-August 2023`; `#49` gold `2022` vs pred `11 August 2023` (bị judge chấm là *đúng* — xem §5.1).

### 4.7 open_domain: không kết luận được

n = 96 → CI ±10 pp. ASEM 0.500 vs FC 0.531, p = 0.72. Không có ý nghĩa thống kê, và cũng không có
dấu hiệu ASEM tốt hơn. Riêng nhóm này tải ingestion cao (23/26 câu IDK trước đây thuộc ingestion) vì
câu hỏi đòi suy luận/counterfactual mà note lưu fact thô không chứa.

### 4.8 Nghịch lý F1/EM vs judge

| System | p50 ký tự | p50 từ | % từ chối | F1 | EM | Judge | Judge/F1 |
|---|---|---|---|---|---|---|---|
| ASEM | 78 | 14 | 1.7% | 0.237 | 0.0055 | 0.678 | 2.86 |
| FastASEM | 54 | 9 | 11.4% | 0.282 | 0.0740 | 0.500 | 1.77 |
| FullContext | 64 | 11 | 3.0% | 0.348 | 0.0745 | 0.704 | 2.02 |
| NoMemory | 51 | 8 | 47.5% | 0.068 | 0.0035 | 0.184 | 2.72 |

- ASEM trả lời thành câu kèm mệnh đề phụ (attribution, "which suggests…", "family roots") → precision
  giảm → EM ≈ 0 dù nội dung đúng. Hiện tượng **lặp lại ở cả 10 hội thoại** và ở cả 5 nhóm ⇒ khác biệt
  phong cách hệ thống, không phải nhiễu.
- Chênh F1 của FullContext tập trung ở single_hop (+0.205) ⇒ phần lớn là **sao chép nguyên văn** (§4.2).
- FastASEM có F1 cao hơn ASEM nhưng judge thấp hơn 17.8 pp ⇒ F1 cao không hàm ý tốt hơn.

---

## 5. Hiệu chỉnh dụng cụ đo

### 5.1 Judge (LLM-as-a-judge, `eval/llm_as_a_judge.py`)

Header của file kết quả ghi judge config **trùng** config backbone ⇒ **self-judge (Qwen3-4B chấm
Qwen3-4B)**, có nguy cơ tự ưu tiên và không đủ lực phân biệt các hệ thống gần nhau.

| System | judge=True & có chuỗi gold trong câu trả lời | judge=True & **thiếu** chuỗi gold | judge=True nhưng `judge_error ≠ null` |
|---|---|---|---|
| ASEM | 610 (30.7%) | 737 (37.1%) | **48 (2.4% toàn bộ)** |
| FastASEM | 458 (23.1%) | 535 (26.9%) | 31 (1.6%) |
| FullContext | 691 (34.8%) | 707 (35.6%) | 37 (1.9%) |
| NoMemory | 53 (2.7%) | 313 (15.8%) | 7 (0.4%) |

- **Verdict tự mâu thuẫn**: có dòng `judge_correct=true` kèm `judge_error="contradiction"` và
  reasoning ghi nguyên văn *"specifies a date in 2023, which contradicts the gold answer of 2022"*
  (idx 49, 324, 505). Tỉ lệ 2.4% **xấp xỉ chính hiệu số 2.6 pp** đang được tranh luận ⇒ không thể
  dùng judge đơn lẻ để khẳng định "ngang bằng".
- **Quá dễ dãi với câu diễn giải**: gold `"Single"` được chấp nhận cho câu trả lời về "transition
  changed her relationships"; gold `2022` được chấp nhận cho câu trả lời `2023`.
- Trong các dòng temporal được chấm đúng, **28% (ASEM) / 31% (FastASEM) / 42% (FullContext)** không
  chứa con số nào của gold ⇒ tỉ lệ dương tính giả có thể tới ~3–4% toàn bộ mẫu.

**Khuyến nghị:** dùng judge mạnh hơn (hoặc ít nhất khác model), thêm kiểm tra chéo
`is_correct` ↔ `error`, và báo cáo kèm nhiễu đo được.

### 5.2 Metric từ vựng

- **EM gần 0 ở mọi hệ thống** (< 0.075) ⇒ gần như vô nghĩa ở đây. Nếu muốn dùng, phải chuẩn hoá/ép
  định dạng ngắn trước khi tính.
- **BERTScore bị nén dải** 0.775–0.900 (chênh giữa 4 hệ thống chỉ ~0.05) vì chưa rescale với baseline
  ⇒ nên bật `rescale_with_baseline=True` cho `evaluate.load("bertscore")`.
- Judge là **nhị phân** (242/446 = 0.5426 khớp chính xác) ⇒ với n = 1986, CI 95% ≈ ±2 pp; riêng
  open_domain (n = 96) ≈ ±10 pp.

### 5.3 Chi phí và độ trễ — còn thiếu trong bảng điểm

Hệ thống bộ nhớ chỉ có giá trị nếu đạt độ chính xác tương đương với **ít token hơn**. Lần chạy này
không ghi token/độ trễ theo từng dòng, nhưng số đo từ tooling sẵn có
(`scratch_diag/measure_context.py`, `scratch_diag/context_size.py`):

| Hệ thống | prompt trả lời p50 | max |
|---|---|---|
| ASEM (distil + recovery) | ≈ 3.4k token | 4.8k |
| FastASEM (`direct_answer`) | ≈ 0.5k token | 0.9k |
| FullContext (sau cắt ở 8192) | ≈ 7.6k token | 7.6k |

⇒ ASEM rẻ hơn FullContext **~2.2×** mỗi câu. Đây là luận điểm nên đưa vào bảng kết quả chính, cùng
cột "token/câu" và "giây/câu" để `run_static_eval` bổ sung.

### 5.4 Thiếu baseline

Chưa có RAG chuẩn (BM25 / dense top-k thuần) và chưa có hệ thống bộ nhớ khác (Mem0, A-Mem, LongMem…)
⇒ chưa tách được đóng góp riêng của cấu trúc ASEM so với "retrieval + prompt".

### 5.5 Eval tĩnh không đo được Stage 5

Xem §2.3 điểm 3: `q` không bao giờ được cập nhật trong `read_path` ⇒ kênh utility/RL bị vô hiệu hoá
trong mọi lần eval tĩnh. Nếu luận văn muốn chứng minh giá trị của `q`, phải chạy biến thể có
`ValueRetrievalOnly`/vòng reward, hoặc báo cáo rõ đây là ablation chưa đo.

---

## 6. Đối chiếu với bản phân tích ban đầu

| Nhận định trong bản phân tích trước | Kết quả kiểm chứng |
|---|---|
| ASEM ≈ FullContext về ngữ nghĩa (judge) | **Một phần đúng**: 96.3% mức judge, nhưng khác biệt **có ý nghĩa thống kê** (p = 0.047) và thành 8.1 pp nếu bỏ adversarial |
| F1/EM ngược chiều với judge | **Đúng**, và nguyên nhân chính xác hơn: FullContext được **sao chép nguyên văn** ở single_hop; ASEM thêm mệnh đề phụ |
| FastASEM đánh đổi quá nhiều độ chính xác | **Đúng** (judge 0.500), nhưng lý do không phải "ngữ cảnh làm mô hình tự tin sai" mà là **từ chối trả lời** (29.6%) |
| NoMemory là mốc sàn hợp lý | Đúng (0.184) |
| Nhãn loại câu hỏi có thể bị gán sai | **Đúng** — đã xác minh bằng evidence-turn (1↔4 đảo, 3 = open_domain) |
| ASEM mạnh ở adversarial nhờ biết từ chối/không bịa; nên kiểm tra thủ công | **Nửa đúng**: ASEM *không* từ chối (2.9%), nó trả lời **fact gốc** đúng như gold cat 5 mong đợi. Cần kiểm tra thủ công: đồng ý, và judge tỏ ra quá dễ dãi |
| FastASEM adversarial thấp hơn NoMemory ⇒ lọc/xác minh bằng chứng yếu ⇒ **nên thêm từ chối** | **Sai dấu**: cần **bỏ** quyền từ chối (guideline 5 mâu thuẫn guideline 3) |
| Conversational (841) là điểm yếu lớn nhất | **Đúng về số**, nhưng đó là **single_hop**, và FullContext ở đó được trao đáp án ⇒ kết luận phải nêu bối cảnh |
| single_hop (282) thua 8.9 điểm do recall | Đúng về hướng, nhưng kèm bối cảnh: gold thường cần **hợp nhất nhiều phiên** (58% là ingestion) |
| Khoảng cách ASEM–FC 2.6 pp "sát ngưỡng", cần kiểm định ghép cặp | **Đúng và đã làm** → p = 0.047, CI [−5.0, −0.1] pp (không giao 0) |
| Judge nên ghi rõ model, có thể thiên vị | **Đúng**: là self-judge, và 2.4% verdict tự mâu thuẫn |
| BERTScore nén dải, nên rescale | Đúng (0.775–0.900) |
| EM gần 0 nên vô nghĩa | Đúng |
| Chưa đo chi phí (token/độ trễ) | Đúng; số đo sẵn có: ASEM ≈ 3.4k vs FullContext ≈ 7.6k token/câu |
| Thiếu baseline RAG và Mem0/A-Mem | Đúng |
| Chỉ một model nhỏ 4B nên ưu thế temporal có thể thu hẹp ở model lớn | Hợp lý, chưa có dữ liệu để kết luận |

---

## 7. Việc cần làm

| # | Việc | File | Kỳ vọng |
|---|---|---|---|
| 1 | Sửa `CATEGORY_NAMES` (1→multi_hop, 3→open_domain, 4→single_hop) | `scripts/run_locomo10_experiments.py` | Sửa cách diễn giải toàn bộ bảng per-category; rẻ, làm ngay |
| 2 | Bỏ quyền từ chối ở guideline 5, làm rõ guideline 3 | `data/prompts/P_temporal_qa.txt` | FastASEM adversarial kỳ vọng +15–20 pp (29.6% → gần 0 từ chối) |
| 3 | Truyền `direct_mode` và khối `retriever` khi build ASEM v1 | `eval/systems.py` | Công bằng giao thức sinh; `max_hops` 1 → 2 như config |
| 4 | Lưu `used_notes` (+ id note mang gold nếu biết) vào record | `eval/static_eval.py` | Bỏ được bước replay retriever khi phân tích lỗi |
| 5 | Rerank top-20 (cross-encoder hoặc RRF có trọng số theo `X`), chỉnh `k2`/λ | `asem/retriever.py` | Nhắm 49% lỗi retrieval ở single_hop |
| 6 | Note tổng hợp theo chủ đề/người (union nhiều phiên) | `asem/batch_ingestion.py`, `asem/fast_ingest.py` | Nhắm 58% lỗi ingestion ở multi_hop |
| 7 | Judge bằng model khác + lọc verdict tự mâu thuẫn (`is_correct` vs `error`) | `eval/llm_as_a_judge.py` | Loại ~2.4% nhiễu, ngang độ lớn hiệu ứng đang đo |
| 8 | Ghi `tokens` và `elapsed` mỗi dòng; thêm bảng chi phí vào báo cáo | `eval/static_eval.py`, `eval/static_eval.render_report_table` | Biến "rẻ hơn 2.2×" thành số liệu chính thức |
| 9 | Thêm arm `FullContext-no-evidence` (history đóng trước session evidence) | `scripts/run_locomo10_experiments.py` | Đo long-context thật, không bị oracle context |
| 10 | Thêm baseline BM25/dense top-k và (nếu được) Mem0/A-Mem | `eval/baselines.py` | Tách đóng góp của cấu trúc ASEM |
| 11 | Chạy lại Phase B sau khi đổi định dạng context; **re-ingest** nếu muốn caps ở tầng lưu có hiệu lực | `scripts/build_static_banks.py`, `scripts/run_static_eval.py` | Đo tác động accuracy của định dạng mới (§9) |

---

## 8. Context samples — context thật của từng ca (đúng và sai)

Artifact: **`scratch_diag/context_samples/`**

| File | Nội dung |
|---|---|
| `REPORT.md` | Bản đọc được: §A note mang gold bị bỏ khỏi context · §B bank thiếu note mang gold · §C FullContext mất evidence · rồi từng ca theo hệ thống → loại → sai/đúng, kèm excerpt context |
| `contexts/<System>__<conv>__<idx>.txt` | **Prompt đầy đủ** (header metadata + toàn bộ context gửi cho model) |
| `index.jsonl` | 1 dòng/ca: metadata, verdict, `diagnosis`, `gold_bearing_note_ids`, `retrieved_note_ids`, `evidence_in_prompt`, retriever stats, kích thước prompt, đường dẫn file context |
| `manifest.json` | Tham số dump (mẫu, seed, danh sách system/category) để tái lập đúng tập ca |

**Tập mẫu hiện tại: 96 ca** — 24 ca mỗi hệ thống = 2 ca sai + 1 ca đúng cho mỗi (hệ thống × loại câu hỏi),
cộng 9 idx bắt buộc (`49, 104, 112, 118, 137, 157, 159, 324, 505`), seed = 0.
Phân bố: `GENERATION` 19 · `INGESTION` 15 · `RETRIEVAL` 14 · `FULL_CONTEXT` 24 · `NO_RETRIEVAL` 24.

### 8.1 Vì sao replay là trung thực

`scratch_diag/dump_context_samples.py` dựng lại từng hệ thống trên bank `ds_fixed` với **probe backend**
(ghi prompt, trả câu trả lời giả, không gọi LLM):

1. Bank system chạy đúng `read_path(strip_query_prefix(query))` với cùng retriever, cùng
   `max_context_notes` và cùng ngân sách token (`answer_budget_from_config`).
2. `MemoryBank.list_notes` được cache 1 lần/bank — note không đổi trong eval read-only, nên
   `bm25_search` / `search_by_entities` / `ann_search` trả về đúng kết quả của lần chạy thật.
3. Câu trả lời giả là JSON hợp lệ (`{"selected_ids": [], ...}`) và **không** bị coi là abstention,
   nên `_recover` không chạy ⇒ prompt ghi lại **chính là prompt first-pass** đã tạo ra `preds/*.jsonl`.

### 8.2 Phát hiện mới từ dump (bổ sung cho §4.5)

Ở FastASEM, các ca adversarial bị từ chối (`#152`, `#157`, `#159`) rơi vào bucket **RETRIEVAL**, không
phải GENERATION: note mang fact (ví dụ `0837ba0c-ceed-4598-bc7d-61fc657dc28e` cho "Melanie's necklace")
**có trong bank nhưng không được lấy**, vì query gọi sai tên người nên kênh dense/BM25/entity kéo về
các note của người kia. Cùng câu hỏi đó, ASEM trả lời **đúng**.

⇒ Thất bại adversarial của FastASEM có **hai** nguyên nhân độc lập:
(a) guideline 5 trong `P_temporal_qa.txt` cho phép từ chối khi sai người (sửa ở prompt), và
(b) truy hồi không lấy được note mang fact khi câu hỏi mis-attribute (sửa ở retriever: thêm kênh
entity của *người bị gán sai*, hoặc truy hồi hai chiều "fact thuộc ai").
Chỉ sửa (a) là chưa đủ.

### 8.3 Chạy lại / mở rộng

```powershell
conda activate memory-r1
# mặc định: 4 hệ thống, 5 loại, 2 sai + 1 đúng mỗi ô
python scratch_diag/dump_context_samples.py
# chỉ một hệ thống và một loại, lấy nhiều ca sai hơn
python scratch_diag/dump_context_samples.py --systems ASEM --categories single_hop --wrong-per-cell 10
# dump một ca cụ thể
python scratch_diag/dump_context_samples.py --must-have 104 --wrong-per-cell 0 --right-per-cell 0
# đổi nơi lưu
python scratch_diag/dump_context_samples.py --out scratch_diag/context_samples_v2
```

---

## 9. Thiết kế lại context truy vấn (numbered blocks)

Ca `#1941` trong `scratch_diag/context_samples/` cho thấy vấn đề rõ nhất: context cũ là **một dòng JSON**
chứa 8 object, mỗi object có `id` UUID 36 ký tự, `keywords` tới 49 phần tử, `description` dài tới
~1.700 ký tự (một đoạn văn gộp hơn 10 lượt thoại của nhiều người, có `\u2014` escape), kèm
`relations[].target_id` là UUID của note khác, và `also_linked` dạng `"extends:3, same-topic:3"`.
Ba hệ quả:

1. **Không đọc được.** 4B phải tự tách câu, tự đoán câu nào của ai, tự giải mã escape.
2. **Phí token.** Trường `relations` từng chiếm 49.9% prompt; `id` + `target_id` là phần lớn phần còn lại.
3. **Sai cơ chế chọn note.** Model phải **copy lại UUID** vào `selected_ids`. Một ký tự sai → parser
   trả về rỗng → code rơi vào fallback "dùng tất cả candidates", tức mất luôn tín hiệu chọn lọc.

### 9.1 Định dạng mới

```text
[1] 20 April 2023 · said by Calvin · about: Calvin
      • Calvin learned a lot and received great advice from music industry professionals, including a producer
        who told him to stay true to himself and sound unique, prompting him to think about his music's
        direction; Dave found Calvin's authenticity motivating and asked where he sees his music taking him.
      • Calvin also said he started making music to follow his dreams …
      • [+1 more fact(s) in this note, not shown]
      turn: "[Calvin] I learned a lot and got some great advice from professionals in the music industry. It was
             inspiring!"
      topics: calvin, advice, music industry, professionals, producer, unique sound, music direction, dave
      links: [5] same-topic, [6] extends  (+7 more not in this list: extends:3, same-topic:3, semantic:1)
```

| Thay đổi | Trước | Sau |
|---|---|---|
| Định danh note | UUID 36 ký tự | **số thứ tự `[n]`** (cũng là giá trị `selected_ids` trả về) |
| Đích của link | `target_id` UUID | **số thứ tự** `[5]` + tên quan hệ |
| `description` | 1 khối JSON, tới 1.700 ký tự | **bullet theo câu**, tối đa 4 câu / 420 ký tự, có dấu `[+k more fact(s)]` |
| Ngày | `2023-04-20T16:15:00Z` | `20 April 2023` |
| Attribution | trường `speaker` rời | `said by Calvin` ngay trên header |
| `keywords` | tới 49 phần tử | `topics:` tối đa 8 |
| `also_linked` | chuỗi count | `(+7 more not in this list: …)` gắn vào dòng `links:` |
| `content` | luôn in, cắt giữa câu | chỉ in khi bổ sung từ mà `description` không có |

### 9.2 Số đo (replay thật trên bank `ds_fixed`, ASEM, 61 câu)

| | p50 | mean | max |
|---|---|---|---|
| OLD chars | 7 913 | 8 090 | 13 282 |
| NEW chars | 5 577 | 5 544 | 7 099 |
| OLD tokens | 1 978 | 2 022 | 3 320 |
| NEW tokens | **1 394** | 1 386 | **1 774** |

→ context **nhỏ hơn 31.5%** (1.46×), riêng prompt dài nhất giảm từ 3 320 → 1 774 token.
Xem `scratch_diag/context_format_preview.md` (before/after cho `#1941, 104, 157, 505, 49`),
số liệu thô trong `scratch_diag/_render_measure.txt`;
tái lập: `python scratch_diag/render_context_preview.py --measure 60`.

### 9.3 Sửa luôn nguyên nhân gốc ở tầng lưu

Mega-`description` không phải do extraction (P1 đã yêu cầu "1-2 câu") mà do **memory evolution** (P3)
không tuân thủ. Prompt đã chặt hơn **và** giới hạn được ép ở write path:

| Thay đổi | File |
|---|---|
| `cap_description()` — cắt theo ranh giới câu, mặc định 600 ký tự | `asem/note.py` |
| `cap_keywords()` — dedupe (không phân biệt hoa/thường) + tối đa 12 | `asem/note.py` |
| Áp cap khi evolution ghi lại note (batch + single) | `asem/link_evolver.py` |
| Áp cap khi ingest (ASEM v2 batch, FastASEM) | `asem/batch_ingestion.py`, `asem/fast_ingest.py` |
| P3 batch/đơn: "1–2 câu, ≤ 45 từ, REWRITE, không append" | `data/prompts/P3_batch_evolution.txt`, `P3_memory_evolution.txt` |
| P1/P4: "≤ 45 từ, một fact mỗi note" | `data/prompts/P1_note_construction.txt`, `P1_batch_note_construction.txt`, `P4_batch_note_extraction.txt` |
| FastASEM: `fact` = một câu, ≤ 45 từ | `asem/fast_ingest.py` |
| Prompt trả lời mô tả định dạng block + `selected_ids` theo số | `data/prompts/P_distil.txt`, `P_temporal_qa.txt` |

Đồng thời sửa một lỗi merge trong `P1_note_construction.txt` (bullet `description` và `Step 4` bị lặp).

### 9.4 Phạm vi ảnh hưởng

- **Định dạng render áp dụng ngay** cho bank cũ (`ds_fixed`) — không cần ingest lại.
- **Caps ở tầng lưu chỉ có hiệu lực cho bank mới ingest**; bank hiện có vẫn giữ `description` dài,
  nhưng renderer đã cắt còn 420 ký tự/note.
- Điểm accuracy của run Qwen3-4B **chưa** phản ánh thay đổi này: cần chạy lại Phase B (§7, việc 11).
- Lưu ý: định dạng số chỉ ảnh hưởng đường distil của ASEM v1 (FastASEM dùng `direct_answer`, không có
  `selected_ids`); trước đây một UUID bị copy sai làm `selected_notes` rỗng và code âm thầm quay về
  "dùng tất cả candidates", tức mất tín hiệu chọn lọc — nay không còn khả năng đó.
- Test: **169 passed** (1 deselected là test Ollama cần server, đã lỗi trước đó).

---

## 10. Phụ lục

### A. Lệnh tái lập phân tích (LLM-free, env `memory-r1`)

```powershell
conda activate memory-r1
python scratch_diag/qwen3_result_audit.py              # -> scratch_diag/_qwen3_audit.txt
python scratch_diag/fullcontext_evidence_survival.py   # % evidence còn sống trong prompt FullContext
python scratch_diag/retrieval_recall_qwen.py           # tách INGESTION / RETRIEVAL / GENERATION
```

### B. Script và số liệu liên quan

| Script | Việc |
|---|---|
| `scratch_diag/qwen3_result_audit.py` | Nhãn category + evidence-turn; độ dài/abstention/judge theo hệ thống và nhóm; McNemar + bootstrap; hiệu chỉnh judge; chế độ lỗi `judge_error`; near-miss theo dải F1 |
| `scratch_diag/fullcontext_evidence_survival.py` | Replay `FullContext.answer` bằng spy backend → evidence có còn trong prompt không, vị trí của nó |
| `scratch_diag/retrieval_recall_qwen.py` | Replay retriever thật trên bank `ds_fixed` → tách 3 tầng lỗi theo lát cắt |
| `scratch_diag/idk_decomposition.py` | Bản cũ (không dùng judge) cho tập IDK của run deepseek |
| `scratch_diag/measure_context.py`, `context_size.py`, `field_budget.py` | Đo token prompt theo từng field/hệ thống |

### C. Chế độ lỗi judge (`judge_error`) theo hệ thống

| System | contradiction | missing_info | hallucination | vague |
|---|---|---|---|---|
| ASEM | 356 (52%) | 297 (43%) | 34 (5%) | 0 |
| FastASEM | 630 (62%) | 329 (32%) | 65 (6%) | 0 |
| FullContext | 443 (71%) | 154 (25%) | 21 (3%) | 7 (1%) |
| NoMemory | 1020 (63%) | 489 (30%) | 118 (7%) | 0 |

Theo nhóm (tỉ lệ trong các câu judge nói sai):

| Loại | ASEM | FastASEM | FullContext |
|---|---|---|---|
| adversarial | contradiction 59% / missing 35% | contradiction 78% / missing 14% | contradiction 91% / missing 7% |
| temporal | contradiction 83% / missing 16% | contradiction 86% / missing 13% | contradiction 74% / missing 21% |
| open_domain | contradiction 55% / missing 43% | contradiction 56% / missing 36% | contradiction 64% / missing 31% |
| multi_hop | missing 70% / contradiction 23% | missing 64% / contradiction 30% | missing 64% / contradiction 28% |
| single_hop | missing 49% / contradiction 45% | missing 49% / contradiction 44% | missing 52% / contradiction 42% |

Near-miss (chỉ các câu judge nói sai), theo dải token-F1:

| System | F1 < 0.2 | 0.2–0.5 | > 0.5 |
|---|---|---|---|
| ASEM | 538 | 90 | 11 |
| FastASEM | 817 | 121 | 55 |
| FullContext | 435 | 125 | 28 |
| NoMemory | 1523 | 90 | 7 |

ASEM có ít "sai nhưng gần đúng" hơn FullContext ⇒ lỗi của ASEM là **lệch thông tin**, không phải
diễn đạt.

### D. Ghi chú kỹ thuật đã xác minh trong code

- `HybridRetriever.__post_init__` không có; các default thực tế khi builder không truyền:
  `max_link_hops=1`, `link_traversal_topn=3`, `rrf_k=60`, `dense/bm25/entity/temporal = 1.0/0.8/0.6/0.5`.
- `_RELATION_BOOST = {contradicts 1.35, causal 1.25, temporal 1.20, extends 1.10, same-topic 1.00,
  semantic 1.00, linked 1.00}`; nhãn cạnh phổ biến trong bank `ds_nothink`: same-topic 12 235,
  extends 6 635, semantic 2 386, causal 284, temporal 196, contradicts 60.
- `AnswerAgent.is_abstention` (dùng trong báo cáo này): từ chối ở đầu câu, hoặc câu ngắn (≤ 240 ký tự)
  nói rằng note không chứa thông tin. Đã kiểm chứng trên 1790 dự đoán của run `ds_nothink` (deepseek):
  421 dòng bị gắn cờ (23.5%), 0 bỏ sót. Trên run Qwen này số câu bị gắn cờ là: ASEM 33,
  FastASEM 227, FullContext 59, NoMemory 944.
- `AnswerAgent._fit` cắt từ **đuôi** danh sách ứng viên; `ASEMPipeline._recover` sắp xếp lại pool
  theo cosine với query trước khi gọi, nếu không sẽ cắt mất chính các note mà pass mở rộng tìm được.
- ASEM v1 dùng `P_distil.txt` (không phải `P4`/`direct_answer` — đó là đường ASEM v2 **chưa được đánh giá**).
- `strip_query_prefix` áp dụng cho ASEM/FastASEM (bỏ tiền tố `"Conversation between X and Y. Question:"`);
  baseline FullContext/NoMemory vẫn nhận query có tiền tố.
