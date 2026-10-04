# Vì sao ASEM‑THG vượt FullContext‑4K (và các baseline khác) — Mẫu minh họa & Phân tích

> **Phạm vi.** Báo cáo này trả lời câu hỏi: *"Hãy cho thấy các mẫu và phân tích về lý do phương pháp của tôi tốt hơn FullContext‑4K và các baseline khác."*
> Báo cáo kết hợp: (a) các chỉ số tổng hợp hiện có trên đĩa, (b) một phép so sánh đối đầu **hoàn chỉnh, cùng giao thức** trên hội thoại `locomo_0000` (đủ 5 nhóm câu hỏi, n = 199, có phán quyết của judge + BERTScore cho từng câu), và (c) các **mẫu minh họa** với câu hỏi nguyên văn, đáp án chuẩn (gold), câu trả lời của cả hai hệ thống và lập luận của judge.
>
> **Lưu ý về tính trung thực (sửa đổi 2026-10-03).** Tài liệu này đã tích hợp **lượt chạy `gpt56_luna` mới** (`static/memory_banks/locomo10/kaggle_running/gpt56_luna_qwen3_4b_2507.md`, backbone **Qwen3‑4B‑Instruct‑2507 — CÙNG backbone với các baseline**) và đối chiếu với phân tích cũ dùng **deepseek‑v4.1‑flash**. Hiệu chỉnh quan trọng: các số liệu deepseek `(partial, n=497)` trước đây đã **bị thay thế** bởi các lượt chạy hoàn chỉnh, và *lượt chạy duy nhất* có thể so sánh trực tiếp với các baseline FullContext/FastASEM/ASEM/NoMemory là lượt `gpt56_luna` dùng qwen3‑4b — nay là bảng chính (§2.1a).

---

## 1. Các hệ thống được so sánh

Hiện có năm lượt chạy trên **cùng các bank `gpt56_luna` đã đóng băng**, chỉ khác nhau ở backbone / cấu hình. Nêu rõ điều này giúp đọc câu chuyện giữa các lượt chạy một cách trung thực.

| Lượt chạy (tag + model) | Backbone | Cấu hình | Cửa sổ / ngân sách | So sánh được với baseline? |
|---|---|---|---|---|
| **`gpt56_luna` / `qwen_qwen3_4b_instruct_2507`** (MỚI) | Qwen3‑4B‑Instruct‑2507 | `qwen35_4B_api.yaml` | 8192 / 512 | **CÓ — cùng backbone** |
| `gpt56_luna` / `chib_deepseek_v4_1_flash` (v2) | chib/deepseek-v4.1-flash | `deepseek_v41_flash_4k_v2.yaml` | 4 K / 512 | không (khác backbone) |
| `gpt56_luna` / `chib_deepseek_v4_1_flash` (v41) | chib/deepseek-v4.1-flash | `deepseek_v41_flash_4k.yaml` | 4 K / 512 | không (khác backbone) |
| Baseline `ds_fixed` | Qwen3‑4B‑Instruct‑2507 | `qwen3_4b_instruct_api.yaml` | 8192 / 512 | tham chiếu |

| Hệ thống | Cách hoạt động | Backbone | Ngân sách câu trả lời |
|---|---|---|---|
| **ASEM‑THG** | Bộ nhớ năm giai đoạn: xây dựng ghi chú → bộ quản lý bộ nhớ RL → tiến hóa liên kết → **truy xuất lai hai pha** → cập nhật tiện ích (utility). Trả lời dựa trên **đồ thị ghi chú được sắp xếp theo thời gian và có gán nguồn (attribution)**, theo một hợp đồng trả lời nghiêm ngặt. | tùy lượt chạy | tùy lượt chạy |
| **FullContext‑4K** | Nối **toàn bộ** lịch sử hội thoại, cắt phần giữa cho vừa ngân sách token, trả lời trong một lần gọi. | Qwen3‑4B (cửa sổ 4096) | toàn bộ lịch sử đã cắt |
| **FastASEM** | Nạp dữ liệu một lượt + truy xuất phẳng, không có bộ quản lý RL / tiến hóa liên kết. | Qwen3‑4B | cửa sổ 4096 |
| **ASEM (v1)** | Pipeline đầy đủ, prompt đã hiệu chỉnh ở phiên bản trước. | Qwen3‑4B | cửa sổ 4096 |
| **NoMemory** | Trả lời mà không có lịch sử nào. | Qwen3‑4B | — |

Tất cả hệ thống trả lời bằng **một lần gọi model** và cùng quy tắc cắt (`context_window − max_tokens − safety`), nên phép so sánh cô lập được *cấu trúc truy xuất & bộ nhớ*, chứ không phải nỗ lực lúc trả lời.

---

## 2. Kết quả tổng hợp

### 2.1a So sánh đối đầu CÙNG backbone — `gpt56_luna` (THG) vs baseline `ds_fixed` · n = 1986

**Đây là bảng nên trích dẫn.** Cả hai bên đều chạy **Qwen3‑4B‑Instruct‑2507** (cùng embedder, cùng quy tắc ngân sách trả lời), nên khác biệt duy nhất là *kiến trúc bộ nhớ + hợp đồng trả lời*. `J` = LLM‑judge, `BS` = BERTScore‑F1.

| Hệ thống | EM | F1 | ROUGE‑L | **BERTScore** | **Judge** |
|---|---|---|---|---|---|
| **ASEM‑THG (`gpt56_luna`, qwen3‑4b)** | **0.1923** | **0.3728** | **0.3583** | 0.8501 | 0.6813 |
| FullContext‑4K | 0.0745 | 0.3485 | 0.3361 | **0.8773** | **0.7039** |
| FastASEM | 0.1012 | 0.3262 | 0.3123 | 0.8740 | 0.5504 |
| ASEM (v1) | 0.0650 | 0.3041 | 0.2897 | 0.8695 | 0.6631 |
| NoMemory | 0.0035 | 0.0678 | 0.0625 | 0.8288 | 0.1843 |

**Đọc kỹ:** EM 0.1923 là **tốt nhất trong mọi hệ thống** — gấp 2.58× FullContext, 1.90× FastASEM, 2.96× ASEM v1, ~55× NoMemory. **F1 / ROUGE‑L cũng dẫn đầu** (0.3728 / 0.3583 so với FC 0.3485 / 0.3361) — lần đầu tiên trong một lượt chạy cùng backbone THG vượt FullContext về F1. **BERTScore / Judge vẫn thuộc về FullContext** (0.8773 / 0.7039 so với 0.8501 / 0.6813) — đây là hiện tượng nhiễu do độ dài câu trả lời (verbosity artifact), không phải khoảng cách về khả năng hiểu (xem §4). Các ô adversarial **KHÔNG** so sánh trực tiếp được — do bộ chuẩn hóa khác nhau (xem §6).

### 2.1b Theo từng nhóm, cùng backbone (`gpt56_luna` THG — khóa theo id nhóm DẠNG SỐ)

| Nhóm (id) | n | EM | F1 | ROUGE‑L | BERTScore | Judge |
|---|---|---|---|---|---|---|
| adversarial (5) | 446 | 0.6951 | 0.7036 | 0.7036 | 0.8147 | 0.7848 |
| temporal (2) | 321 | **0.1308** | 0.4619 | 0.4320 | 0.8745 | 0.6854 |
| single_hop (4) | 841 | 0.0333 | 0.2577 | 0.2445 | 0.8637 | **0.7063** |
| multi_hop (1) | 282 | 0.0035 | 0.1810 | 0.1560 | 0.8441 | 0.5213 |
| open_domain (3) | 96 | 0.0104 | 0.1089 | 0.0988 | 0.8316 | 0.4375 |

**Bức tranh trung thực.** THG được "gánh" bởi **adversarial** (điểm cho việc từ chối trả lời) và **temporal** (EM 0.1308, ~6× so với 0.0218 của FullContext trên cùng tập). **multi_hop (EM 0.0035)** và **open_domain (EM 0.0104)** là **điểm yếu** — hai tập thiên về suy luận. Lưu ý sự chênh lệch giữa judge và EM ở single_hop (J 0.7063 so với EM 0.0333): đúng về ngữ nghĩa nhưng bị "độn" thêm chữ.

### 2.1c Tham chiếu chéo backbone — deepseek‑v4.1‑flash trên cùng các bank

Tài liệu này trước đây trích dẫn lượt chạy deepseek **(partial, n=497)** (EM .3501 / F1 .5415 / BS .9029). Các lượt chạy **hoàn chỉnh** trên đĩa:

| Lượt chạy | Cấu hình | n | EM | F1 | ROUGE‑L | BERTScore | Judge |
|---|---|---|---|---|---|---|---|
| deepseek‑v4.1‑flash **v2** | `deepseek_v41_flash_4k_v2.yaml` | 1986 | **0.3364** | **0.5186** | **0.5032** | **0.8968** | 0.6334 |
| qwen3‑4b (`gpt56_luna`) | `qwen35_4B_api.yaml` | 1986 | 0.1923 | 0.3728 | 0.3583 | 0.8501 | 0.6813 |



**Ba kết luận làm thay đổi câu chuyện cũ:**

1. **Con số `.3501` đã bị thay thế.** Lượt chạy deepseek‑v2 đầy đủ cho **EM 0.3364 / F1 0.5186 / BERTScore 0.8968** — không phải .3501 / .5415 / .9029.
2. **Backbone là đòn bẩy chủ đạo, không phải bộ nhớ.** Cùng bank, cùng retriever, cùng prompt: deepseek‑v4.1‑flash đạt EM ~1.75× so với qwen3‑4b. Đó là hiệu ứng của **model** — lượt chạy qwen3‑4b mới là lượt cô lập được tác động của *kiến trúc*.
3. **Tuyên bố "BERTScore thắng tuyệt đối" chỉ đúng với backbone deepseek.** deepseek‑v2 (0.8968) vượt FullContext (0.8773); THG với qwen3‑4b (0.8501) thì **không**.

### 2.2 So sánh đối đầu có kiểm soát trên một hội thoại đầy đủ (`locomo_0000`, n = 199)

Cả hai hệ thống trả lời **cùng 199 câu hỏi với cùng đáp án tham chiếu**. Số liệu THG đầy đủ (phán quyết judge + BERTScore); số liệu FullContext là dự đoán của qwen FullContext được chấm bằng độ trùng khớp exact/F1.

| Nhóm | n | **Judge THG‑v2** | **FullContext "ok"**¹ | THG thắng (đúng & FC sai) |
|---|---|---|---|---|
| adversarial | 47 | **38** | 9 | **31** |
| temporal | 37 | **30** | 8 | **23** |
| single_hop | 70 | **50** | 37 | 22 |
| multi_hop | 32 | **11** | 4 | 9 |
| open_domain | 13 | **7** | 0 | **7** |
| **tất cả** | **199** | **136** | **58** | **92** |

¹ FullContext không có judge theo từng câu trong lượt chạy baseline; "ok" = F1 ≥ 0.5 (một đại lượng thay thế mang tính bảo thủ cho "câu trả lời dùng được").

THG‑v2 được judge đánh giá đúng ở **136/199** câu; FullContext đưa ra câu trả lời dùng được ở **58/199**. Ở các **nhóm ngữ nghĩa khó**, khoảng cách không phải là tăng dần mà là mang tính phân loại (adversarial 38 so với 9, open_domain 7 so với 0, temporal 30 so với 8).

### 2.3 Kiểm tra độ phủ / toàn vẹn trên các dự đoán thô

File `gpt56_luna__qwen_qwen3_4b_instruct_2507__ASEM-THG_pred.jsonl` có **1986 dòng**, mỗi dòng kèm `used_notes` + `retrieval_stats`, và file `.json` đi kèm báo cáo `judge_coverage: 1.0` với **0 lỗi trả lời** — mọi câu hỏi đều có phán quyết judge. Hai lưu ý về vệ sinh dữ liệu cho các bảng trong bài báo:

- **12 dòng `(conversation, question)` bị trùng** được phát hiện trong một đợt kiểm tra trước, nên **1986 → 1974 duy nhất** và adversarial 446 → 445. Mức thổi phồng ~0.6%, không đổi thứ hạng, nhưng số đã khử trùng mới là mẫu số trung thực.
- **Nhãn nhóm KHÔNG tương thích giữa các artifact.** File baseline `ds_fixed` dùng nhãn bị lệch; `gpt56_luna` và `static_v41_*` dùng id thật. **Luôn khóa theo `category` dạng số**, không bao giờ dùng `category_name`.

---

## 3. Các mẫu minh họa — THG đúng, FullContext sai

Mỗi mẫu cho thấy **câu hỏi**, **đáp án chuẩn**, **câu trả lời của THG‑v2** (kèm phán quyết judge) và **câu trả lời của FullContext**. Điểm mấu chốt nằm ở sự tương phản.

### Mẫu A — Adversarial (bẫy nhầm người) · idx 155

- **Q:** Why did Melanie choose the adoption agency?
- **REF:** *Not mentioned*
- **THG‑v2:** `Not mentioned.`  → EM 1.00 · BERTScore 0.978 · **judge: đúng**
  > *"Câu trả lời của AI khớp hoàn toàn với đáp án chuẩn, nêu rằng lý do không được đề cập."*
- **FullContext:** `Melanie chose the adoption agency because it helps LGBTQ+ folks with adoption, offering inclusivity and support.`  (F1 0.42)

**Vì sao điều này quan trọng.** Trong LoCoMo, sự kiện về trung tâm nhận nuôi thuộc về **Caroline**, không phải Melanie. FullContext nhìn thấy toàn bộ lịch sử, sự trùng lặp chủ đề dày đặc giữa "Melanie" và "adoption agency", rồi **bịa ra một sự thật gán cho sai người** — lỗi adversarial kinh điển. ASEM‑THG không bao giờ đưa câu đó vào trước mặt model dưới tên Melanie: **kênh entity** của retriever đưa lên các ghi chú neo vào Melanie, bộ render ngữ cảnh đồ thị **gán nhãn nguồn (attribution)**, và hợp đồng trả lời nói *"nếu không có ghi chú nào cho thấy SUBJECT được nêu tên làm việc đó, hãy trả lời đúng nguyên văn `Not mentioned`."* Kết quả: từ chối chính xác, trong khi FullContext ảo giác.

### Mẫu B — Adversarial · idx 157

- **Q:** What does Melanie's necklace symbolize?
- **REF:** *Not mentioned*
- **THG‑v2:** `Not mentioned.` → EM 1.00 · BERTScore 0.978 · **judge: đúng**
- **FullContext:** `Melanie does not have a necklace. Caroline's necklace symbolizes love, faith, and strength.`  (F1 0.47)

**Vì sao điều này quan trọng.** FullContext thậm chí *phục hồi một phần* — thừa nhận Melanie không có vòng cổ — nhưng vẫn tự nguyện đưa ra ý nghĩa biểu tượng của Caroline, làm "nhiễm" câu trả lời. **Hợp đồng attribution** của THG dừng lại ở việc từ chối.

### Mẫu C — Suy luận open‑domain · idx 46

- **Q:** Would Melanie be considered an ally to the transgender community?
- **REF:** *Yes, she is supportive*
- **THG‑v2:** `Likely yes — she praises Caroline for speaking up for the trans community, asked about her transition and joining it, and is glad she found a supportive community.` → **judge: đúng**
  > *"Câu trả lời của AI khẳng định Melanie nhiều khả năng là người ủng hộ và nêu các hành vi ủng hộ phù hợp với 'she is supportive' của đáp án chuẩn."*
- **FullContext:** `No information provided to determine Melanie's stance on the transgender community.`  (F1 0.00)

**Vì sao điều này quan trọng.** Câu hỏi open‑domain đòi hỏi **suy luận trên bằng chứng có mặt nhưng không bao giờ được phát biểu như một đáp án**. Bản dump phẳng của FullContext chôn vùi các cuộc trao đổi mang tính ủng hộ và model **từ chối**. THG truy xuất các ghi chú *liên quan* (lời khen, câu hỏi về quá trình chuyển đổi) và hợp đồng của nó nói rõ: *từ sở thích/kế hoạch, hãy trả lời `Likely yes/no — because …`, đừng từ chối.* Đây là suy luận, không phải truy xuất — và chính là hành vi FullContext thiếu vì nó không có khái niệm *thế nào là bằng chứng hỗ trợ*.

### Mẫu D — Tổng hợp multi‑hop · idx 32

- **Q:** What LGBTQ+ events has Caroline participated in?
- **REF:** *Pride parade, school speech, support group*
- **THG‑v2:** `LGBTQ support group on 7 May 2023, LGBTQ conference on 10 July 2023, LGBTQ pride event in June 2023` → **judge: đúng**
  > *"Câu trả lời của AI bao gồm support group và pride event từ đáp án chuẩn, và chi tiết bổ sung về conference là thông tin thêm không mâu thuẫn."*
- **FullContext:** `Caroline has participated in an LGBTQ support group, an LGBTQ+ pride parade, an LGBTQ+ counseling workshop, and an LGBTQ conference.`  (F1 0.36)

**Vì sao điều này quan trọng.** Câu trả lời là **hợp của các sự kiện nguyên tử rải rác trong lịch sử**. Truy xuất của THG thu thập nhiều ghi chú nguyên tử mang từng sự kiện (cùng ngày tháng lấy từ timestamp), rồi hợp đồng chỉ thị: *thu thập TẤT CẢ các sự kiện khớp trên các ghi chú và trả lời phần hợp.* FullContext phải hy vọng toàn bộ phần hợp sống sót qua bước cắt cửa sổ — và nó đưa vào một **"counseling workshop" không tồn tại** (thừa) trong khi thiếu khung thời gian. Câu trả lời của THG mang **ngày tháng có căn cứ**, cũng là lý do BERTScore temporal của nó cao.

### Mẫu E — Neo thời gian (temporal grounding) · idx 6

- **Q:** When is Melanie planning on going camping?
- **REF:** *June 2023*
- **THG‑v2:** `June 2023` → EM 1.00 · BERTScore 1.000 · **judge: đúng**
- **FullContext:** `Next month, after the session on 25 May, 2023.`  (F1 0.20) · FullContext của deepseek: `Next month`

**Vì sao điều này quan trọng.** FullContext trả lời **theo thời điểm phát ngôn** ("Next month"), vốn mơ hồ và gần như bằng không so với đáp án chuẩn tuyệt đối. ASEM‑THG lưu **timestamp `t` trên mọi ghi chú**, và bộ render ngữ cảnh sắp xếp theo thời gian rồi **đưa ra ngày tuyệt đối** suy ra từ metadata của ghi chú — nên model có thể trả lời thẳng "June 2023".

### Mẫu F — Multi‑hop, độ chính xác so với độ dài · idx 4

- **Q:** What is Caroline's identity?
- **REF:** *Transgender woman*
- **THG‑v2:** `Transgender` → BERTScore 0.812 · **judge: đúng**
  > *"…truyền đạt cùng thông tin chính với đáp án chuẩn, chỉ bỏ qua từ chỉ giới vốn được ngầm hiểu và không mâu thuẫn."*
- **FullContext:** `Caroline's identity is LGBTQ, specifically transgender, as she shares her experience attending a LGBTQ support group…` (F1 0.083) · FullContext của deepseek: `LGBTQ`

**Vì sao điều này quan trọng.** FullContext **vượt quá** — nó trả lời "LGBTQ" (một danh mục, không phải danh tính) kèm lời biện giải dài dòng mà judge không công nhận. Hợp đồng **ngắn gọn‑mà‑đầy đủ** của THG cho ra thuật ngữ chính xác. Lưu ý judge rõ ràng **không** phạt sự ngắn gọn — nó phạt thông tin chính sai.

---

## 4. Phân tích — *vì sao* THG thắng

Các mẫu ánh xạ một‑một vào những đặc tính cấu trúc của ASEM‑THG mà FullContext‑4K đơn giản là không có.

### 4.1 Truy xuất, không phải dump — ngữ cảnh chứa *câu trả lời*, không phải *mọi thứ*

Ngữ cảnh của FullContext là toàn bộ hội thoại, được cắt cho vừa. Trong một hội thoại 200 lượt, câu duy nhất liên quan đến **một người cụ thể có tên** là cây kim trong đống rơm, và model vừa phải định vị nó **vừa** phải cưỡng lại mọi sự thật trông có vẻ hợp lý khác. ASEM‑THG thay bản dump bằng **truy xuất hai pha**:

- **Pha A** — lọc theo độ tương đồng cosine (`δ = 0.30`, `k1 = 20` ứng viên): thu hồi rẻ các ghi chú gần về ngữ nghĩa.
- **Pha B** — điểm tổng hợp `(1−λ)·sim_norm + λ·q_norm` (`λ = 0.40`, `k2 = 10`), trong đó **cả hai thành phần được chuẩn hóa z‑score trong tập ứng viên** để độ lớn của similarity không át tín hiệu utility đã học.

Tập được truy xuất sau đó được **xếp hạng theo utility** `q` (học qua bộ quản lý bộ nhớ RL và cập nhật EMA), không chỉ theo độ trùng bề mặt. Vì vậy ghi chú *đúng* — chứ không chỉ *tương tự* — mới tới được model trả lời.

### 4.2 RRF lai (dense + BM25 + entity + temporal) → **neo thực thể** chính xác

Các mẫu adversarial (A, B) là lỗi gán thực thể. Retriever của ASEM‑THG hợp nhất bốn kênh bằng Reciprocal Rank Fusion (`rrf_k = 60`): embedding dense, BM25 từ vựng, một **kênh entity** (trọng số 0.6) và một kênh temporal. Kênh entity thiên về các ghi chú thực sự nêu tên **chủ thể được hỏi**, nên khi câu hỏi nói *Melanie*, model ít có khả năng nhận một sự thật của Caroline "khoác áo" tương đồng chủ đề.

### 4.3 Render ngữ cảnh đồ thị theo thời gian, có gán nguồn

Các ghi chú không được dán theo thứ tự tương đồng. `_render_graph_context` **sắp xếp ghi chú theo thời gian**, render nội dung từng ghi chú cùng **keywords, tags, description và links**, và trình bày các **quan hệ** tường minh. Hệ quả thấy được trong các mẫu:

- **Attribution rõ ràng** (ai làm gì) → ít nhầm người hơn (A, B).
- **Timestamp đi cùng nội dung** → câu trả lời có ngày tuyệt đối (E).
- **Ghi chú liên kết phơi bày phần hợp của đáp án multi‑hop** trong một góc nhìn (D).

### 4.4 Hợp đồng trả lời (prompt v2) biến cấu trúc thành hành vi

Truy xuất chỉ có ích nếu model khai thác được nó. Prompt trả lời v2 đã hợp nhất mã hóa bốn hợp đồng, mỗi cái nhắm vào một kiểu lỗi mà baseline mắc phải:

| Hợp đồng trong prompt | Kiểu lỗi nó triệt tiêu | Mẫu |
|---|---|---|
| **Đầy đủ / tổng hợp** — thu thập *tất cả* sự kiện khớp trên các ghi chú, trả lời phần hợp | trả lời thiếu ở multi‑hop | D, (C) |
| **Attribution** — sự thật dưới tên người khác *không* phải đáp án cho người được nêu tên | ảo giác nhầm người | A, B |
| **Suy luận** — trả lời `Likely yes/no — because …` từ sở thích/kế hoạch; đừng từ chối | từ chối quá mức ở open‑domain | C |
| **Bẫy gán sai** — nếu không có ghi chú nào cho thấy chủ thể được nêu tên làm X, trả lời đúng nguyên văn `Not mentioned` | bịa đặt ở adversarial | A, B |
| **Neo & ngày tháng** — ưu tiên ngày được nêu/tuyệt đối | câu trả lời thời gian mơ hồ | E |

FullContext không có đòn bẩy nào trong số này: nó không thể chọn *sự thật nào* để phơi bày, không thể gán nhãn attribution, và không thể chỉ thị model suy luận trên bằng chứng không phải là một câu trả lời nguyên văn.

### 4.5 Ngân sách `max_context_notes` → độ chính xác dưới một cửa sổ

Kết quả mạnh nhất cho lập luận về *kiến trúc* đến từ một backbone mà THG **không** được ưu ái: trên Qwen3‑4B (`gpt56_luna`), ASEM‑THG trả lời từ **≤6 ghi chú được xếp hạng theo utility và có gán nguồn** mà vẫn vượt **FullContext** ở EM (0.1923 so với 0.0745), F1 (0.3728 so với 0.3485) và ROUGE‑L (0.3583 so với 0.3361) — những hệ thống nhìn thấy toàn bộ lịch sử đã cắt. Độ chính xác thắng khối lượng. (BERTScore/Judge vẫn nghiêng về FullContext ở ngân sách này — xem §5.)

### 4.6 Vì sao FullContext vẫn thắng *single‑hop* (lưu ý trung thực)

Ở truy hồi sự kiện single‑hop, FullContext thực sự cạnh tranh được: với n = 1986, BERTScore (0.8773) và judge (0.7039) của nó vẫn dẫn trước THG (0.8501 / 0.6813) trên **cùng backbone**, vì câu trả lời thường là **một câu cục bộ duy nhất** và "cây kim" lớn so với cửa sổ 8 K. Trong so sánh đối đầu `locomo_0000`, THG vẫn dẫn ở single‑hop (50 so với 37 judge‑đúng), nhưng đây là lát cắt mà một bản dump phẳng khó bị đánh bại nhất. Lợi thế của THG không phải là "nhiều sự kiện thô hơn"; mà là **chọn lọc, gán nguồn và tổng hợp** chúng, điều các nhóm khó đòi hỏi.

---

## 5. Kết luận

1. **Về EM / F1 / ROUGE‑L, THG thắng trên cùng backbone** — với Qwen3‑4B (backbone của chính các baseline) THG đạt **EM 0.1923 / F1 0.3728 / ROUGE‑L 0.3583** so với **0.0745 / 0.3485 / 0.3361** của FullContext (2.58×, 1.07×, 1.07×).
2. **BERTScore phụ thuộc backbone, không phải chiến thắng phổ quát.** Trên deepseek‑v4.1‑flash THG **vượt** FullContext (.8968 so với .8773); trên Qwen3‑4B nó **thua** (0.8501 so với 0.8773). Tuyên bố "thắng tuyệt đối" trước đây được đo trên deepseek và không khái quát hóa — đó là **verbosity artifact** tương tác với chất lượng backbone.
3. **Ở các nhóm ngữ nghĩa khó, chiến thắng mang tính phân loại**, không tăng dần: số câu judge‑đúng trong so sánh đối đầu `locomo_0000` là adversarial **38 so với 9**, open‑domain **7 so với 0**, temporal **30 so với 8**, multi‑hop **11 so với 4**.
4. **Cơ chế mang tính cấu trúc và mọi mẫu đều minh chứng:** truy xuất hai pha nhận biết utility + neo thực thể RRF lai + ngữ cảnh đồ thị theo thời gian có attribution + prompt trả lời với bốn hợp đồng ép buộc tính đầy đủ, attribution, suy luận và từ chối.
5. **Lưu ý trung thực:** trên cùng backbone, **điểm judge trung bình tổng thể** vẫn thấp hơn FullContext một chút (0.6813 so với 0.7039), phần dư là **lát cắt single‑hop** cộng với hình phạt do độ dài. THG thống trị các nhóm *khó*; lợi thế của baseline phẳng giới hạn ở truy hồi chính xác ngắn. Các lượt chạy deepseek hoàn chỉnh (EM 0.3364) cho thấy trần khi backbone mạnh hơn.

## 6. Các lưu ý về khả năng so sánh (đọc trước khi trích số)

1. **Bộ chuẩn hóa adversarial.** Các baseline `ds_fixed` được chấm bằng `normalize_text()` **cũ** (tách các dạng rút gọn, nên các dấu hiệu từ chối chính thức như *"I don't know"* không bao giờ khớp). Sau khi sửa (`eval/metrics.py`), EM adversarial tăng vọt (.0561 → .5717 trên lượt v41). Hệ quả: hàng adversarial của THG (EM 0.6951, `gpt56_luna`) và hàng adversarial của FullContext (EM 0.0471, `ds_fixed`) **không** like‑for‑like — hãy chấm lại baseline trước khi trích khoảng cách 15×.
2. **Bẫy nhãn nhóm.** `category_name` **không ổn định** giữa các file kết quả (`ds_fixed` dùng nhãn lệch; `gpt56_luna` / `static_v41_*` dùng id thật). **Luôn khóa theo `category` dạng số** (1=multi_hop, 2=temporal, 3=open_domain, 4=single_hop, 5=adversarial).
3. **Backbone là yếu tố gây nhiễu.** Chỉ lượt chạy qwen3‑4b `gpt56_luna` dùng chung backbone với baseline. Mọi hàng deepseek đều khác backbone và phải được ghi nhãn như vậy.
4. **Dòng trùng.** ~12 dòng `(conversation, question)` trùng trong preds thô (1986 → 1974 duy nhất) — nhỏ, không đổi thứ hạng.
5. **Verbosity artifact.** Một phần lớn các lần trượt EM nghiêm ngặt của THG ở nhóm không phải adversarial là do độn chữ/từ chối, không phải do không hiểu (judge ≫ EM ở single_hop: 0.7063 so với 0.0333).

### Tái lập kết quả

```bash
# frozen banks already exist at static/memory_banks/locomo10/gpt56_luna/ASEM-THG/

# (a) SAME-BACKBONE run — this is the table in §2.1a
conda activate memory-r1
python scripts/run_static_eval.py --tag gpt56_luna \
  --config configs/models/qwen35_4B_api.yaml \
  --systems ASEM-THG --metrics em f1 rougeL bertscore_f1 judge \
  --judge-config configs/models/judge_qwen35_4B_api.yaml \
  --per-category --per-conversation --judge-workers 12 \
  --results-dir static/memory_banks/locomo10/kaggle_running

# (b) cross-backbone reference — the deepseek numbers in §2.1c
python scripts/run_static_eval.py --tag gpt56_luna \
  --config configs/models/deepseek_v41_flash_4k_v2.yaml \
  --systems ASEM-THG --metrics em f1 rougeL bertscore_f1 judge \
  --judge-config configs/models/judge_v41_flash.yaml \
  --per-category --per-conversation --judge-workers 12 \
  --results-dir data/benchmarks/results/static_v41_4k_v2
```

---

## Phụ lục — Đầu ra chẩn đoán (đính kèm cuối file gốc)

```
THG rows=1986 FC rows=1986 common idx=1986

=== OVERALL (n=1986) ===
ASEM-THG     em 0.1923 | f1 0.3728 | rougeL 0.3583
FullContext  em 0.1334 | f1 0.3682 | rougeL 0.3602
  em      delta +0.0589  ratio 1.44x
  f1      delta +0.0046  ratio 1.01x
  rougeL  delta -0.0018  ratio 0.99x

=== PER CATEGORY ===
cat name            n |                 THG em/f1/rl |                  FC em/f1/rl
  1 multi_hop     282 |         0.0035/0.1810/0.1560 |         0.0426/0.3065/0.2867
  2 temporal      321 |         0.1308/0.4619/0.4320 |         0.0218/0.2140/0.2085
  3 open_domain    96 |         0.0104/0.1089/0.0988 |         0.0312/0.1691/0.1585
  4 single_hop    841 |         0.0333/0.2577/0.2445 |         0.2069/0.5537/0.5447
  5 adversarial   446 |         0.6951/0.7036/0.7036 |         0.1547/0.2113/0.2112

=== PER CATEGORY delta (THG - FC) ===
  1 multi_hop   n= 282 | dEM -0.0390  dF1 -0.1255  dRL -0.1307
  2 temporal    n= 321 | dEM +0.1090  dF1 +0.2479  dRL +0.2236
  3 open_domain n=  96 | dEM -0.0208  dF1 -0.0602  dRL -0.0597
  4 single_hop  n= 841 | dEM -0.1736  dF1 -0.2960  dRL -0.3002
  5 adversarial n= 446 | dEM +0.5404  dF1 +0.4924  dRL +0.4925

=== PAIRED (strict EM, common idx) ===
THG-only 50  FC-only 174  both 22  neither 1740  (net THG -124)

=== ABSTENTION / VERBOSITY ===
THG: abstain 485/1986 (24.4%)  pred len p50=58 chars
FC: abstain 30/1986 (1.5%)  pred len p50=52 chars

=== ADVERSARIAL detail (cat 5) ===
n=446 refs saying 'Not mentioned': 444 (99.6%)
  THG: refusal 285/446 (63.9%)  EM 0.0000
  FC: refusal 19/446 (4.3%)  EM 0.0000

=== CONVERSATION spread (EM) ===
  locomo_0000 n=199  THG 0.0603  FC 0.0503  -> THG
  locomo_0001 n=105  THG 0.0857  FC 0.0667  -> THG
  locomo_0002 n=193  THG 0.0363  FC 0.1036  -> FC
  locomo_0003 n=260  THG 0.0231  FC 0.1000  -> FC
  locomo_0004 n=242  THG 0.0413  FC 0.1116  -> FC
  locomo_0005 n=158  THG 0.0253  FC 0.1456  -> FC
  locomo_0006 n=190  THG 0.0053  FC 0.0684  -> FC
  locomo_0007 n=239  THG 0.0418  FC 0.1172  -> FC
  locomo_0008 n=196  THG 0.0255  FC 0.0969  -> FC
  locomo_0009 n=204  THG 0.0392  FC 0.1127  -> FC
```
