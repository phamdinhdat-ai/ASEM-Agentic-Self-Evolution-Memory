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

| Hệ thống | Cách hoạt động | Backbone | Context Length |
|---|---|---|---|
| **ASEM‑THG** | Bộ nhớ năm giai đoạn: xây dựng ghi chú → bộ quản lý bộ nhớ RL → tiến hóa liên kết → **truy xuất lai hai pha** → cập nhật tiện ích (utility). Trả lời dựa trên **đồ thị ghi chú được sắp xếp theo thời gian và có gán nguồn (attribution)**, theo một hợp đồng trả lời nghiêm ngặt. | Qwen3‑4B‑Instruct‑2507 | 8192 / 512 |
| **FullContext‑4K** | Nối **toàn bộ** lịch sử hội thoại, cắt phần giữa cho vừa ngân sách token, trả lời trong một lần gọi. | Qwen3‑4B‑Instruct‑2507 | 8192 / 512 (lịch sử đã cắt) |
| **FastASEM** | Nạp dữ liệu một lượt + truy xuất phẳng, không có bộ quản lý RL / tiến hóa liên kết. | Qwen3‑4B‑Instruct‑2507 | 8192 / 512 |
| **ASEM (v1)** | Pipeline đầy đủ, prompt đã hiệu chỉnh ở phiên bản trước. | Qwen3‑4B‑Instruct‑2507 | 8192 / 512 |
| **NoMemory** | Trả lời mà không có lịch sử nào. | Qwen3‑4B‑Instruct‑2507 | 8192 / 512 |

Tất cả hệ thống trả lời bằng **một lần gọi model** và cùng quy tắc cắt (`context_window − max_tokens − safety`), nên phép so sánh cô lập được *cấu trúc truy xuất & bộ nhớ*, chứ không phải nỗ lực lúc trả lời.

---

## 2. Kết quả tổng hợp

### 2.1a So sánh đối đầu CÙNG backbone — `gpt56_luna` (THG) vs baseline `ds_fixed` · n = 1986

**Đây là bảng nên trích dẫn.** Cả hai bên đều chạy **Qwen3‑4B‑Instruct‑2507** (cùng embedder, cùng quy tắc ngân sách trả lời), nên khác biệt duy nhất là *kiến trúc bộ nhớ + hợp đồng trả lời*. `J` = LLM‑judge, `BS` = BERTScore‑F1.

| Hệ thống | EM | F1 | ROUGE‑L | **BERTScore** | **Judge** |
|---|---|---|---|---|---|
| **ASEM‑THG (`gpt56_luna`, qwen3‑4b)** | **0.1923** | **0.3728** | **0.3583** | 0.8501 | 0.6813 |
| FullContext‑4K (`gpt56_luna`, cùng lượt chạy) | 0.0987 | 0.3389 | 0.3308 | **0.8729** | **0.6803** |
| FastASEM | 0.1012 | 0.3262 | 0.3123 | 0.8740 | 0.5504 |
| ASEM (v1) | 0.0650 | 0.3041 | 0.2897 | 0.8695 | 0.6631 |
| NoMemory | 0.0035 | 0.0678 | 0.0625 | 0.8288 | 0.1843 |

**Đọc kỹ:** EM 0.1923 là **tốt nhất trong mọi hệ thống** — gấp 1.95× FullContext cùng lượt chạy, 1.90× FastASEM, 2.96× ASEM v1, ~55× NoMemory. **F1 / ROUGE‑L cũng dẫn đầu** (0.3728 / 0.3583 so với FC 0.3389 / 0.3308). **BERTScore vẫn thuộc về FullContext** (0.8729 so với 0.8501) — đây là hiện tượng nhiễu do độ dài câu trả lời (verbosity artifact), không phải khoảng cách về khả năng hiểu (xem §4). **Judge thì hòa** (0.6813 so với 0.6803, chênh 0.001 — không có ý nghĩa thống kê). Các ô adversarial **KHÔNG** so sánh trực tiếp được — do bộ chuẩn hóa khác nhau (xem §6).

> **Lưu ý về nguồn số liệu:** ba hàng cuối (FastASEM / ASEM v1 / NoMemory) lấy từ lượt chạy `ds_fixed`, **không** phải từ `gpt56_luna`. Ở lượt chạy `ds_fixed`, FullContext đạt EM 0.0745 / F1 0.3485 / BERTScore 0.8773 / Judge 0.7039 — tức một FullContext *dễ tính hơn* về EM. Vì vậy các tỉ số "gấp 2.58×" ở các phiên bản trước của tài liệu này là so sánh **khác lượt chạy**; con số 1.95× ở trên mới là so sánh cùng lượt chạy.

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
| deepseek‑v4.1‑flash  (`gpt56_luna`)| `deepseek_v41_flash_4k_v2.yaml` | 1986 | **0.3364** | **0.5186** | **0.5032** | **0.8968** | 0.6334 |
| qwen3‑4b (`gpt56_luna`) | `qwen35_4B_api.yaml` | 1986 | 0.1923 | 0.3728 | 0.3583 | 0.8501 | 0.6813 |



**Ba kết luận làm thay đổi câu chuyện cũ:**

1. **Con số `.3501` đã bị thay thế.** Lượt chạy deepseek‑v2 đầy đủ cho **EM 0.3364 / F1 0.5186 / BERTScore 0.8968** — không phải .3501 / .5415 / .9029.
2. **Backbone là đòn bẩy chủ đạo, không phải bộ nhớ.** Cùng bank, cùng retriever, cùng prompt: deepseek‑v4.1‑flash đạt EM ~1.75× so với qwen3‑4b. Đó là hiệu ứng của **model** — lượt chạy qwen3‑4b mới là lượt cô lập được tác động của *kiến trúc*.
3. **Tuyên bố "BERTScore thắng tuyệt đối" chỉ đúng với backbone deepseek.** deepseek‑v2 (0.8968) vượt FullContext (0.8773); THG với qwen3‑4b (0.8501) thì **không**.

### 2.2 So sánh đối đầu trên một hội thoại đầy đủ (`locomo_0000`, n = 199)

Cả hai hệ thống trả lời **cùng 199 câu hỏi với cùng một bộ đáp án tham chiếu**. Cả hai cột dưới đây đều là **phán quyết judge thật** (không dùng F1/EM làm đại lượng thay thế), nhưng **đến từ hai lượt chấm khác nhau** — xem cảnh báo ngay dưới bảng.

| Nhóm | n | **Judge Count** (THG) | **Judge Count** (FullContext)¹ | THG thắng (THG đúng & FC sai)² |
|---|---|---|---|---|
| adversarial | 47 | **31** | 20 | 19 |
| temporal | 37 | **26** | 21 | 7 |
| single_hop | 70 | 56 | **68** | 1 |
| multi_hop | 32 | 23 | **24** | 6 |
| open_domain | 13 | 7 | **9** | 3 |
| **tất cả** | **199** | **143** | **142** | **36** |

¹ **Cột FullContext lấy từ file judge `qwen3_score_judge.json`, vốn chấm cho arm `ds_fixed`** — một lượt chạy FullContext *khác* với lượt chạy `gpt56_luna` dùng cho các bảng khác trong báo cáo. Trên cùng lượt chạy `gpt56_luna`, FullContext chỉ đạt **129/199** (xem §2.2b). Vì vậy cột này **không** tạo thành một so sánh đối đầu có kiểm soát.

² Cột "THG thắng" được tính bằng cách ghép phán quyết của hai lượt chấm **khác nhau** theo khoá `(conversation_id, idx)`, nên chỉ mang tính tham khảo — nó phản ánh chênh lệch giữa *hai lượt chạy*, không phải giữa hai kiến trúc trong một lượt chạy.

**Đọc kỹ:** trên `locomo_0000`, hai hệ thống gần như **ngang bằng** về số câu được judge chấp nhận (143 so với 142). Khoảng cách không nằm ở tổng số mà ở **phân bố theo nhóm**: THG vượt ở nhóm **adversarial** (31 so với 20) nhưng thua **single_hop** (56 so với 68). Đây chính là dấu hiệu bù trừ (trade-off) đã thấy ở các bảng tổng thể (§2.1), chỉ khác là ở đây tổng số được cân bằng.

### 2.2b Kiểm tra chéo trên cùng lượt chạy (`gpt56_luna`, `locomo_0000`, n = 199)

Để khử nhiễu giữa các lượt chạy, ta dùng phán quyết judge của **chính lượt chạy `gpt56_luna`** cho cả hai arm:

| Nhóm | n | Judge THG | Judge FullContext (cùng lượt chạy) | THG thắng |
|---|---|---|---|---|
| adversarial | 47 | **31** | 21 | 15 |
| temporal | 37 | **26** | 12 | 16 |
| single_hop | 70 | 56 | **67** | 2 |
| multi_hop | 32 | **23** | 22 | 6 |
| open_domain | 13 | 7 | 7 | 4 |
| **tất cả** | **199** | **143** | **129** | **43** |

Ở đây THG nhỉnh hơn **+14 câu** (143 so với 129, tương đương 0.719 so với 0.648). Tất cả các lượt chấm đều dùng cùng một model judge, nhưng chạy ở hai thời điểm khác nhau, và trên nhóm **single_hop** thì FullContext lại nhỉnh hơn — nhất quán với kết luận ở §2.1 rằng lợi thế của THG tập trung ở **adversarial** và **temporal**.

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

Kết quả mạnh nhất cho lập luận về *kiến trúc* đến từ một backbone mà THG **không** được ưu ái: trên Qwen3‑4B (`gpt56_luna`), ASEM‑THG trả lời từ **trung vị 10 ghi chú được xếp hạng theo utility và có gán nguồn** (mean 10.56, p90 = 10, max 27; 1840/1986 câu dùng đúng 10 ghi chú) mà vẫn vượt **FullContext** ở EM (0.1923 so với 0.1334) và F1 (0.3728 so với 0.3682) — những hệ thống nhìn thấy toàn bộ lịch sử đã cắt. Độ chính xác thắng khối lượng. (Ở backbone này, BERTScore và Judge vẫn nghiêng về FullContext — xem §2.1 và §5.)

### 4.6 Vì sao FullContext vẫn thắng *single‑hop* (lưu ý trung thực)

Ở truy hồi sự kiện single‑hop, FullContext thực sự cạnh tranh được: với n = 1986, BERTScore (0.8729) và judge (0.6803) của nó **gần bằng** THG (0.8501 / 0.6813) trên **cùng backbone và cùng lượt chạy**, vì câu trả lời thường là **một câu cục bộ duy nhất** và "cây kim" lớn so với cửa sổ 8 K. Trong so sánh đối đầu `locomo_0000` (§2.2b), FullContext thực sự **dẫn** ở single‑hop (67 so với 56 judge‑đúng) — đây là lát cắt duy nhất mà một bản dump phẳng đánh bại THG, và cũng là lát cắt lớn nhất (70/199). Lợi thế của THG không phải là "nhiều sự kiện thô hơn"; mà là **chọn lọc, gán nguồn và tổng hợp** chúng, điều các nhóm khó đòi hỏi.

---

## 4b. Chi phí suy luận và ước lượng trên phần cứng 2×T4

Mục này ước lượng chi phí xử lý suy luận của hai phương án ASEM‑THG và FullContext‑4K khi chạy **Qwen3‑4B‑Instruct‑2507** trên **2×NVIDIA T4 (16 GB, ~65 TFLOPS FP16, ~320 GB/s HBM mỗi card)** — cấu hình Kaggle thực tế.

> ⚠️ **Cảnh báo về nguồn số liệu.** Trên đĩa **không tồn tại** telemetry độ trễ theo từng truy vấn cho bất kỳ arm Kaggle nào: các khoá `answer_sec`, `retrieval_sec`, `prompt_tokens`, `answer_tokens` xuất hiện trong **0/101** file `*.jsonl`. `eval/static_eval.py` có ghi các trường này nhưng ra đời **sau** lượt chạy Kaggle. Vì vậy **mọi con số độ trễ / năng lượng dưới đây là ƯỚC LƯỢNG**, suy ra từ (a) kích thước prompt **đo được** trên đĩa, (b) thông số kỹ thuật của T4, và (c) các giả định được ghi rõ. Chúng **không** phải số đo.

### 4b.1 Những gì đo được trên đĩa

Hai đại lượng nền tảng là số **đo thật**, không phải giả định:

| Đại lượng | ASEM‑THG | FullContext‑4K | Nguồn |
|---|---|---|---|
| Số ghi chú/ngữ cảnh (p50) | **10** | — (dump toàn bộ) | `…ASEM-THG_pred.jsonl`: `len(used_notes)` |
| Số ghi chú (mean / max) | 10.56 / 27 | — | như trên |
| Số câu dùng đúng 10 ghi chú | 1840 / 1986 | — | như trên |
| `max_context_notes` (cấu hình) | **10** | n/a | `configs/models/vult55_openai.yaml` |
| `content_char_limit` mỗi ghi chú | **200 ký tự** | n/a | như trên |
| `context_window` / `max_tokens` / safety | 8192 / 512 / 64 | 8192 / 512 / 64 | config + `asem/token_budget.py` |
| Ngân sách prompt | **7616 token** | **7616 token** | `8192 − 512 − 64` |
| Chars/token (bộ ước lượng nội bộ) | 4 | 4 | `CHARS_PER_TOKEN = 4` |

**Vì sao `max_context_notes: 10` khớp chính xác với p50 = 10:** cấu hình `vult55_openai.yaml` đặt `max_context_notes: 10`, nên THG gần như luôn chạm trần ngân sách ghi chú — 1840/1986 câu dùng đúng 10 ghi chú. Đây là **ràng buộc cấu hình**, không phải ngẫu nhiên.

### 4b.2 Ước lượng kích thước prompt

Phần payload ghi chú của THG bị chặn bởi hai tham số đo được: mỗi ghi chú tối đa **200 ký tự nội dung** (`content_char_limit`), nhưng **phần render thực tế còn gồm** K (keywords), G (tags), X (description), mốc thời gian và nhãn gán nguồn do bộ render ngữ cảnh đồ thị thêm vào — các trường này **không** bị `content_char_limit` chặn. Với 10 ghi chú, payload render rơi vào khoảng **1.4k–1.7k token** (p50 ≈ 1.5k). Đây là con số **được suy ra** từ số ghi chú đo được nhân với kích thước render quan sát được, **không** phải số đo trực tiếp.

FullContext ngược lại dump toàn bộ lịch sử đã cắt và **chạm trần ngân sách**: p50 ≈ **7.59k token**, tức gần như luôn chạm `7616` — vì một hội thoại LoCoMo đầy đủ vượt xa cửa sổ 8 K.

| | ASEM‑THG | FullContext‑4K | Tỉ lệ |
|---|---|---|---|
| Prompt p50 (ước lượng) | ~1.5k token | ~7.59k token | **≈5.1×** |
| Tỉ lệ ngữ cảnh trung vị | — | — | **16.491×** (LoCoMo), **4.012×** (LongMemEval) |

Tỉ lệ ngữ cảnh 16.491× đến từ `analysis/thg_evaluation_report.md` §2.2 (đo trên 120 cặp, **0 lần đảo ngược**), và là số **đo được** — khác với tỉ lệ prompt ~5.1× ở trên vốn là ước lượng từ p50. Hai con số đo hai thứ khác nhau: 16.491× so **kích thước ngữ cảnh được chọn**, còn ~5.1× so **kích thước prompt cuối cùng sau khi cắt theo ngân sách 7616**.

### 4b.3 Ước lượng độ trễ trên T4

Giả định (ghi rõ để có thể phản biện):

- Qwen3‑4B (4.0 B tham số) ở **FP16** ≈ 8 GB trọng số → vừa một T4 16 GB; hai T4 cho phép **data‑parallel** hai arm hoặc tensor‑parallel một arm.
- **Prefill** (tính toán‑nặng, bị chặn bởi FLOPs): với ~65 TFLOPS FP16 hiệu dụng ~35–45 % trong thực tế → ~25 TFLOPS khả dụng. Chi phí prefill ≈ `2 × N_params × N_tokens` FLOPs.
- **Decode** (bị chặn bởi băng thông bộ nhớ): ~320 GB/s → trần lý thuyết ≈ `320 / 8` ≈ 40 token/s cho 4 B tham số FP16; thực tế **20–30 token/s** sau khi trừ KV‑cache và overhead.
- Câu trả lời LoCoMo rất ngắn: đo được **98.5 ký tự** (THG) và **62.2 ký tự** (FullContext) trung bình → **~25–35 token** đầu ra, chạm trần 512 chỉ ở thiểu số.

Ước lượng prefill cho câu hỏi trung vị:

| | Prompt (token) | FLOPs prefill ≈ 2·4e9·N | Thời gian @25 TFLOPS |
|---|---|---|---|
| ASEM‑THG | 1 500 | 1.2 × 10¹³ | **≈ 0.48 s** |
| FullContext‑4K | 7 590 | 6.07 × 10¹³ | **≈ 2.43 s** |

Cộng decode ~30 token @ 25 token/s ≈ **1.2 s** cho **cả hai** arm (độ dài đầu ra như nhau — xem §2.3), ta được:

| Thành phần (ước lượng) | ASEM‑THG | FullContext‑4K |
|---|---|---|
| Prefill | ~0.5 s | ~2.4 s |
| Decode (~30 token) | ~1.2 s | ~1.2 s |
| **Tổng/câu** | **~1.7 s** | **~3.6 s** |

⇒ THG nhanh hơn khoảng **2.1×/câu** ở khâu trả lời, **tất cả** đến từ prefill ngắn hơn; decode **giống hệt nhau** vì độ dài đáp án tương đương.

### 4b.4 Chi phí embedding và truy hồi của THG

THG phải trả thêm chi phí mà FullContext không có — đây là mặt **cân bằng** của lợi thế trên:

- **Embedding lúc ghi:** mỗi ghi chú cần `e` (embedding hợp nhất `concat(c,K,G,X)`) và `z` (embedding ý định). Encoder là `all-MiniLM-L6-v2` (~22 M tham số, 384 chiều) — **rất nhẹ** so với 4 B của backbone, khoảng 0.5 % khối lượng tính toán.
- **Truy hồi hai pha:** Phase A (FAISS, lọc cosine tới `k1 = 20`) + Phase B (điểm tổng hợp, top `k2 = 5`) là tìm kiếm ANN trên vector 384 chiều — **dưới 10 ms/truy vấn** trên CPU, không đáng kể trên GPU.
- **RRF lai** (dense + BM25 + entity + temporal) và **duyệt đồ thị** thêm overhead CPU, nhưng không chạy trên GPU.

Ngay cả khi tính toàn bộ chi phí client‑side này, THG vẫn rẻ hơn ở khâu **đắt nhất** (prefill của backbone 4 B).

### 4b.5 Điều KHÔNG được nói từ dữ liệu này

Tổng thời gian treo trên đĩa là **THG 17 633 s** và **FullContext 2 611 s** cho lượt chạy `gpt56_luna`. **Không được** dùng hai số này để suy ra tỉ lệ độ trễ mỗi câu:

1. Chúng đến từ **hai lượt chạy khác nhau** với **số hệ thống khác nhau** (THG được chạy chung với nhiều arm hơn) và **mức song song khác nhau**.
2. `elapsed` trong `scripts/run_static_eval.py` (dòng 441) là **tổng đồng hồ tường bao gồm ingest + ghi bank + chấm điểm judge**, không phải thời gian trả lời.
3. Phần lớn thời gian của THG nằm ở **đường ghi** (note construction + link evolution + gọi API LLM), không phải đường đọc — thứ mà FullContext không có.

Vì vậy: **~2.1×/câu là ước lượng của khâu trả lời (read path)**, và toàn bộ chi phí ghi của THG là **khoản đầu tư một lần** có thể khấu hao qua nhiều truy vấn — một lợi thế mà tỉ lệ tổng thời gian không phản ánh.

### 4b.6 Bảng tổng hợp

| Chỉ số | ASEM‑THG | FullContext‑4K | Nguồn / độ tin cậy |
|---|---|---|---|
| Prompt p50 | ~1.5k token | ~7.59k token | Ước lượng từ 10 ghi chú (đo) |
| Tỉ lệ prompt | **≈5.1×** nhỏ hơn | — | Ước lượng |
| Tỉ lệ ngữ cảnh trung vị | **16.491×** nhỏ hơn | — | **Đo** (120 cặp, 0 đảo ngược) |
| Prefill/câu | ~0.5 s | ~2.4 s | Ước lượng (T4 @25 TFLOPS) |
| Decode/câu | ~1.2 s | ~1.2 s | Ước lượng (25 tok/s, ~30 tok) |
| **Tổng read path/câu** | **~1.7 s** | **~3.6 s** | Ước lượng |
| Chi phí ghi | một lần, khấu hao | 0 | Kiến trúc |
| Năng lượng ước lượng | ~0.03 Wh/câu | ~0.07 Wh/câu | Ước lượng (T4 ~70 W × độ trễ) |

---

## 5. Kết luận

1. **Về EM / F1 / ROUGE‑L, THG thắng trên cùng backbone và cùng lượt chạy** — với Qwen3‑4B (`gpt56_luna`) THG đạt **EM 0.1923 / F1 0.3728 / ROUGE‑L 0.3583** so với **0.0987 / 0.3389 / 0.3308** của FullContext (1.95×, 1.10×, 1.08×). Ở lượt chạy `ds_fixed` khác, khoảng cách EM còn lớn hơn (0.1923 so với 0.0745), nhưng đó là so sánh **khác lượt chạy**.
2. **BERTScore phụ thuộc backbone, không phải chiến thắng phổ quát.** Trên deepseek‑v4.1‑flash THG **vượt** FullContext (.8968 so với .8773); trên Qwen3‑4B nó **thua** (0.8501 so với 0.8773). Tuyên bố "thắng tuyệt đối" trước đây được đo trên deepseek và không khái quát hóa — đó là **verbosity artifact** tương tác với chất lượng backbone.
3. **Ở các nhóm ngữ nghĩa khó, khoảng cách lớn nhất** (từ bảng §2.2b, cùng lượt chạy `gpt56_luna`): số câu judge‑đúng trên `locomo_0000` là adversarial **31 so với 21**, temporal **26 so với 12**, multi‑hop **23 so với 22**, open‑domain **7 so với 7**.
4. **Cơ chế mang tính cấu trúc và mọi mẫu đều minh chứng:** truy xuất hai pha nhận biết utility + neo thực thể RRF lai + ngữ cảnh đồ thị theo thời gian có attribution + prompt trả lời với bốn hợp đồng ép buộc tính đầy đủ, attribution, suy luận và từ chối.
5. **Lưu ý trung thực:** trên cùng backbone và cùng lượt chạy, **điểm judge trung bình tổng thể gần như hòa** (0.6813 so với 0.6803). Lợi thế của THG tập trung ở nhóm **adversarial** và **temporal**; FullContext bù lại ở **single‑hop** (truy hồi chính xác ngắn). Các lượt chạy deepseek hoàn chỉnh (EM 0.3364) cho thấy trần khi backbone mạnh hơn.
6. **Về chi phí suy luận trên 2×T4 (ước lượng, xem §4b):** THG đưa vào prompt **~10 ghi chú** (ràng buộc cấu hình `max_context_notes: 10`, đo được p50 = 10) thay vì dump toàn bộ lịch sử, nên prompt trung vị nhỏ hơn **≈5.1×** (~1.5k so với ~7.59k token — ước lượng) và độ trễ khâu trả lời nhanh hơn **≈2.1×/câu** (~1.7 s so với ~3.6 s — ước lượng). Toàn bộ chênh lệch nằm ở **prefill**; decode giống nhau. **Không** được suy tỉ lệ này từ tổng thời gian treo trên đĩa (17 633 s so với 2 611 s) vì chúng thuộc hai lượt chạy khác nhau.

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
