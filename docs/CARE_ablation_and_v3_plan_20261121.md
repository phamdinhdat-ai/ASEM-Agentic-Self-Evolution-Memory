# CARE: kế hoạch hoàn thiện bài báo (21/11/2026) — BẢN NHÁP ĐỂ REVIEW

Trạng thái: **kế hoạch, chưa phải protocol.** Chưa có kernel nào được submit, chưa file code nào bị sửa.
Sau khi bạn duyệt, mỗi workstream có một protocol riêng được viết và khóa **trước** khi chạy
(giống `CARE_factorial_protocol_20261117.md` và `CARE_protocol_v3_20261118.md`).

## 0. Những gì đã kiểm tra hôm nay (đều tái lập được)

| Kiểm tra | Kết quả |
|---|---|
| Bundle v3 dựng lại từ working tree hiện tại | SHA-256 `1459b45a…8144` = giá trị trong protocol v3 |
| `longmemeval_s_cleaned.json` | SHA-256 `d6f21ea9…a442` = protocol v3 |
| Code worker hiện tại vs pilot v3 | `code_sha256 837e0b25…49afea7d` giống hệt manifest của pilot |
| Pilot v3 trên Kaggle (4 câu) | COMPLETE, 42 lời gọi, ~12 s/câu |
| 5 shard v3 | **Chưa submit** (không có kernel `asem-care-v3-lmes-s*`) |
| Test CARE (`test_care_hybrid/factorial/bridge`) | 13 passed |
| E4 trên trace LoCoMo đã lưu (1,986 câu) | 1,004 lần repair: **0 do ρ một mình**. Trigger: 867 do search query (558 + 309), 137 do invalid (78 + 59) |
| Secrets (`kaggle.json`, `*.key`, `.env`) | đã bị `.gitignore` chặn |
| Tiến trình nền | PID 17912 (`schedule_care_second_backbone`) poll từ 14/11, chưa submit gì |

Lưu ý: pilot trên Kaggle có slug `asem-care-v3-longmemeval-s-pilot`, khác slug trong
`prepare_care_v3_kaggle.py` (`asem-care-v3-lmes-pilot`). Không ảnh hưởng kết quả, vì code hash khớp.
Các shard sẽ push theo slug mới `asem-care-v3-lmes-s0..s4`.

## 1. Thứ tự và phụ thuộc

```
W0  Khóa code (commit chọn lọc + tag)                      ── cần trước mọi thứ
W1  LongMemEval-S, protocol v3, 5 shard                    ── chạy ngay, bundle đã đóng băng
W2  Mechanism ablation trên LoCoMo (+ tái hiện trên LME oracle)  ── code mới ở file mới; viết trong lúc W1 chạy
W3  Baseline full-context 30K                               ── sau pilot đo tốc độ
W4  Viết v7                                                 ── sau W1 và W2
(tùy chọn) W5  Qwen3-4B 50 câu
```

W1 và W2 độc lập về dữ liệu. W2 **không được sửa** các file nằm trong bundle v3
(`bridge.py`, `hybrid.py`, `sourcefirst.py`, `run_care_factorial.py`…), nếu không hash v3 đổi và protocol v3
mất hiệu lực. Vì vậy W2 dùng **file mới** (mục 3.3).

## 2. W0 + W1 — khóa code và chạy LongMemEval-S

### 2.1 W0: khóa code
- Commit **chọn lọc**: `asem/care/`, `scripts/*care*`, `tests/test_care*`, `docs/CARE_*`, `docs/ASEM_CARE_*`,
  `paper_asem/*.tex|*.bib|figures`, `analysis/CARE_*`, `kaggle/vllm-experiment-lock.json`.
  **Không** dùng `git add -A`: `build/`, `kaggle_outputs/`, `data/atlas/` đang untracked và không bị ignore, nên sẽ kéo cả
  nghìn file kết quả/bundle vào commit.
- Tag `care-v3-freeze-20261121`; ghi SHA commit vào protocol v3 dưới dạng phụ lục "provenance"
  (đây không phải đổi nội dung protocol).
- Hiện tại `paper_asem/`, `asem/care/` đều chưa được commit, nên rủi ro mất dữ liệu là có thật.

### 2.2 W1: các bước và cổng kiểm tra
| Cổng | Điều kiện để đi tiếp | Nếu fail |
|---|---|---|
| G1 | `bundle()` SHA = `1459b45a…` và dataset SHA = `d6f21ea9…` (chạy lại ngay trước khi push) | dừng, không push |
| G2 | Pilot đã COMPLETE (đã đạt) và `code_sha256` pilot = hiện tại (đã đạt) | — |
| G3 | Push **s0 trước**, đợi xong, kiểm tra rồi mới push s1–s4 | sửa lỗi hạ tầng rồi khai báo deviation |
| G4 | Mỗi shard xong: số event = số câu kỳ vọng (98/110/100/97/95), không trùng ID, `finish_reason` ∈ {stop, length}, số call ≤ 2000, manifest có `code_sha256` | chạy lại shard (resume được) |
| G5 | Đủ 5 shard mới summarize; script từ chối nếu thiếu shard (theo mẫu `summarize_care_factorial.py`) | — |

Tốc độ: pilot 12 s/câu nhưng gồm cả encode lịch sử dài. Tôi **ước lượng ~2–5 giờ GPU** cho 500 câu, chưa đo. Vì vậy
s0 chạy trước để lấy số thật.

Phân tích (viết mới `scripts/summarize_care_v3.py`, không sửa script cũ): chỉ H-main (`bm25_hyb − bm25_lex`),
bootstrap theo câu 10,000 lần, khoảng 95%. Các tương phản khác gắn nhãn **exploratory**. Báo cáo thêm recall của
bằng chứng trên haystack đầy đủ và so sánh với `rag_dense_top18`, vì đây mới là phép thử thật cho "memory retrieval".

Kết quả có thể là *confirmed*, *not confirmed* hoặc *reversed*; bài sẽ ghi đúng kết quả đó.

## 3. W2 — Mechanism ablation (khoảng trống lớn nhất của bài)

### 3.1 Câu hỏi và mệnh đề cần kiểm
Factorial hiện chỉ đổi retriever. Chưa có bằng chứng thành phần nào của CARE tạo ra lợi ích. Các thành phần
nằm trong `bridge.py` và `hybrid.py`.

### 3.2 Các nhánh (cơ sở: CARE = `bm25_lex`, BM25 top-8, repair lexical)

| ID | Tên | Thay đổi so với CARE | Trả lời câu hỏi |
|---|---|---|---|
| X0 | `care` | không | tham chiếu; phải tái hiện câu trả lời đã lưu 1,986/1,986 |
| X1 | `no_cite_req` | chấp nhận câu trả lời không có trích dẫn hợp lệ | trích dẫn nguồn có thật sự cần không (claim "source-anchored") |
| X2 | `no_binding` | tắt kiểm tra speaker-binding | kiểm tra gán người nói |
| X3 | `no_event` | tắt kiểm tra holiday và cessation | hai check này có phải overfit LoCoMo (W4) |
| X4 | `no_checks` | X2 + X3 cùng tắt (giữ citation) | toàn bộ code check ngoài citation |
| X5 | `gate_always` | repair cho **mọi** câu | gate tiết kiệm bao nhiêu, mất gì |
| X6 | `repair_expand_only` | lần 2 dùng bundle mở rộng, **không** có correction text | tách hiệu ứng của text sửa |
| X7 | `repair_retry_only` | lần 2 dùng **bundle đầu** + correction text | "đọc lại" thuần túy có đủ không |
| X8 | `repair_no_search` | mở rộng chỉ từ câu hỏi, bỏ search query của reader | reader-guided search có giá trị không |

Đã bỏ: `gate_no_rho`. Trên 1,004 repair của LoCoMo, ρ không bao giờ là trigger duy nhất (E4 ở mục 0),
nên nhánh này sẽ giống hệt X0. Thay vào đó bài ghi đúng số này và bỏ ρ khỏi Eq. (2), hoặc giữ ρ và nói rõ nó chưa
từng tác động.
Tùy chọn (chưa đưa vào kế hoạch): `no_date_hints` (đổi prompt lượt đầu, cần +1,986 lời gọi).

Dữ liệu có sẵn: `bm25_first` (không repair) từ factorial. Phân rã lợi ích của repair:
`bm25_first → X7 (đọc lại) → X6 (mở rộng) → X0 (cả hai)`.

### 3.3 Cách cài đặt (file mới, không đụng bundle v3)
- `asem/care/ablation.py`: `AblationFlags` + `CareAblation(CareTwoStage)`, ghi đè `_reader` (bật/tắt từng check),
  `first_pass` (gate), `repair` (correction/expansion/search). Với toàn bộ cờ mặc định, kết quả phải **giống từng byte**
  `CareTwoStage.repair(mode="lexical")`.
- `scripts/run_care_ablation.py`: worker mô phỏng `run_care_factorial.py`.
  - **Backend có memo theo hash prompt, lưu cả câu trả lời thô** (hiện `calls.jsonl` chỉ lưu hash và số token). Các nhánh
    cùng prompt dùng chung một lời gọi; chi phí báo cáo cho từng nhánh vẫn tính như khi chạy riêng. Memo còn cho phép
    phân tích lại offline mà không cần GPU.
  - Mọi nhánh X1–X4 dùng chung lượt đọc đầu (check chỉ áp dụng sau khi có đầu ra thô).
- `scripts/prepare_care_ablation_kaggle.py` và `scripts/summarize_care_ablation.py`: bundle riêng, SHA riêng.
- Tests: (a) mặc định == `CareTwoStage` lexical call-for-call; (b) mỗi cờ chỉ đổi đúng hành vi của nó (backend giả có kịch bản);
  (c) kế toán call khi dùng memo; (d) resume; (e) gold không bao giờ vào prompt.

### 3.4 Dữ liệu, thống kê, quy tắc quyết định (sẽ khóa trong protocol)
- **Chính:** LoCoMo, 1,986 câu (dữ liệu dev, ghi rõ). Tái hiện: LongMemEval oracle 500 câu cho X1, X5–X8
  (X2–X4 là check đặc thù LoCoMo nên bỏ), ghi rõ oracle đã bị xem trước.
- Thước đo: điểm benchmark; phụ: điểm câu có đáp án, số refusal đúng /446, số câu có đáp án bị từ chối, calls, token.
- Tương phản: `Xi − X0`, ghép cặp, bootstrap theo hội thoại 10,000 lần, Bonferroni cho 8 phép so sánh (khoảng 99.375%);
  thêm số hội thoại thắng/10 và bootstrap theo câu (vì chỉ có 10 cụm nên khoảng theo cụm lạc quan).
- Kết luận cho từng thành phần: **đóng góp** nếu khoảng đã hiệu chỉnh nằm hẳn dưới 0 khi gỡ bỏ; **không cần** nếu nằm trong
  [−0.01, +0.01]; **gỡ còn tốt hơn** nếu hẳn trên 0 (ghi là phát hiện post-hoc, đề xuất đơn giản hóa);
  còn lại **không kết luận**. Biên ±0.01 là giả định của tôi, bạn cần xác nhận.
- Không dùng kết quả ablation để nói CARE hơn hệ thống khác; đây là phân tích nội tại.

### 3.5 Ước lượng chi phí (dựa trên tốc độ đã đo ở factorial: ~2.2 s/lời gọi, LoCoMo)
~1,986 lượt đầu + ~8 × 1,004 repair + ~1,986 cho `gate_always` ≈ **12,000 lời gọi ≈ 5–7 giờ GPU** (chia 5 shard).
LME oracle tái hiện: khoảng 1 giờ. Con số này chưa đo; pilot 4 câu sẽ cho số thật trước khi chạy đầy đủ.

## 4. W3 — Baseline full-context
Lần thử 40K lỗi vì KV-cache chỉ chứa 30,432 token khi `max_model_len=40960`, dùng 2×T4 và utilization 0.88.
Lịch sử LoCoMo dài khoảng 36K token, nên mốc 30K vẫn **cắt bớt**.
- Đề xuất: đặt `max_model_len` ≈ 30,000, gọi là **FullContext-30K (truncated)**, không gọi là "full".
  Thử `--kv-cache-dtype fp8` trong pilot, nhưng chưa biết T4 có hỗ trợ hay không; nếu không, dùng 30K.
- Chi phí chưa rõ: prefill 30K token/lời gọi trên T4 có thể chậm hơn nhiều so với 2.2 s. Pilot 20 câu đo s/call; nếu chạy
  1,986 câu quá đắt thì dùng tập con phân tầng (30 câu/hội thoại = 300 câu) và nói rõ trong bài.
- Giá trị cho bài: bỏ được cảnh báo "full-context chỉ 8K" ở bảng chính. Kết quả có thể bất lợi cho CARE nếu full-context 30K mạnh;
  vẫn báo cáo nguyên văn.

## 5. W5 (tùy chọn) và dọn dẹp
- Qwen3-4B 50 câu: chỉ là kiểm tra độ nhạy, không đủ để nói về tính tổng quát. Nên làm nếu còn quota, không bắt buộc.
- Tắt scheduler PID 17912 (đang poll vô ích, và nếu Kaggle trả về trạng thái khác nó có thể tự submit kernel 4B).

## 6. W4 — Hướng viết v7 (sau khi có W1 và W2)

**Điểm mạnh nên đặt lên trước (đều có số liệu):**
1. CARE vượt RAG cùng reader trên LoCoMo: +0.064, khoảng [+0.039, +0.089], tốn 1.6 lần token (1,836 so với 1,180/câu).
2. Repair cải thiện 279 câu có đáp án và không làm giảm câu nào; chi phí là 46 refusal đúng bị mất.
3. Định lý break-even với ngưỡng G/L = 3.89, kèm bằng chứng 6 verifier đều dưới ngưỡng. Đây là đóng góp phân tích thực chất.
4. Ablation thành phần (W2): phân rã lợi ích của repair và xác định check nào thật sự cần.
5. Quy trình: đăng ký giả thuyết trước, báo cả kết quả âm, tái lập từng byte, chỉ dùng model mở.

**Sẽ không viết** (dữ liệu hiện tại không đỡ được): graph giúp ích; memory evolution giúp ích; hơn các hệ thống đã công bố;
hybrid repair (CARE-H) tốt hơn đã được xác nhận (chỉ viết sau khi W1 xác nhận).

Cấu trúc: đóng góp 1 = đo chi phí/lợi ích của đọc lần hai + định lý; đóng góp 2 = so sánh có kiểm soát + ablation thành phần;
đóng góp 3 = phát hiện về encoder; kết quả âm (graph, verifier) đặt ở mục riêng "what did not work".
Sửa thêm: bỏ ρ khỏi Eq. (2) hoặc ghi chú E4; cập nhật Table 1 khi có 30K; thêm mục LME-S; cập nhật phụ lục provenance.

## 7. Rủi ro

| Rủi ro | Tác động | Giảm thiểu |
|---|---|---|
| LME-S không xác nhận CARE-H | mất claim "hybrid tốt hơn" | đã có sẵn cách viết; abstract hiện tại đã dè dặt |
| Ablation cho thấy nhiều check "không cần" | CARE đơn giản hơn, ít "novel" hơn | đây là kết quả trung thực và là thông tin có ích; viết như vậy |
| Quota GPU Kaggle không đủ (tổng ước ~8–15 giờ, chưa đo) | trễ | s0 và pilot ablation đo thật trước; ưu tiên W1 > W2 > W3 |
| OOM encoder trên LME-S (đã gặp ở oracle) | shard lỗi | batch 2,048 đã áp dụng; resume được; khai báo deviation |
| Sửa nhầm file trong bundle v3 | mất hiệu lực protocol | G1 kiểm hash trước khi push; W2 chỉ dùng file mới |
| Tuning trên LoCoMo lần nữa | lạm dụng dữ liệu dev | ablation chỉ **gỡ** thành phần, không tìm tham số mới; gắn nhãn dev |

## 8. Cần bạn quyết định

1. **Commit/tag chọn lọc (W0)** như mục 2.1: đồng ý không?
2. **Danh sách nhánh X1–X8**: thêm/bớt? Có muốn thêm `no_date_hints`?
3. **Biên "không cần" ±0.01** và Bonferroni: giữ hay đổi?
4. **Có chạy tái hiện trên LME oracle** (≈ +1 giờ GPU) hay chỉ LoCoMo?
5. **Full-context**: chấp nhận nhãn "30K (truncated)" và tập con 300 câu nếu quá đắt?
6. **Tắt scheduler 4B** và có làm W5 không?
7. **Quota GPU Kaggle** hiện còn bao nhiêu giờ? Tôi không đọc được số này qua CLI.
8. Thứ tự: bắt đầu push s0 của W1 ngay sau khi commit, trong lúc tôi viết code W2?
