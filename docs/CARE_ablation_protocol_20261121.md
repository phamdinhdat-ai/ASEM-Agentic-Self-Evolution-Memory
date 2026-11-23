# Protocol: CARE component ablation (21/11/2026) — ĐÃ KHÓA

Trạng thái: **khóa ngày 21/11/2026 (khoảng 16:15 UTC), trước khi bất kỳ kernel ablation nào được push.** Người dùng yêu cầu
chạy ("tìm cách chạy đi nhé") sau khi đọc bản nháp; các mục ⚠ được chốt theo **giá trị mặc định ghi trong bản nháp**:
biên "không cần" ±0.01, Bonferroni cho 8 nhánh, có chạy LongMemEval oracle, không có nhánh `no_date_hints`. Không sửa nội dung
sau khi có kết quả đầu tiên; mọi lệch hướng ghi vào mục "Deviations".

## Mục đích
Factorial chỉ đổi retriever; chưa có bằng chứng thành phần nào của CARE tạo ra lợi ích. Ablation này tắt **từng** cơ chế
của CARE và đo hậu quả, để (i) nói đúng thành phần nào đóng góp, (ii) bỏ thành phần không cần, (iii) tách lợi ích của
repair thành "đọc lại" và "mở rộng bằng chứng". Đây là phân tích nội tại; không dùng để tuyên bố hơn hệ thống khác.

## Code under test
- Bundle ablation (tách biệt với bundle v3 đã khóa): SHA-256 `5699cf7f86bdba5a71c5d1e8d4f51896a4bc5348d38e05d0475f5415ff62a67b`
  (`python -m scripts.prepare_care_ablation_kaggle`).
- `code_sha256` của worker (ablation.py, memo_backend.py, hybrid.py, bridge.py, sourcefirst.py, run_care_ablation.py):
  `449cf06696e06773635ce8c71d16e14b074d64e5893ea969b656bc94420cff28`. Scheduler từ chối shard có hash khác.
- Bundle v3 vẫn là `1459b45a…8144` (kiểm tra lại sau khi thêm các file mới): ablation chỉ thêm file, không sửa file nào của v3.
- Tests: `tests/test_care_ablation.py` (12 passed): cờ mặc định == `CareTwoStage` lexical call-for-call trên 60 câu LoCoMo,
  kể cả khi ép chạy lại đường kiểm tra lại; mỗi cờ chỉ đổi đúng cơ chế của nó; kế toán memo; worker + resume.

## Nhánh (cơ sở = CARE: BM25 top-8, lặp lại lexical, một lần repair)
| Nhánh | Thay đổi so với CARE | Cờ |
|---|---|---|
| `first_pass` | không repair (tham chiếu, 1 call) | — |
| `care` | không (tham chiếu; phải tái hiện câu trả lời đã lưu) | — |
| `no_cite_req` | chấp nhận câu trả lời không trích dẫn hợp lệ | `require_citation=False` |
| `no_binding` | tắt kiểm tra speaker-binding | `check_binding=False` |
| `no_event` | tắt kiểm tra holiday và cessation | `check_event=False` |
| `no_checks` | tắt cả binding và event (giữ citation) | hai cờ trên |
| `gate_always` | repair mọi câu | `gate_always=True` |
| `repair_expand_only` | lần 2 dùng bundle mở rộng, không có dòng CORRECTION | `correction_text=False` |
| `repair_retry_only` | lần 2 dùng **bundle đầu** + CORRECTION | `expand=False` |
| `repair_no_search` | mở rộng chỉ từ câu hỏi, bỏ search của reader | `use_search=False` |

Không có nhánh `gate_no_rho`: trên 1,004 repair đã lưu của LoCoMo, ρ chưa bao giờ là trigger duy nhất (867 do search, 137
do câu trả lời invalid), nên nhánh này sẽ giống hệt `care`. Bài sẽ ghi số này thay vì chạy lại.
Không có `no_date_hints` (đổi prompt lượt đầu, cần +1,986 call): ⚠ thêm nếu bạn muốn.

## Cài đặt cố định
Qwen3-8B (revision ghim, vLLM, fp16, 2×T4, cửa sổ 8,192, nhiệt độ 0, không thinking, ≤224 token), giới hạn bằng chứng
10,000 ký tự, bundle đầu 8 turn, tối đa một repair. **Không dùng encoder GPU** (chỉ BM25), nên ít rủi ro OOM hơn factorial.
Gold chỉ đọc sau suy luận (`gold_in_model_context=False` trong manifest). Mỗi arm báo calls và token như thể chạy riêng; số
lời gọi vật lý thấp hơn vì các arm gửi cùng prompt dùng chung một lần gọi (memo theo hash prompt, lưu cả phản hồi thô).

## Dữ liệu và vai trò
- **LoCoMo**, 1,986 câu: dữ liệu phát triển. Kết luận chính của ablation nằm ở đây, ghi rõ là dev.
- **LongMemEval cleaned oracle**, 500 câu: tái hiện. Kết quả đã được xem trước nên không độc lập; check đặc thù LoCoMo
  (`no_binding`, `no_event`) vẫn chạy vì worker chạy mọi arm, nhưng chỉ báo cáo kèm cảnh báo.
- ⚠ Có chạy LME oracle (khoảng +1 giờ GPU)? Mặc định: có.

## Thước đo và kiểm định (khóa)
- Chính: điểm trung bình theo benchmark (LoCoMo scorer; LME diagnostic F1). Phụ: điểm câu có đáp án, refusal đúng, câu có đáp án
  bị từ chối, số repair, calls, token.
- Tương phản `nhánh − care`, ghép cặp; bootstrap theo hội thoại trên LoCoMo, theo câu trên LME; 10,000 lần, seed 20261121;
  khoảng 95% và khoảng Bonferroni cho 8 nhánh (99.375%).
- **Quy tắc kết luận cho mỗi cơ chế** (biên ⚠ ±0.01 là giả định của tôi, bạn xác nhận):
  - khoảng đã hiệu chỉnh nằm hẳn dưới 0 → **đóng góp** (gỡ bỏ làm giảm điểm);
  - nằm hẳn trên 0 → **gỡ còn tốt hơn** (phát hiện post-hoc, đề xuất đơn giản hóa);
  - nằm trong [−0.01, +0.01] → **không cần**;
  - còn lại → **không kết luận**.
- Phân rã lợi ích repair: `first_pass → repair_retry_only → repair_expand_only → care` (phần chia sẻ của mức tăng).
- Báo cáo thêm số hội thoại thắng và điểm theo loại câu hỏi.

## Kiểm tra tái lập (khóa)
- `care` phải tái hiện đúng câu trả lời CARE đã lưu (`bm25_lex` trong `kaggle_outputs/care_fx`) trên **100%** câu của mỗi benchmark;
  summarizer in PASS/FAIL. Nếu FAIL, phân tích bị gắn cờ và phải điều tra trước khi dùng.
- Pilot (4 câu mỗi benchmark) phải hoàn tất không lỗi và qua kiểm tra của scheduler trước khi push shard nào.

## Chi phí (ước lượng, chưa đo; pilot cho số thật)
LoCoMo: ~1,986 lượt đầu + repair của các nhánh, sau khi chia sẻ ≈ 7,000 lời gọi vật lý (trần 2,600 mỗi shard); tốc độ
factorial ~2.2 s/call có encoder, nên ~3–4 giờ GPU tổng. LME oracle ≈ 1 giờ. Chạy sau v3, tối đa 2 kernel cùng lúc.

## Cách chạy (sau khi chốt)
1. Build lại và kiểm hash trước khi push (đã kiểm lúc khóa: `5699cf7f…`): `python -m scripts.prepare_care_ablation_kaggle`.
2. `python -m scripts.schedule_care_ablation --poll-seconds 300` (mặc định chờ v3 xong rồi mới push; `--no-wait-for-v3`
   nếu muốn chen khi v3 chưa dùng hết slot). Trạng thái: `build/care/ablation_scheduler_status.json`.
3. Khi đủ 10 shard hợp lệ, scheduler chạy `summarize_care_ablation` cho cả hai benchmark và ghi
   `analysis/CARE_ablation_{locomo,lme}_20261121.{json,md}`.

## Điều được và không được tuyên bố
- Được: thành phần nào đóng góp/không cần trên LoCoMo (dev) và có tái hiện trên LME oracle hay không; phần của "đọc lại" và "mở rộng".
- Không: CARE hơn hệ thống khác; kết luận độc lập (LoCoMo là dev); kết luận về check đặc thù LoCoMo trên LME.

## Deviations
(trống; điền khi có)
