# Nhật ký chạy protocol v3 (LongMemEval-S), 21/11/2026

Protocol `CARE_protocol_v3_20261118.md` không bị sửa. Ghi tại đây những gì đã làm và độ lệch so với kế hoạch.

## Đã kiểm tra trước khi submit
- Bundle dựng lại: SHA-256 `1459b45ac907794c283cbf76f17946a1cdb84e97b1245880a80f9780c72a8144` (khớp protocol).
- Dataset `longmemeval_s_cleaned.json`: SHA-256 `d6f21ea9…a442` (khớp).
- `code_sha256` của worker `837e0b259a638ed1…` giống manifest của pilot v3.
- Pilot v3 (4 câu) đã COMPLETE trên Kaggle (slug `asem-care-v3-longmemeval-s-pilot`).

## Đã submit
- `asem-care-v3-lmes-s0` và `asem-care-v3-lmes-s1`, 21/11/2026, trạng thái QUEUED lúc 14:25 UTC.
- s2 bị Kaggle từ chối: `Maximum batch GPU session count of 2 reached`. Vì vậy s2, s3, s4 được xếp hàng bằng scheduler.

## Độ lệch đã khai báo
- Cổng G3 (đợi s0 xong rồi mới push shard khác) được nới theo yêu cầu "gửi và chạy luôn": s0 và s1 chạy song song.
  Không ảnh hưởng kết quả vì mọi shard dùng cùng bundle và cùng hash.

## Scheduler
- `python -m scripts.schedule_care_v3 --poll-seconds 300` (file mới, không nằm trong bundle).
- Tiến trình nền đã khởi động (PID 5148). Trạng thái: `build/care/v3_scheduler_status.json`; log: `build/care/v3_scheduler.log`.
- Mỗi lần poll: kiểm tra trạng thái kernel, push shard kế tiếp khi còn chỗ (tối đa 2 đang chạy), tải và kiểm tra shard xong
  (số dòng, ID không trùng, hash dataset và code, `gold_in_model_context=False`, số call ≤ 2000, đủ arm).
  Shard lỗi hoặc không qua kiểm tra thì scheduler dừng và ghi `failed`, không tự thử lại.
- Khi đủ 5 shard hợp lệ, scheduler chạy `scripts.summarize_care_v3` và ghi
  `analysis/CARE_v3_lmes_s_20261121.{json,md}`.
- **Nếu máy khởi động lại hoặc tiến trình chết** (đã từng xảy ra 15/11): các kernel đang chạy trên Kaggle vẫn tiếp tục;
  chạy lại lệnh trên, scheduler đọc state và tiếp tục. Không chạy hai bản cùng lúc.
- Đã chạy `--once` hai lần trong khi tiến trình nền hoạt động (không gây push trùng vì chưa có chỗ trống).

## Việc cần làm khi có kết quả
1. Đọc `analysis/CARE_v3_lmes_s_20261121.md`. Chỉ H-main (`bm25_hyb − bm25_lex`) là kiểm định xác nhận; mọi so sánh khác là exploratory.
2. Cập nhật `paper_asem/asem_v7.tex`: các chỗ `\pending{…}` (mục 4, mục 6, giới hạn h) và đoạn nói về LongMemEval-S.
3. Nếu H-main là "not confirmed" hoặc "reversed", giữ nguyên câu CARE-H là phát hiện post-hoc (Phụ lục B).

## Lưu ý khác
- Tiến trình `schedule_care_second_backbone` (PID 17912) vẫn poll với `pilot_status UNKNOWN`. Nó chưa submit gì;
  nếu muốn chắc chắn không có kernel 4B nào bị submit ngoài ý muốn thì tắt nó.
- Chưa commit/tag. `paper_asem/`, `asem/care/`, `scripts/*care*`, `docs/CARE_*` vẫn untracked.

## Cập nhật 15:10 UTC
- s0 và s1 vẫn QUEUED (Kaggle chưa cấp GPU). Kaggle CLI không có lệnh hủy kernel, chỉ có `delete`; không xóa vì sẽ mất chỗ trong hàng đợi.
- Đã dừng tiến trình `schedule_care_second_backbone` (PID 17912). Scheduler v3 (PID 5148) vẫn chạy.
- Ablation (W2) đã được cài đặt, kiểm thử và đóng gói, chưa push: xem `docs/CARE_ablation_protocol_20261121.md`.
  Bundle v3 được kiểm tra lại sau khi thêm file mới: vẫn `1459b45a…8144`.

## Cập nhật 16:15 UTC
- s0 và s1 QUEUED đã 1 giờ 50 phút. Máy local không có GPU (không `nvidia-smi`, `torch.cuda` False), nên không chạy local được.
  Đổi sang API model khác làm hỏng kiểm tra tái lập, nên không làm.
- Protocol ablation đã khóa theo các mặc định. Scheduler ablation chạy nền (PID 7880, log `build/care/ablation_scheduler.log`,
  state `build/care/ablation_scheduler_status.json`); nó đợi cho tới khi v3 đã submit hết shard rồi mới dùng slot rảnh.
- Nếu máy khởi động lại: chạy lại `python -m scripts.schedule_care_v3 --poll-seconds 300` và
  `python -m scripts.schedule_care_ablation --poll-seconds 300`; cả hai đọc state và tiếp tục.

## Cập nhật 17:00 UTC
- Kaggle vẫn QUEUED (s0, s1) sau hơn 2.5 giờ. Đã làm đường chạy trên máy GPU khác: `docs/CARE_remote_run_runbook_20261121.md`.
- Scheduler khởi động lại để nhận marker ingest và tùy chọn `--skip`: v3 PID 13700, ablation PID 24072 (state giữ nguyên).

## Cập nhật 22/11/2026 (21:30 giờ máy)
- s0 và s1 đều ERROR: `torch.OutOfMemoryError` trong `DenseIndex` (embed_care_turns.py). vLLM dùng `gpu_memory_utilization=0.80`
  (TP=2, ~11,65 GiB/GPU) nên bộ nhúng chỉ còn ~1,8 GiB trên GPU 0. s0 chết sau 611 s, s1 sau 2189 s. Log trong `build/care/v3_scheduler_status.failed_oom_20261122.json`.
- Sửa: `gpu_memory_utilization` 0.80 → 0.70 trong `scripts/prepare_care_v3_kaggle.py` (tham số notebook, không thuộc `care_code`).
  Bundle SHA không đổi: `1459b45a…8144`. Chỉ là cài đặt chạy, không đổi dữ liệu hay mã đánh giá.
- Đã đẩy lại s0 (RUNNING) và s1 (QUEUED). Scheduler v3 và ablation khởi động lại.
- Ablation pilot LME và LoCoMo đã COMPLETE; scheduler ablation sẽ nhận khi poll.
