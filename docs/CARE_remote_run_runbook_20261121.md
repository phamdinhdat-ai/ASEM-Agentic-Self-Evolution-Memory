# Chạy CARE trên máy GPU khác (runbook, 21/11/2026)

Dùng khi hàng đợi Kaggle quá lâu. Kết quả hợp lệ như chạy trên Kaggle: cùng bundle (SHA-256 kiểm khi giải nén), cùng model và
revision đã ghim, cùng cờ vLLM, cùng tham số worker. Protocol không đổi; chỉ đổi nơi chạy. Ghi vào "Deviations" của protocol
tương ứng rằng shard nào chạy ngoài Kaggle và trên GPU nào.

## Yêu cầu máy
Linux, Python ≥ 3.10, internet (Hugging Face, PyPI), **hai GPU NVIDIA ~15 GB** (T4), ~60 GB đĩa. Cấu hình vLLM đã kiểm định dùng
tensor-parallel 2, fp16, cửa sổ 8,192, một chuỗi mỗi lần. GPU khác T4 có thể đổi làm tròn fp16; check tái lập của protocol
(`care` == câu trả lời CARE đã lưu, 100%) sẽ cho biết có đổi hay không. Nếu FAIL, **không dùng** shard đó trước khi điều tra.

## Bộ kit
`build/care_remote_kit_20261121.zip` (931 KiB, 18 file): runner độc lập `run_care_on_gpu_box.py`, 12 notebook ablation, 5 notebook v3,
`kit-manifest.json` (SHA-256 từng file) và README. Không cần repo trên máy GPU. Runner chạy lại đúng hai ô code của notebook Kaggle,
chỉ thay `/kaggle/working` bằng thư mục làm việc và thêm tham số cache. `--expect-sha` từ chối notebook có bundle khác.

## Quy trình
1. Chuyển zip sang máy GPU, giải nén, làm theo `README.txt`: pilot ablation trước, rồi shard.
2. Chép `~/care/care_results_*.tar.gz` về máy này, giải nén.
3. Nhập kết quả (kiểm tra cùng cổng với scheduler Kaggle, copy vào `kaggle_outputs/`, ghi marker):
   `python -m scripts.ingest_remote_results --protocol ab --src <thư mục đã giải nén>` (và `--protocol v3`).
4. Scheduler đang chạy tự đọc marker, coi job là xong và không push nó nữa. Khi đủ shard, phần phân tích tự chạy.

## Tránh chạy trùng
Scheduler không biết bạn đang chạy shard nào ở xa cho đến khi nhập kết quả. Để khỏi tốn slot Kaggle cho job đã chạy xa, khởi động lại
scheduler với `--skip`, ví dụ nếu máy GPU nhận v3 s2–s4 và toàn bộ ablation:
- v3: `python -m scripts.schedule_care_v3 --skip 2 3 4`
- ablation: dừng scheduler ablation (không còn gì để push). Các job vẫn được tính khi nhập kết quả.
Kernel Kaggle đang QUEUED cho job đã chạy xong ở xa vẫn sẽ chạy trừ khi bạn xóa nó trên kaggle.com (CLI không có lệnh hủy).

## Thứ tự gợi ý
v3 có ưu tiên (đã khóa trước ablation): `s2 s3 s4` trước, rồi pilot ablation, rồi các shard ablation. Pilot phải qua kiểm tra
(`ingest` báo INGESTED) trước khi chạy shard ablation. Thời gian chưa đo: ước lượng v3 ~2–5 giờ GPU và ablation ~4–5 giờ
cho toàn bộ, tức xấp xỉ 6–10 giờ nếu chạy tuần tự trên một máy; pilot cho số thật.

## Đã kiểm thử ở máy này
`tests/test_care_remote.py` (7 passed): notebook → script chạy được, không còn đường dẫn Kaggle, hash bundle đọc đúng; ingest chấp nhận
một kết quả đúng dạng (hash dataset thật), từ chối kết quả sai hash code và không ghi marker; scheduler hấp thụ marker. Chưa chạy
được trên GPU thật vì máy này không có GPU Linux: lần chạy pilot đầu tiên trên máy GPU là phép thử đầu tiên với vLLM thật.
