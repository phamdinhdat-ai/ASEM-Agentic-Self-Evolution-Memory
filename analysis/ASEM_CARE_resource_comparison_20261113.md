# So sánh tài nguyên và tính toán của CARE trên LoCoMo

Ngày kiểm toán: 13/11/2026. Cùng 1.986 câu hỏi, 10 hội thoại LoCoMo, Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, vLLM FP16 trên 2 NVIDIA T4 (tensor parallel 2), cửa sổ 8.192 token, tối đa 224 token sinh, `max-num-seqs=1`. Mỗi nhánh dùng cùng dữ liệu và backbone nhưng **prompt trả lời khác nhau**.

## Token mô hình và lượt gọi

Số token bên dưới lấy từ **receipt của máy chủ**. Với CARE, năm tệp `calls.json` cùng `events.jsonl` của năm notebook Kaggle được tải lại và ghép theo thứ tự sự kiện. Tổng số lượt gọi khớp chính xác với `new_calls` của mọi sự kiện: 2.990 CARE và 3.000 Relation+Temporal. Với ba đối chứng còn lại, token nằm trong báo cáo gộp; số đếm tokenizer cục bộ trùng receipt máy chủ ở toàn bộ 1.986 câu.

| Phương án | Điểm LoCoMo | Lượt gọi | Input tổng | Output tổng | Tổng token/câu | Giây/câu* |
|---|---:|---:|---:|---:|---:|---:|
| NoMemory | 0,0774 | 1.986 | 135.877 | 18.449 | 77,7 | 0,405 |
| SimRetrieval | 0,4069 | 1.986 | 2.207.928 | 24.214 | 1.123,9 | 1,060 |
| **CARE** | **0,5256** | **2.990** | **3.525.360** | **121.767** | **1.836,4** | **3,125** |
| Relation+Temporal | 0,5251 | 3.000 | 3.545.421 | 122.113 | 1.846,7 | 3,140 |
| FullContext-8K | 0,1961 | 1.986 | 15.648.292 | 17.118 | 7.887,9 | 4,745 |

\* Thời gian là tổng thời gian ghi cho từng câu chia 1.986, **không phải** thời gian chạy toàn notebook. CARE tính cả truy hồi tại câu hỏi và API; SimRetrieval chỉ bắt đầu đồng hồ sau khi embed câu hỏi và truy hồi; FullContext-8K không tính dựng context cố định. Mọi nhánh đều không tính tải backbone, khởi động vLLM và tiền xử lý chung. Không dùng các số giây này như một so sánh latency end-to-end được kiểm soát.

CARE thêm **1.004 lượt gọi** so với hệ một lượt/câu; 50,6% câu hỏi có lượt sửa. Lượt đầu dùng trung bình **673,8 input token**; lượt sửa của 1.004 câu dùng trung bình **2.178,4 input token**. Cộng trên mọi câu, CARE dùng trung bình **1.775,1 input** và **61,3 output token/câu**. Output cao hơn SimRetrieval (12,2 token/câu) vì CARE trả JSON gồm đáp án, mã nguồn và truy vấn bổ sung, đôi khi qua hai lượt. Prompt dài nhất của **một** lượt CARE là 3.321 token; số 4.089 input token/câu cực đại là tổng của *hai* lượt, không phải một prompt đơn.

So với SimRetrieval, CARE thêm **1.414.985 token** toàn bộ (+63,4%) để tăng **0,1187** điểm LoCoMo; recall mã nguồn được chú thích tăng từ **0,5753** lên **0,7033**. So với FullContext-8K, CARE dùng **23,3%** tổng token (3,647 triệu so với 15,665 triệu), dù gọi mô hình nhiều hơn. FullContext-8K thấp điểm chủ yếu trong chế độ bị cắt nguồn, nên chênh lệch này không chứng minh CARE hơn một reader đủ context.

## Ngưỡng context và lợi thế truy vấn

| Cách đọc | Input token trung bình/câu | Input token tối đa/một lượt | Nguồn được đọc |
|---|---:|---:|---|
| NoMemory | 68,4 | 88 | Không có lịch sử |
| SimRetrieval | 1.111,7 | 1.596 | Top 18 lượt gốc theo embedding, giới hạn 10.000 ký tự |
| CARE | 1.775,1 tính cả lượt sửa | 3.321 | Top 8 BM25; mở rộng khi thiếu/sai bằng chứng, giới hạn 10.000 ký tự/lượt |
| FullContext-8K | 7.879,3 | 7.935 | 5 lượt đầu và đoạn cuối dài nhất vừa cửa sổ |

CARE có lợi thế truy vấn khi nguồn đáp án nằm xa phần cuối lịch sử: nó có thể đưa lượt cũ vào prompt theo câu hỏi, rồi truy hồi thêm bằng truy vấn do reader đề xuất. Trên 1.527 câu có đáp án và chú thích nguồn hợp lệ, FullContext-8K không còn **bất kỳ nguồn được chú thích nào ở 1.060 câu (69,4%)**. Trong nhóm này CARE đạt F1 **0,4892** so với **0,1191** của FullContext-8K. Khi toàn bộ nguồn được chú thích vẫn trong 8K, FullContext-8K lại đạt **0,5926** so với **0,4939** của CARE. CARE không thắng ở mọi chế độ; loại câu suy luận commonsense cũng thấp hơn FullContext-8K (0,1647 so với 0,2014).

Lợi thế của CARE so với SimRetrieval tập trung ở truy vấn **thời gian** (0,4415 so với 0,3019; +0,1396) và **ngữ cảnh hội thoại** (0,5763 so với 0,3658; +0,2105). Trên toàn bộ 1.540 câu có đáp án, chênh lệch F1 là +0,1524. Hai cơ chế có prompt và quy tắc trả lời khác nhau, nên các mức tăng này mô tả toàn pipeline; chưa tách riêng tác động của BM25, kiểm tra nguồn hay lượt sửa.

Nếu chừa 224 token sinh và 32 token an toàn như chính sách FullContext, **prompt dài nhất đã quan sát** đòi cửa sổ tối thiểu lần lượt khoảng **344** token cho NoMemory, **1.852** cho SimRetrieval, **3.577** cho một lượt CARE, và **8.191** cho FullContext-8K. Đây là ngưỡng *trên đúng 1.986 câu đã chạy*, không phải bảo đảm cho câu hỏi hoặc dữ liệu mới; riêng CARE vẫn được phục vụ bằng cửa sổ 8.192 token trong thí nghiệm.

Để đo ngưỡng cho một FullContext **không cắt lượt**, tôi dùng đúng tokenizer Qwen3-8B, cách dựng prompt của nhánh FullContext, và câu dài nhất được chọn theo chính sách dự trù của chương trình cho mỗi hội thoại. Số token prompt đầy đủ cộng 224 token trả lời và 32 token an toàn nằm trong khoảng **19.144–36.164 token** (trung bình theo 10 hội thoại: **30.879**). Tất cả 10 hội thoại vượt 8.192; **5/10 vượt 32.768**. Vì vậy một thí nghiệm đọc trọn cả mười hội thoại theo cùng prompt cần cửa sổ khoảng **36,2K token** theo chính sách dự trù này, và phần cứng khác có thể cần thiết. Đây là phép **tokenize ngoại tuyến**, chưa phải điểm chất lượng của một FullContext 36K. Tokenizer được kiểm tra bằng cách tái tạo 1.986 prompt NoMemory (đều khớp receipt máy chủ) và dựng lại cả 10 context bị cắt (SHA-256 và metadata đều khớp báo cáo gốc).

## Chi phí tính toán ngoài số token

- **CARE:** lập chỉ mục từ vựng/stem trên lượt gốc, xếp hạng BM25 trên các lượt của từng hội thoại và kiểm tra mã nguồn bằng CPU. Lượt sửa quét thêm xếp hạng stem, láng giềng cùng phiên và tối đa hai truy vấn gợi ý. Không cần mô hình embedding phụ và không ghi lại memory. Cài đặt hiện quét qua toàn bộ $N$ lượt cho mỗi phép xếp hạng: hai lượt quét ở nhánh một lời gọi và tối đa tám lượt quét khi sửa với hai truy vấn gợi ý. Chi phí mỗi lượt quét tăng xấp xỉ theo $N$ và số từ truy vấn; với lịch sử lớn hơn, cần cache hoặc chỉ mục đảo để tránh quét lặp.
- **SimRetrieval:** cần thêm MiniLM-L6-v2 chạy CPU để embed 5.882 lượt gốc một lần và embed từng câu hỏi; sau đó tích vô hướng với ma trận vector 384 chiều và chọn top 18. Chi phí và thời gian embed/truy hồi này **chưa nằm trong cột giây/câu**. Ma trận float32 đơn thuần có kích thước khoảng 5.882 × 384 × 4 = **9,0 MB**, chưa kể trọng số encoder, tokenizer và dữ liệu gốc.
- **FullContext-8K:** không có mô hình truy hồi phụ, nhưng Qwen phải xử lý gần 8K token prompt cho **mỗi** câu. Tổng $\sum L_j^2$ trên các prompt là **123,3 tỷ**, so với **5,83 tỷ** của CARE và **2,50 tỷ** của SimRetrieval. Đây chỉ là chỉ số quy mô phép attention ở prefill, **không phải FLOPs hay chi phí GPU đo trực tiếp**; các lớp tuyến tính, decoding, vLLM và phần cứng làm thay đổi latency thực.
- **NoMemory:** rẻ nhất nhưng không có bằng chứng hội thoại; điểm tổng chỉ 0,0774. Nhánh này còn dùng prompt yêu cầu đoán khi không chắc, nên kết quả câu không có đáp án không trực tiếp so sánh được với chính sách từ chối của CARE.

Kết luận theo ngân sách: với cửa sổ 8K và ngân sách token thấp, SimRetrieval rẻ hơn CARE nhưng kém chất lượng; CARE đổi thêm lượt gọi và token để cải thiện câu có đáp án; FullContext-8K trong cấu hình hiện tại vừa tốn token hơn vừa mất phần lớn lịch sử. Một FullContext 36K, prompt đồng nhất và đo GPU/RAM thực tế vẫn cần chạy trước khi kết luận về hiệu quả tính toán tổng quát.

## Dữ liệu kiểm toán

- [Tổng hợp token và thời gian](ASEM_CARE_resource_metrics_20261113.json)
- [Ngưỡng token cho lịch sử đầy đủ](ASEM_CARE_full_context_budget_20261113.json)
- Script: chạy `python scripts/summarize_care_resources.py` và `python -m scripts.audit_care_full_context_budget`.
- CARE Kaggle output: `phamdinhdatai/asem-care-relation-full-locomo-shard-{0..4}`, tệp `results/calls.json` và `results/events.jsonl`; bản tải về nằm tại `build/care/resource_audit/`.
