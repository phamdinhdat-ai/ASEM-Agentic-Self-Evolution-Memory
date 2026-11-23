# Đánh giá CARE so với hướng graph cũ và RAG đơn giản

Ngày kiểm tra: 12/11/2026. Tập LoCoMo công khai có 1.986 câu hỏi trên 10 hội thoại. CARE Bridge v2 đã chạy đủ bộ; Relation+Temporal là ablation của CARE. Graph cũ chỉ có 3/8 shard hoàn thành và BM25 chỉ có checkpoint ghép cặp trên 1.128 câu. Mọi con số dưới đây được tính trên **cùng câu hỏi** trong từng phép so sánh; không coi tập con cũ là kết quả full.

## Điểm LoCoMo

Các câu thuộc category 1–4 dùng token F1 theo quy tắc LoCoMo; category 5 dùng điểm nhị phân cho câu trả lời từ chối. Vì vậy “điểm tổng” không phải token F1 thuần túy. Dự đoán cũ được chấm lại bằng chính scorer của CARE, không sửa văn bản dự đoán.

| Phép so sánh | Số câu | Điểm cũ | CARE Bridge v2 | Chênh lệch CARE |
|---|---:|---:|---:|---:|
| Graph cũ, cùng 746 câu | 746 | 0,3786 | 0,5188 | +0,1402 |
| Source-hybrid cũ, cùng 746 câu | 746 | 0,3467 | 0,5188 | +0,1721 |
| BM25 RAG cũ, cùng 1.128 câu | 1.128 | 0,1372 | 0,4912 | +0,3540 |

Trên **câu có đáp án**, CARE hơn graph cũ 0,4679 so với 0,2635 (590 câu) và hơn BM25 0,4477 so với 0,1622 (954 câu). Lợi thế này không chỉ do thay câu từ chối. Tuy nhiên, trên 156 câu adversarial của graph cũ, graph từ chối đúng 0,8141 còn CARE đạt 0,7115; source-hybrid cũ đạt 0,8718. BM25 cũ đạt 0 trên 174 câu adversarial vì dùng câu “I don't know”, không khớp hợp đồng chấm điểm từ chối của LoCoMo.

CARE tăng mạnh ở category 2, 3, 4 so với graph cũ. **Category 1 là ngoại lệ:** graph cũ đạt 0,2616, CARE đạt 0,2418 trên cùng 102 câu liệt kê. Ví dụ `conv-50:53`: đáp án chuẩn “friends and team”, graph cũ có cụm này trong danh sách nên được 1, CARE chỉ trả “Dave” nên được 0. Ở `conv-42:239`, graph cũ từ chối đúng còn CARE tạo ra một mô tả phim dù câu thuộc nhóm adversarial.

## BERTScore F1

Các dự đoán cũ và CARE được chấm lại bằng cùng `roberta-large` revision `722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, no IDF và cùng phiên bản phần mềm. Bảng dùng **raw F1**, trên đúng tập câu của từng phép so sánh.

| Phép so sánh | Số câu | BERTScore F1 cũ | CARE Bridge v2 | Chênh lệch CARE |
|---|---:|---:|---:|---:|
| Graph cũ | 746 | 0,8310 | 0,9083 | +0,0774 |
| Source-hybrid cũ | 746 | 0,8428 | 0,9083 | +0,0655 |
| BM25 RAG cũ | 1.128 | 0,7979 | 0,9093 | +0,1114 |

Trên câu có đáp án, CARE cũng hơn graph cũ 0,9139 so với 0,8224 và hơn BM25 0,9131 so với 0,7863. Nhưng BERTScore **không thay thế điểm từ chối đúng**: trên câu adversarial, CARE có BERTScore cao hơn graph cũ dù graph cũ có accuracy từ chối tốt hơn. Cách diễn đạt “No information available” và reference “Not mentioned” ảnh hưởng similarity mà không cho biết câu trả lời có căn cứ hay không.

## Giới hạn và kết luận

- Cả CARE và graph cũ/BM25 dùng cùng dataset SHA và cùng revision Qwen3-8B cho câu trả lời. Prompt, ngân sách, retrieval, reader và hợp đồng từ chối khác nhau; phép so sánh đánh giá **toàn pipeline**, không chứng minh một thành phần graph hay memory cụ thể gây ra chênh lệch.
- Graph cũ chỉ hoàn tất shard 0, 1, 3; BM25 là tập checkpoint bị gián đoạn. Khoảng bootstrap theo hội thoại trong [báo cáo điểm LoCoMo](ASEM_CARE_vs_previous_20261112.md) mô tả các tập đã có, không phải kiểm định xác nhận trên benchmark độc lập.
- FullContext Qwen3-8B cũ bị chặn vì lịch sử vượt giới hạn evidence 4.096 token; điểm 0 của nhánh này không phải phép so chất lượng. FullContext, SimRetrieval và ASEM-THG Qwen3-4B trước đó dùng backbone và giao thức khác, nên chưa có đối đầu công bằng với CARE.
- CARE Bridge v2 có cải thiện rõ rệt trên phần đối chiếu được của graph cũ và BM25, nhưng còn yếu ở câu liệt kê và từ chối sai. Relation+Temporal không cải thiện Bridge v2 trong lượt full: 0,5251 so với 0,5256, chỉ đổi 15/1.986 câu. Chưa có bằng chứng rằng phần mở rộng này tạo đóng góp mới.

Nguồn kiểm tra: [bảng ghép cặp và từng category](ASEM_CARE_vs_previous_20261112.md), [BERTScore ghép cặp](ASEM_CARE_vs_previous_20261112_bertscore.md), [kết quả full CARE](ASEM_CARE_full_20261112.md).
