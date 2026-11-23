# Tổng quan thực nghiệm CARE và các phương án ngữ cảnh trên LoCoMo

Ngày tổng hợp: 12/11/2026. Báo cáo này dùng các kết quả đã hoàn tất, không thay đổi dự đoán hay điểm CARE đã khóa.

## 1. Phương pháp thực sự đang làm gì?

**CARE Bridge v2** giữ các lượt hội thoại gốc cùng mã nguồn, người nói và ngày. Với mỗi câu hỏi, nó xếp hạng lượt hội thoại bằng khớp từ/BM25, đọc tối đa 8 nguồn ở lượt đầu, yêu cầu mô hình trả lời kèm mã nguồn, và kiểm tra ràng buộc người nói, sự kiện, thời gian. Nếu thiếu bằng chứng hoặc câu trả lời không qua kiểm tra, nó mở rộng nguồn bằng truy hồi từ gốc và chỉ gọi mô hình thêm một lần. Nguồn gốc cuối cùng vẫn là lượt hội thoại gốc; CARE không tạo một kho ký ức đã huấn luyện hay tự tiến hóa trong thí nghiệm này.

Đối chứng hiện tại:

| Phương án | Thông tin đưa vào câu trả lời | Số lần gọi mô hình trả lời |
|---|---|---:|
| CARE Bridge v2 | Nguồn gốc được xếp hạng, kiểm tra mã nguồn, sửa bằng chứng có điều kiện | 2.990/1.986 câu; 1.004 câu cần lượt hai |
| SimRetrieval hiện tại | Top 18 lượt gốc theo cosine của MiniLM-L6-v2; giới hạn 10.000 ký tự | 1.986 |
| FullContext-8K | 5 lượt đầu và chuỗi lượt gần nhất vừa cửa sổ 8K | 1.986 |
| NoMemory | Chỉ câu hỏi, không có hội thoại | 1.986 |

SimRetrieval ở đây là **truy hồi dense trên lượt gốc**, khác với baseline SimRetrieval cũ trong mã dự án vốn xây atomic notes bằng LLM trước khi tìm kiếm. Do đó đây là đối chứng cho truy hồi tương đồng đơn giản, không phải phép chạy lại nguyên xi hệ thống cũ.

## 2. Phạm vi và cách chấm

- Cùng 10 hội thoại LoCoMo, 1.986 câu hỏi, 5 shard Kaggle, không lỗi suy luận. Dữ liệu SHA256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`. Đầu vào gồm văn bản, ngày và caption ảnh có sẵn; đây không phải thí nghiệm đa phương thức với pixel ảnh.
- Cùng Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` trên Kaggle T4 x2, cửa sổ 8.192 token, tối đa 224 token trả lời, nhiệt độ 0, tắt thinking.
- Điểm **LoCoMo tổng hợp** là trung bình token F1 cho loại 1–4 và điểm từ chối nhị phân cho loại 5. Không nên gọi điểm tổng hợp là token F1 thuần. BERTScore là F1 tương đồng ngữ nghĩa, dùng cùng `roberta-large` revision `722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17, không IDF; nó không xác thực sự thật hay nguồn dẫn.
- Đáp án vàng và mã nguồn vàng chỉ được dùng sau suy luận để chấm và phân tích. Câu trả lời của bốn nhánh được đối chiếu trên đúng cùng ID câu hỏi. Tuy nhiên lời nhắc trả lời khác nhau theo phương pháp, nhất là NoMemory được yêu cầu đoán khi không chắc.

## 3. Kết quả đầy đủ, cùng Qwen3-8B

| Phương án | LoCoMo tổng hợp | F1 loại 1–4, n=1.540 | Từ chối đúng loại 5, n=446 | BERTScore F1 thô | BERTScore F1 rescaled |
|---|---:|---:|---:|---:|---:|
| **CARE Bridge v2** | **0,5256** | **0,4648** | **328/446 (73,5%)** | **0,9088** | **0,4595** |
| Relation+Temporal, ablation CARE | 0,5251 | 0,4603 | 334/446 (74,9%) | 0,9083 | 0,4565 |
| SimRetrieval lượt gốc | 0,4069 | 0,3124 | 327/446 (73,3%) | 0,8896 | 0,3460 |
| FullContext-8K | 0,1961 | 0,2406 | 19/446 (4,3%) | 0,8912 | 0,3556 |
| NoMemory | 0,0774 | 0,0998 | 0/446 | 0,8776 | 0,2745 |

CARE hơn SimRetrieval **0,1187 điểm tổng hợp**, bootstrap ghép theo hội thoại 95% `[+0,0837; +0,1503]`; trên 1.540 câu có đáp án, chênh lệch là **+0,1524**. Chênh lệch từ chối giữa hai hệ thống chỉ **1/446 câu**, khoảng bootstrap cho chênh lệch này chứa 0. Vì vậy lợi ích chính nằm ở việc trả lời câu có đáp án, không phải khả năng từ chối vượt trội so với SimRetrieval.

CARE cần **2.990 lượt gọi** để trả lời 1.986 câu, nhiều hơn 50,6% so với đối chứng một lượt gọi. Tỷ lệ recall của mã nguồn vàng trong bundle trả lời là **0,7033** ở CARE, so với **0,5753** ở SimRetrieval. Điều này phù hợp với lợi ích của bước mở rộng bằng chứng, nhưng các hệ thống cũng khác lời nhắc và kiểm tra nguồn nên chưa thể quy toàn bộ cải thiện cho riêng truy hồi.

## 4. Theo loại câu: mạnh ở đâu, yếu ở đâu?

| Loại LoCoMo | n | CARE | SimRetrieval | FullContext-8K | NoMemory |
|---|---:|---:|---:|---:|---:|
| 1 — single-hop, gồm câu liệt kê | 282 | **0,2609** | 0,2283 | 0,2353 | 0,0966 |
| 2 — thời gian | 321 | **0,4415** | 0,3019 | 0,2367 | 0,0336 |
| 3 — commonsense/suy luận | 96 | 0,1647 | 0,1263 | **0,2014** | 0,1256 |
| 4 — ngữ cảnh hội thoại | 841 | **0,5763** | 0,3658 | 0,2484 | 0,1232 |
| 5 — tiền đề sai/thiếu thông tin | 446 | **0,7354** | 0,7332 | 0,0426 | 0,0000 |

CARE có lợi thế lớn ở loại 2 và 4, là hai nhóm cần tìm đúng người, sự kiện và thời điểm trong lịch sử dài. **Loại 3 là điểm yếu:** FullContext-8K cao hơn CARE 0,2014 so với 0,1647. Ở loại 1, lợi thế CARE trước FullContext-8K nhỏ; so với graph cũ trên tập ghép cặp, CARE còn thấp hơn ở nhóm này. Relation+Temporal chỉ đổi đáp án **15/1.986 câu**, điểm tổng giảm nhẹ 0,5256 xuống 0,5251; chưa có bằng chứng rằng phần mở rộng này cải thiện phương pháp.

## 5. Vì sao FullContext-8K thấp, và khi nào nó lại tốt?

LoCoMo có **5.882 lượt hội thoại**; FullContext-8K chỉ giữ **1.515 lượt (25,8%)** trên 10 hội thoại, từ 21,9% đến 43,9% mỗi hội thoại. Mọi hội thoại đều bị cắt giữa: giữ 5 lượt đầu và phần cuối gần nhất. Prompt thực tế trung bình khoảng **7.879 token**, sát giới hạn 8K; cả 1.986 câu đều dừng bình thường, không phải lỗi tràn output. SimRetrieval dùng trung bình khoảng **1.112 prompt token**, NoMemory khoảng **68**. Số token CARE theo từng lần gọi không có trong bộ kết quả tải về, nên không tính tổng token tương đương cho CARE.

Đối chiếu mã nguồn vàng sau suy luận cho **1.527 câu có đáp án và chú thích nguồn hợp lệ** (9 câu có mã nguồn vàng sai định dạng/không tồn tại, 4 câu không có chú thích):

| Nguồn vàng còn trong context 8K | Số câu | CARE F1 | FullContext-8K F1 | SimRetrieval F1 |
|---|---:|---:|---:|---:|
| Giữ **đủ** nguồn | 337 (22,1%) | 0,4939 | **0,5926** | 0,3344 |
| Giữ **một phần** | 130 (8,5%) | 0,2152 | **0,3242** | 0,1941 |
| **Không giữ** nguồn nào | 1.060 (69,4%) | **0,4892** | 0,1191 | 0,3211 |

Trên nhóm giữ đủ nguồn, FullContext-8K hơn CARE **0,0987 F1**; khoảng bootstrap ghép theo hội thoại cho *CARE trừ FullContext* là `[-0,1378; -0,0620]`. Khi không giữ nguồn nào, CARE hơn **0,3701 F1**, khoảng `[+0,3309; +0,4079]`. Đây là bằng chứng trực tiếp rằng hạn chế context là một nguyên nhân lớn của điểm FullContext-8K thấp. Chú thích nguồn vàng có thể không bao quát mọi cách trả lời; nhóm “không giữ nguồn” chỉ nói không có **nguồn được chú thích**, không khẳng định mọi thông tin liên quan đều vắng mặt.

**Kết luận đúng phạm vi:** CARE vượt FullContext **bị giới hạn 8K** vì còn truy cập được các lượt cũ. Khi nguồn đã nằm trong cửa sổ, đọc ngữ cảnh trực tiếp trong lần chạy này lại có ưu thế. Chưa có phép chạy Qwen3-8B đọc trọn lịch sử với cùng hợp đồng trả lời để khẳng định CARE hơn một FullContext dài thực sự.

## 6. Đặt cạnh các lần chạy trước

Các phương pháp cũ dùng cùng bộ dữ liệu và cùng revision Qwen3-8B, nhưng chỉ có tập con hoàn tất; các con số dưới đây **ghép đúng câu hỏi**, không phải điểm full của phương pháp cũ:

| Đối chứng cũ | n ghép cặp | Điểm cũ | CARE trên chính tập đó | Chênh lệch |
|---|---:|---:|---:|---:|
| Graph memory | 746 | 0,3786 | 0,5188 | +0,1402 |
| Source-hybrid | 746 | 0,3467 | 0,5188 | +0,1721 |
| BM25 RAG | 1.128 | 0,1372 | 0,4912 | +0,3540 |

Khoảng bootstrap 95% của chênh lệch CARE trên từng tập lần lượt là graph `[+0,0970; +0,1826]`, source-hybrid `[+0,1295; +0,2077]` và BM25 `[+0,3103; +0,3987]`. Khi chấm lại BERTScore F1 thô bằng cùng mô hình chấm, graph đạt **0,8310** so với CARE **0,9083**; source-hybrid **0,8428** so với **0,9083**; BM25 **0,7979** so với **0,9093** trên từng tập ghép cặp.

Trên 156 câu loại 5 của tập graph cũ, graph đạt **81,4%** từ chối đúng, CARE **71,2%**; source-hybrid đạt **87,2%**. Graph cũ cũng hơn CARE ở loại 1 trên 102 câu ghép cặp (0,2616 so với 0,2418). CARE cải thiện tổng thể trên các tập này nhưng không thống trị mọi loại câu. Pipeline cũ khác prompt, bộ nhớ, budget và câu từ chối nên đây không phải ablation cô lập thành phần.

Một báo cáo **Qwen3-4B-Instruct-2507** cũ trên 1.986 câu ghi FullContext F1 **0,3485**, NoMemory **0,0678**; còn ASEM **0,2372**, FastASEM **0,2821**. Kết quả đó dùng mô hình và cách chấm/định dạng khác, đặc biệt loại 5 không theo cùng điểm từ chối nhị phân hiện tại; không được xếp chung thứ hạng hay lấy chênh lệch số học với bảng Qwen3-8B. Một FullContext Qwen3-8B cũ khác bị chặn bởi giới hạn evidence 4.096 token; điểm 0 của lần ấy là lỗi giao thức, không phải đánh giá chất lượng.

## 7. Bức tranh tổng quát và điều còn thiếu

CARE hiện là **pipeline chọn bằng chứng và trả lời có kiểm tra nguồn** hiệu quả trên hội thoại dài: nó vượt NoMemory, SimRetrieval lượt gốc, FullContext-8K và các tập con graph/RAG cũ theo điểm tổng hợp. Lợi thế với SimRetrieval chủ yếu nằm ở câu có đáp án và đi cùng recall nguồn cao hơn, đổi lại thêm khoảng 1.004 lượt gọi mô hình. Lợi thế với FullContext-8K chịu ảnh hưởng rất lớn từ việc cắt mất nguồn đáp án; không nên diễn giải nó thành chiến thắng trước mọi phương pháp long-context.

Để chứng minh đóng góp học thuật chặt chẽ hơn, ưu tiên tiếp theo là: (1) Qwen3-8B FullContext đủ dài để chứa mọi lượt, cùng câu lệnh từ chối và bộ chấm; (2) đối chứng cùng reader/prompt, chỉ thay cơ chế truy hồi của CARE và dense similarity để tách lợi ích retrieval; (3) kiểm tra trên hội thoại mới chưa dùng trong phát triển; (4) sửa lỗi loại 3, câu liệt kê loại 1 và các trường hợp từ chối sai. Kết quả hiện tại chưa chứng minh phần **self-evolution của memory** là nguyên nhân tạo điểm tốt, vì CARE full run đọc và kiểm tra lượt gốc bất biến.

## Nguồn nội bộ

- [Bảng ghép bốn phương án và BERTScore](ASEM_CARE_four_way_20261112.md)
- [Kết quả CARE Bridge v2 và Relation+Temporal](ASEM_CARE_full_20261112.md)
- [Đối chứng graph/RAG cũ, cùng câu hỏi](ASEM_CARE_vs_previous_20261112.md)
- [BERTScore của graph/RAG cũ](ASEM_CARE_vs_previous_20261112_bertscore.md)
- [Báo cáo Qwen3-4B cũ](../data/benchmarks/results/static/locomo10/ds_fixed__qwen_qwen3_4b_instruct_2507.md)
- [Dữ liệu phân tích context và bootstrap](ASEM_CARE_overview_20261112_data.json)
