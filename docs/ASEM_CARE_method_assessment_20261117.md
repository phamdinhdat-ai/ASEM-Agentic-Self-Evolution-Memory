# Đánh giá phương pháp ASEM/CARE và kiến trúc hiện tại (17/11/2026)

Tài liệu đánh giá trung thực trạng thái phương pháp tại thời điểm lượt chạy giai thừa đang diễn ra trên Kaggle. Mọi con số đều lấy từ file trong `analysis/`. Phần đánh giá thiết kế thực nghiệm dùng khung của skill `anthropic-evaluations` (`~/.agents/skills/anthropic-evaluations`).

## 1. Hình kiến trúc

| File | Nội dung | Ghi chú |
|---|---|---|
| `paper_asem/figures/asem_architecture_v4.drawio` | Sơ đồ draw.io (skill `drawio-diagram`), mở và chỉnh trong draw.io | Có chú giải, thanh hạ tầng, ghi chú bằng chứng |
| `paper_asem/figures/asem_architecture_v4.drawio.html` | Xem trong trình duyệt, không cần cài draw.io | Tải trình xem từ viewer.diagrams.net |
| `paper_asem/figures/asem_arch_v4.pdf`, `.png` | Bản TikZ cùng nội dung, dùng cho bài báo | Đã kiểm tra bằng mắt |
| `paper_asem/figures/make_architecture_drawio.py` | Script tạo sơ đồ, kiểm tra XML và chồng lấn | Chạy lại khi số liệu thay đổi |

Màu và nét thể hiện **trạng thái bằng chứng**, không chỉ chức năng: xanh lá là đã đánh giá end-to-end; tím là đã code và test, đang đánh giá; tím nét đứt là mới đặc tả.

## 2. Kiến trúc trong một đoạn

Bộ nhớ có hai lớp. **Archive** chỉ được thêm vào và là nguồn trích dẫn duy nhất. **Lớp dẫn xuất** gồm các ghi chú có phiên bản, chỉ thay đổi qua toán tử có cổng kiểm tra bằng code. Trên archive, mỗi lịch sử có một **đồ thị 4 lớp** (turn, thực thể, session, community), xây không cần LLM, với độ ảnh hưởng tính bằng Personalized PageRank. **Đường đọc** là CARE: bundle đầu (BM25 hoặc dense), reader Qwen3-8B trả JSON có trích dẫn và search query, kiểm tra bằng code, gate quyết định repair, và repair mở rộng bằng chứng theo ba cách (lexical, lexical + dense, lexical + đồ thị), tối đa hai lượt gọi.

## 3. Trạng thái từng thành phần

| Thành phần | Trạng thái | Bằng chứng |
|---|---|---|
| Archive, reader, kiểm tra, gate, repair lexical | Đã đánh giá | LoCoMo 1.986 câu và LongMemEval 500 câu |
| Bundle đầu dense (Qwen3-Emb-0.6B) | Đã đánh giá | +0,014 so với BM25 ở first pass (9/10 hội thoại) |
| Repair hybrid (dense) | Đang chạy (giai thừa) | Chưa có |
| Đồ thị 4 lớp + PPR | Recall trên dev | k=8: +0,001; k=18: +0,011 so với dense |
| Repair đồ thị (`dense_graph`) | Đang chạy (giai thừa) | Chưa có (H4) |
| Toán tử ADD/UPDATE/LINK | Code + 15 unit test | Chưa nối vào pipeline |
| SPLIT/MERGE/PRUNE/REORG, toán tử tổng hợp, ghi chú làm nút đồ thị | Mới đặc tả | Không có |
| Verifier chấp nhận repair | Đã đánh giá, **âm tính** | 6 cấu hình, α/β tốt nhất 2,31 < 3,89, đều lỗ |

## 4. Đánh giá phương pháp

### Điểm mạnh

1. **Truy vết được.** Mọi câu trả lời trích dẫn turn gốc; mọi ghi chú bám nguồn theo chuỗi con. Ba tính chất này được chứng minh (Mệnh đề 1–3) và có test.
2. **Chi phí bị chặn.** Tối đa hai lượt gọi reader mỗi câu; giới hạn bằng chứng 10.000 ký tự.
3. **Tái lập chính xác.** Chạy lại arm CARE cho ra câu trả lời giống hệt 98/98 (LongMemEval) và 4/4 (LoCoMo pilot).
4. **Lý thuyết có sức dự đoán.** Định lý hòa vốn dự đoán đúng dấu cho cả 6 verifier.
5. **Chỉ dùng mô hình mở**, phiên bản được ghim.

### Điểm yếu

1. **Đồ thị hiện tại mang ít thông tin mới.** Cạnh ngữ nghĩa 5-NN lặp lại tín hiệu dense; thực thể lấy theo chữ hoa là xấp xỉ thô của NER. Kết quả: gần như không giúp ở k=8; chọn theo community còn tệ hơn hẳn (0,415 so với 0,617). Đồ thị chưa có **quan hệ có kiểu lấy từ văn bản** ("X mua Y", "X chuyển tới Z").
2. **Chưa tự tiến hóa.** Toán tử ghi tồn tại như thư viện, chưa nối vào pipeline. Bài không thể khẳng định "self-evolving".
3. **Nút thắt reader chưa được xử lý.** Multi-session và temporal lỗi ở bước đếm, cộng, xếp thứ tự (probe: recall 0,74 → 0,93 mà F1 không đổi). Toán tử tổng hợp mới là đặc tả.
4. **Đánh đổi abstention.** Repair luôn đổi refusal đúng lấy câu trả lời; 6 verifier không gỡ được (ngưỡng 3,89 là thuộc tính của dữ liệu).
5. **Cắt bundle repair theo thứ tự hội thoại** có thể bỏ turn ở session sau (recall LongMemEval giảm sau repair: multi-session 0,917 → 0,835). Chưa tách được nguyên nhân.
6. **Lợi ích bundle đầu tốt hơn bị thu hẹp sau repair** (+0,014 → −0,009), có thể vì repair được thiết kế quanh lexical. Lượt chạy giai thừa đang kiểm tra trực tiếp điều này.

## 5. Đánh giá thiết kế thực nghiệm theo khung `anthropic-evaluations`

| Bước / nguyên tắc | Hiện trạng | Đánh giá |
|---|---|---|
| **Grader dạng code** (string match, F1) | Toàn bộ điểm là F1 token, cụm từ refusal, F1 chẩn đoán | Nhanh, tái lập được. Nhưng **giòn với biến thể hợp lệ**: "4" so với "four movie festivals", và loại preference không chấm được (F1 ≈ 0 vì đáp án là rubric) |
| **Grader dạng model** | Chưa có | Cần LLM-judge với rubric từng chiều và nhãn **"Unknown"**, đúng như skill khuyến nghị. Đây cũng là cách so được với các số LongMemEval đã công bố |
| **Grader dạng người**, hiệu chỉnh | Hàng đợi 100 câu đã đóng băng nhưng chưa gán nhãn | Chưa hiệu chỉnh được grader nào. Cần gán nhãn trước khi tin vào LLM-judge |
| **Bài toán cân bằng** (nên và không nên làm) | Có: câu có đáp án và câu tiền đề sai (446 LoCoMo, 30 LongMemEval) | Tốt. LongMemEval chỉ có 30 câu abstention, nên ước lượng refusal rất nhiễu |
| **Bài toán rõ ràng, có lời giải tham chiếu** | Có đáp án vàng, nhưng LoCoMo được báo có lỗi nhãn và nhầm tên category | Cần đối chiếu báo cáo audit LoCoMo và chạy phân tích độ nhạy |
| **Harness sạch, không chia sẻ trạng thái** | Mỗi kernel một môi trường mới, temperature 0, `max-num-seqs 1`, phiên bản ghim | Tốt. Tính tất định đã được kiểm chứng (98/98) |
| **Đọc transcript** | Mới đọc thủ công vài ví dụ (probe, verifier) | Thiếu. Cần đọc có hệ thống một mẫu lỗi theo loại để biết lỗi do hệ thống hay do grader |
| **Capability và regression** | CARE (`bm25_lex`) tái lập chính xác | Có thể dùng làm **bộ regression**: mọi thay đổi code phải giữ nguyên câu trả lời của arm này |
| **Không tất định, pass@k / pass^k** | Một lần chạy, greedy | Hợp lý với decoding tất định; không cần pass^k |
| **Bão hòa** | Điểm khoảng 0,5 | Chưa bão hòa; còn nhiều chỗ cải thiện |
| **Chỉ số theo dõi** (token, lượt gọi, độ trễ) | Có token và lượt gọi theo từng arm; thời gian chỉ ghi một phần | Đủ cho phân tích chi phí; độ trễ end-to-end chưa đo chặt |
| **Lấy tập test từ lỗi thực tế** | Dev = LoCoMo, test = LongMemEval-oracle; protocol chốt trước | Tốt về quy trình. Nhưng oracle cung cấp sẵn session liên quan nên chủ yếu đo reader, không đo retrieval |

**Tóm lại:** harness và quy trình thống kê đạt yêu cầu. Khoảng trống lớn nhất là **chỉ có grader dạng code**. Nó vừa giòn, vừa không chấm được loại preference, vừa làm số liệu không so được với các bài công bố (vốn dùng LLM-judge). Bước tiếp theo đáng làm nhất về mặt đánh giá là: gán nhãn người cho 100 câu, xây LLM-judge mở (ví dụ Qwen3-8B ở chế độ thinking) có nhãn "Unknown", hiệu chỉnh judge với nhãn người, rồi chấm lại toàn bộ.

## 6. Khuyến nghị theo thứ tự ưu tiên

1. **Chờ kết quả giai thừa** (H1–H4) và báo cáo mọi arm, kể cả khi đồ thị không giúp.
2. **Grader thứ hai:** LLM-judge mở có nhãn "Unknown", hiệu chỉnh với 100 nhãn người.
3. **Đồ thị có quan hệ thật:** trích xuất bộ ba (chủ thể, quan hệ, đối tượng, thời gian) bằng Qwen3 qua toán tử ADD có cổng kiểm tra, đưa ghi chú làm nút L1. Đây cũng là bước nối đường ghi vào pipeline (self-evolving).
4. **Toán tử tổng hợp** cho multi-session, đo theo Mệnh đề 6 (bỏ sót, trùng, hiểu sai).
5. **LongMemEval-S** (lịch sử đầy đủ) để đo retrieval thật sự.
6. **Đọc transcript có hệ thống** cho 50 lỗi mỗi loại.
