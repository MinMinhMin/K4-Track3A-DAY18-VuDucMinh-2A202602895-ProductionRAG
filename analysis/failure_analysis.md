# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Vũ Đức Minh (`2A202602895`)  
**Khóa:** K4 - Track 3A  
**Evaluation:** 20 câu hỏi, RAGAS 0.1.22

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|--:|
| Faithfulness | 0.7879 | 0.9200 | +0.1321 |
| Answer Relevancy | 0.7180 | 0.8836 | +0.1656 |
| Context Precision | 0.9333 | 1.0000 | +0.0667 |
| Context Recall | 0.9250 | 0.8583 | -0.0667 |

Production improves faithfulness, answer relevancy and precision. Recall falls because some compound or historical questions need evidence from more than the top three parent documents; the current-version policy filter also favors current evidence when the question does not explicitly request history.

## Bottom-5 Failures

### #1 — Tạm ứng quá hạn 5 ngày

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; chính sách quy định phí 2%/tháng trên 15.000.000 VNĐ. Ground truth tính pro-rata khoảng 50.000 VNĐ.
- **Got:** “Phạt cho 5 ngày là: 300.000 VNĐ * (5/30) = 50.000 VNĐ.”
- **Worst metric:** Faithfulness — 0.400; average — 0.738.
- **Error Tree:** Output có phép tính → context có phí 2%/tháng và hạn 15 ngày → context không định nghĩa tính phí theo ngày hoặc tháng 30 ngày → câu trả lời tự thêm quy tắc pro-rata.
- **Root cause:** Ground truth có phép tính pro-rata nhưng tài liệu chính sách chỉ nêu mức theo tháng. Phép tính khớp đáp án kỳ vọng nhưng giả định cách quy đổi không được nêu trong evidence.
- **Suggested fix:** Bổ sung quy tắc tính theo ngày vào chính sách hoặc trả lời đúng phần được chứng minh: “Phí là 2%/tháng trên số tiền chưa hoàn ứng; tài liệu không nêu cách tính phần 5 ngày.” Đồng thời giữ phép tính trong ground truth đồng bộ với chính sách.

### #2 — Lương thử việc Junior

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Lương Junior tối đa 20.000.000 VNĐ/tháng; thử việc nhận 85%, bằng 17.000.000 VNĐ/tháng.
- **Got:** “Lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ (85% của 20.000.000 VNĐ).”
- **Worst metric:** Faithfulness — 0.500; average — 0.831.
- **Error Tree:** Output khớp phép tính kỳ vọng → context có dải lương Junior 12–20 triệu và mức thử việc 85% → cần ghép hai dữ kiện trong hai mục tài liệu → RAGAS chấm faithfulness một phần.
- **Root cause:** Đây là phép suy ra từ bảng lương và chính sách thử việc, không phải một câu trả lời được ghi nguyên văn. Dữ kiện nguồn có đủ để tính nhưng evaluator vẫn đánh dấu một phần nội dung là chưa được entail rõ.
- **Suggested fix:** Ghi ví dụ tính lương thử việc theo cấp bậc ngay trong tài liệu hoặc tách câu trả lời thành hai tiền đề có dẫn nguồn, rồi ghi phép nhân 85% × 20 triệu.

### #3 — Câu hỏi nhiều ý về phép năm và lương Senior

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 18 ngày phép; lương Senior 20–35 triệu VNĐ/tháng.
- **Got:** Trả lời đúng 18 ngày phép nhưng nói “Về lương, không tìm thấy thông tin.”
- **Worst metric:** Context Recall — 0.500; average — 0.846.
- **Error Tree:** Output thiếu phần lương → ba parent context gồm phép năm, nghỉ không lương và nghỉ ốm → không có parent bảng lương → retrieval/reranking không đưa evidence lương vào top-3 → thiếu recall ở M2/M3.
- **Root cause:** Một câu hỏi gộp hai chủ đề. Kết quả phép năm chiếm nhiều vị trí child trong danh sách hybrid trước reranking; tài liệu bảng lương không vào các parent context cuối cùng.
- **Suggested fix:** Phân rã câu hỏi nhiều ý thành các truy vấn “phép năm/thâm niên” và “khung lương Senior”, hợp nhất parent IDs trước reranking, rồi bảo đảm mỗi ý có ít nhất một parent evidence.

### #4 — MFA và phiên bản cũ

- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Bắt buộc MFA cho email, VPN và hệ thống nội bộ; chính sách v1.0 cũ không yêu cầu MFA.
- **Got:** “Có, tất cả nhân viên bắt buộc kích hoạt MFA cho tài khoản email, VPN và các hệ thống nội bộ.”
- **Worst metric:** Context Recall — 0.500; average — 0.848.
- **Error Tree:** Output trả lời câu hỏi hiện hành → context hiện hành ghi rõ MFA bắt buộc và nói văn bản thay thế v1.0 → context không mô tả yêu cầu MFA cụ thể trong v1.0 → phần lịch sử trong ground truth không có evidence truy xuất.
- **Root cause:** Câu hỏi không yêu cầu lịch sử nhưng ground truth có thêm so sánh v1.0; hệ thống ưu tiên context phiên bản hiện hành như thiết kế.
- **Suggested fix:** Đưa nội dung so sánh phiên bản vào tài liệu hiện hành nếu muốn mọi câu trả lời bao gồm lịch sử, hoặc sửa ground truth cho đúng phạm vi câu hỏi. Nếu muốn lịch sử, cần hỏi rõ phiên bản cũ để version-aware retrieval giữ tài liệu đó.

### #5 — Chu kỳ đổi mật khẩu và phiên bản cũ

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Chính sách hiện hành v2.0 yêu cầu 120 ngày; mức cũ 90 ngày đã bị thay thế.
- **Got:** “Mật khẩu phải được thay đổi mỗi 120 ngày.”
- **Worst metric:** Context Recall — 0.500; average — 0.850.
- **Error Tree:** Output trả đúng chu kỳ hiện hành → context v2.0 có 120 ngày và nói thay thế v1.0 → context hiện hành không nêu con số 90 ngày cũ → recall tính cả chi tiết lịch sử trong ground truth.
- **Root cause:** Ground truth yêu cầu lịch sử phiên bản trong khi câu hỏi chỉ hỏi chính sách hiện tại; lọc version giữ tài liệu mới nhất và bỏ bản cũ theo đúng ý định hiện hành.
- **Suggested fix:** Chuẩn hóa ground truth theo câu hỏi hiện tại, hoặc thêm truy vấn lịch sử/version rõ ràng và đánh giá riêng loại câu hỏi đó.

## Case Study — Câu hỏi Senior nhiều ý

**Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

**Error Tree walkthrough:**

1. **Output đủ ý không?** Không. Phần 18 ngày phép đúng nhưng không trả khoảng lương.
2. **Context đúng không?** Context có quy định phép năm hiện hành nhưng không có bảng lương; hai context còn lại nói về nghỉ không lương và nghỉ ốm.
3. **Query xử lý đúng không?** Truy vấn gộp phép năm và lương không được tách thành hai nhu cầu evidence; retrieval/reranking chỉ giữ ba parent không có tài liệu lương.
4. **Fix ở bước nào?** M2/M3: phân rã compound query, hợp nhất candidate parents, rerank evidence đa dạng theo chủ đề rồi mới gửi đủ context cho LLM.

**Nếu có thêm một giờ, sẽ tối ưu:** bổ sung query decomposition cho câu có “và”, kiểm tra coverage từng sub-question trước khi sinh câu trả lời, và thêm một test hồi quy yêu cầu parent lương cùng xuất hiện với parent phép năm.
