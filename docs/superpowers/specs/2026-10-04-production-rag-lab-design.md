# Đặc tả hoàn thiện Lab 18 Production RAG

## Mục tiêu

Hoàn thiện bài Lab 18 trong repo hiện tại theo `ASSIGNMENT.md` và `RUBRIC.md`: triển khai đủ năm module Production RAG, chạy đánh giá trên bộ 20 câu hỏi, tạo báo cáo có thể kiểm tra, phân tích năm lỗi kém nhất và hoàn tất reflection cá nhân. Mục tiêu là đáp ứng toàn bộ tiêu chí 100 điểm và các bonus có thể đạt được bằng pipeline hiện có.

## Hiện trạng và ràng buộc

- Người dùng đã hoàn tất bước baseline và đã cấu hình `OPENAI_API_KEY` trong `.env`; không đọc hoặc ghi khóa ra log, report hay tài liệu.
- Repo có scaffold cho M1–M5, pipeline và các unit test. Baseline report hiện có 0 câu hỏi và 0 điểm, phù hợp với trạng thái trước khi hoàn thiện M2/M4; chạy `main.py` sẽ cập nhật baseline để so sánh.
- Môi trường dự án là Python 3.12 trong `.venv`; các dependency trong `requirements.txt` đã có sẵn. Phiên làm việc này không truy cập được Docker socket, còn `DenseSearch` đã có Qdrant in-memory fallback.
- Giữ API và tên hàm scaffold trừ khi có lý do cần thiết để ghép pipeline hoặc đáp ứng đúng cấu trúc báo cáo.
- Dùng các ngưỡng trong `config.py`: semantic threshold 0.85, parent 2048 ký tự, child 256 ký tự, RRF k=60, rerank top-3.

## Thiết kế

### M1 — Chunking

- `chunk_semantic()` tách câu bằng dấu kết thúc câu và paragraph, mã hóa bằng `all-MiniLM-L6-v2`, rồi nhóm các câu liên tiếp theo cosine similarity và threshold cấu hình. Văn bản rỗng trả về danh sách rỗng; metadata nguồn được giữ lại.
- `chunk_hierarchical()` tạo parent không vượt quá cấu hình kích thước, chia thành child nhỏ hơn, cấp `parent_id` ổn định và ghi cùng id vào metadata của parent và child. Việc chia phải bảo toàn nội dung, kể cả khi một paragraph dài hơn giới hạn.
- `chunk_structure_aware()` gom nội dung theo các heading Markdown cấp 1–3, giữ heading trong text, gắn tên section vào metadata, và giữ phần mở đầu trước heading đầu tiên.

### M2 — Hybrid Search

- BM25 dùng `underthesea.word_tokenize(..., format="text")`, đổi dấu gạch dưới trong từ ghép thành khoảng trắng trước khi lập chỉ mục và truy vấn.
- Dense search mã hóa bằng `BAAI/bge-m3`, ghi payload có text và metadata vào Qdrant, truy vấn qua `query_points()`. Cùng giao diện hỗ trợ Qdrant local và in-memory.
- BM25 bỏ kết quả có điểm không dương. RRF gộp theo nội dung, cộng `1 / (k + rank + 1)` cho mỗi danh sách và trả về top-k với method `hybrid`.

### M3 — Reranking

- `CrossEncoderReranker` tải `sentence_transformers.CrossEncoder` với `BAAI/bge-reranker-v2-m3`, chấm các cặp query-document, sắp xếp giảm dần và trả top-k `RerankResult` với rank bắt đầu từ 0.
- Danh sách ứng viên rỗng trả về rỗng. Nếu mô hình không tải được, lỗi được ghi rõ; phương án dự phòng cho phép kiểm tra/chạy pipeline offline mà không giả nhận điểm cross-encoder.

### M4 — Evaluation và chẩn đoán

- `evaluate_ragas()` tạo Dataset từ câu hỏi, câu trả lời, contexts và ground truth; chạy bốn metric: faithfulness, answer relevancy, context precision và context recall.
- Kết quả trả về bốn điểm trung bình cùng kết quả từng câu hỏi. Lỗi khởi tạo hoặc gọi RAGAS được báo rõ và trả cấu trúc số liệu an toàn để pipeline lưu được trạng thái lỗi, thay vì tạo report sai hình dạng.
- `failure_analysis()` chọn metric thấp nhất cho mỗi câu, ánh xạ qua Diagnostic Tree, sắp xếp theo điểm trung bình tăng dần và tạo Bottom-5 gồm câu hỏi, metric yếu nhất, chẩn đoán và gợi ý sửa.
- `reports/ragas_report.json` lưu aggregate, số câu hỏi, dữ liệu từng câu và failures theo JSON serializable.

### M5 — Enrichment

- Chế độ mặc định dùng `_enrich_single_call()` để gọi `gpt-4o-mini` một lần cho mỗi chunk và lấy JSON gồm `summary`, `questions`, `context`, `metadata`.
- Nội dung trả về được kiểm tra và chuẩn hóa; lỗi API, JSON sai định dạng hoặc không có khóa dùng fallback cục bộ. `original_text` luôn được giữ.
- Pipeline lập chỉ mục text đã làm giàu (context, summary/câu hỏi giả thuyết nếu có, và văn bản gốc) để tăng khả năng khớp tìm kiếm. LLM trả lời dựa trên văn bản quy chế gốc, không dùng nội dung enrichment làm bằng chứng.

### Ghép pipeline, đo lường và deliverables

Thứ tự runtime là load tài liệu → hierarchical chunking → enrichment → BM25 và dense indexing → hybrid retrieval → cross-encoder rerank → ánh xạ child về parent context → sinh câu trả lời → RAGAS. Metadata nguồn và `parent_id` được giữ qua bước enrichment và indexing. Parent context giúp LLM có đủ nội dung trong khi reranker so sánh các child ngắn.

Pipeline ghi thời gian cho chunking, enrichment, indexing, reranker load, retrieval/rerank/answering và RAGAS vào `reports/latency_report.json`. Chạy `main.py` tạo baseline và production report để so sánh. Cập nhật `analysis/failure_analysis.md` bằng kết quả thật của Bottom-5 và Error Tree walkthrough. Tạo `analysis/reflections/reflection_VuDucMinh.md` theo danh tính trong tên thư mục repo, gồm lecture mapping, các vấn đề kỹ thuật quan sát được và action plan có timeline.

## Xử lý lỗi và bảo toàn dữ liệu

- Giữ nguyên `.env`, không đưa API key vào log hoặc artifact.
- Mỗi bước có fallback phù hợp: Qdrant in-memory khi dịch vụ local không sẵn sàng; enrichment fallback khi API/JSON lỗi; lỗi RAGAS được báo và lưu theo schema thống nhất.
- Không sửa hoặc xóa baseline report hiện có trước khi `main.py` tạo bản đánh giá mới; bảo toàn các thay đổi người dùng chưa commit.

## Kiểm tra hoàn tất

- Chạy các test đã có trong `tests/test_m1.py` đến `tests/test_m5.py` và toàn bộ `tests/` theo yêu cầu rubric.
- Chạy pipeline tích hợp để tạo report thật từ 20 câu hỏi, sau đó xác nhận các artifact có dữ liệu và schema đúng.
- Chạy `check_lab.py`, xác nhận không còn TODO trong module bắt buộc và mọi test được báo pass.
- Kiểm tra báo cáo failure analysis, reflection và latency có nội dung dựa trên kết quả chạy thực tế.

## Ngoài phạm vi

- OCR các PDF scan, triển khai dịch vụ Qdrant riêng ngoài môi trường local, và push repo lên GitHub/LMS.
- Cam kết trước một mức điểm RAGAS cố định; điểm phụ thuộc kết quả truy xuất, mô hình và đánh giá thực tế. Tối ưu pipeline theo rubric, rồi ghi nguyên kết quả đã đo.
