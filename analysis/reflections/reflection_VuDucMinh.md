# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Vũ Đức Minh  
**MSSV:** 2A202602895  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng vào code

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | MiniLM mã hóa từng câu và cosine similarity với threshold 0.85 quyết định gộp nhóm. Hierarchical chunking tạo 100 child chunks từ 26 tài liệu; baseline paragraph chunking tạo 57 chunks. Parent ID được gắn với source để trả lại đúng văn bản cha. |
| Hybrid retrieval, BM25 + Dense + RRF | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 bắt từ khóa tiếng Việt sau khi tách từ; dense search dùng `BAAI/bge-m3`; RRF hợp nhất thứ hạng. Pipeline chạy được với Qdrant in-memory khi dịch vụ local không sẵn sàng. Context Precision tăng từ 0.9333 lên 1.0000, nhưng câu hỏi nhiều chủ đề vẫn có thể thiếu tài liệu cần thiết. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` nạp một lần; model load đo được 7.91 giây, inference rerank tổng 7.17 giây trên 20 query (~0.36 giây/query). Giữ child có thứ hạng hybrid/RRF đầu tiên cho mỗi parent trước khi CrossEncoder xếp hạng giúp kết quả cuối dành chỗ cho nhiều tài liệu cha. |
| RAGAS 4 metrics + Error Tree | M4 | `evaluate_ragas()`, `failure_analysis()` | Production đạt Faithfulness 0.9200, Answer Relevancy 0.8836, Context Precision 1.0000, Context Recall 0.8583. Baseline tương ứng là 0.7879, 0.7180, 0.9333, 0.9250. Precision tăng nhưng recall giảm; phân tích từng câu giúp thấy thiếu evidence ở câu hỏi nhiều ý và câu hỏi so sánh lịch sử. |
| Enrichment trước khi embed | M5 | `_enrich_single_call()`, `enrich_chunks()` | Combined mode gọi `gpt-4o-mini` một lần cho mỗi child để sinh summary, câu hỏi giả thuyết, context và metadata. 100 chunks mất 243.70 giây ở lượt đo cuối. Fallback cục bộ giữ cho ingestion tiếp tục khi thiếu key hoặc API/JSON lỗi. |

Production so với baseline tăng Faithfulness **+0.1321**, Answer Relevancy **+0.1656**, Context Precision **+0.0667**; Context Recall thay đổi **-0.0667**. Kết quả cho thấy reranking và enrichment làm evidence được chọn chính xác hơn, nhưng pipeline vẫn cần xử lý tốt compound query và độ phủ tài liệu.

## Phần 2: Khó khăn và cách giải quyết

- **Lỗi kỹ thuật gặp phải (exact error):** `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`.
- **Nguyên nhân và cách debug:** RAGAS đã tính xong 80 metric task, nhưng sau đó `to_pandas()` trả cột `contexts` dưới dạng mảng NumPy. M4 dùng `row.get("contexts", ...) or []`; Python phải ép mảng thành boolean và ném lỗi. Tôi tái tạo lỗi bằng test mock DataFrame chứa `numpy.ndarray`, rồi sửa M4 để chuyển list-like cell sang list mà không truth-test. Test hồi quy chuyển từ đỏ sang xanh; sau đó chạy lại evaluator thật trên đủ 20 câu.
- **Vấn đề model/network:** Lần đầu Hugging Face báo `[Errno -3] Temporary failure in name resolution`; model cache cho phép một số bước tiếp tục, và lần chạy mạng được duyệt hoàn tất. Hai PDF bị loader bỏ qua với cảnh báo scan không có text layer; cần OCR nếu muốn index nội dung đó.
- **Tối ưu thời gian kiểm thử:** Test M1 ban đầu mất 196.50 giây, dài hơn timeout 120 giây của checker. Cache MiniLM làm suite giảm xuống khoảng 100–104 giây. Cache CrossEncoder theo model name tránh nạp lại cùng model giữa các test.
- **Kiến thức cần bổ sung:** RAGAS đo tính entailment dựa trên context được truyền vào, không chỉ đối chiếu đáp án có vẻ hợp lý. Chính sách tạm ứng nói phí 2%/tháng nhưng không nói rõ pro-rata theo ngày; tài liệu và ground truth cần thống nhất trước khi kỳ vọng model trả phép tính 50.000 VNĐ.

## Phần 3: Action Plan cho project cá nhân

### Project: Internal Policy QA Assistant

#### 1. Hiện trạng

- **Pipeline hiện tại:** proof-of-concept dùng tài liệu Markdown/PDF, chunking parent-child, hybrid BM25+dense, CrossEncoder và LLM trả lời theo context.
- **Known issues:** tài liệu PDF scan chưa OCR; compound query có thể bỏ thiếu một chủ đề; ground truth lịch sử đôi khi rộng hơn câu hỏi; enrichment 100 chunks mất khoảng 244 giây.

#### 2. Kế hoạch cải tiến

1. **Chunking:** dùng structure-aware chunking cho Markdown theo heading và parent-child cho nội dung dài; giữ version, effective date, source và ACL trong metadata.
2. **Search:** giữ BM25 + dense + RRF; tách query nhiều ý thành các sub-query rồi hợp nhất parent IDs để tăng recall.
3. **Reranking:** dùng CrossEncoder top-20 → top-3 parent đa dạng; đo riêng cold-start model load và inference latency.
4. **Evaluation:** xây bộ câu hỏi có version conflict, phủ định, nhiều bước và câu hỏi không có đáp án; chạy RAGAS offline mỗi lần thay đổi retrieval, kiểm tra thủ công Bottom-5.
5. **Enrichment:** giữ combined mode một call/chunk, thêm cache theo hash nội dung và version để tránh gọi lại khi tài liệu không đổi; không dùng generated summary thay văn bản nguồn làm evidence trả lời.

#### 3. Timeline hai tuần

- **Tuần 1:** chuẩn hóa nguồn Markdown/PDF, OCR tài liệu scan được phép xử lý, thêm metadata version/ACL; xây test cho current-vs-historical và compound query; lưu enrichment theo content hash.
- **Tuần 2:** chạy benchmark retrieval và RAGAS trên tập câu hỏi được review; kiểm tra quyền truy cập theo nhóm người dùng; thêm dashboard cho faithfulness, recall và p95 latency; pilot với một nhóm nhỏ, sửa các lỗi Bottom-5 trước khi mở rộng.
