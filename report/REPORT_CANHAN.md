# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Lê Hoàng Đạt  
**MSSV:** 2A202602583  
**Nhóm:** 52  
**Ngày:** 19/09/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (tiến gần về 1.0) biểu thị hai vector embedding cùng chỉ về một hướng trong không gian đa chiều, phản ánh hai đoạn văn bản có sự tương đồng lớn về mặt ngữ nghĩa và chủ đề, không phụ thuộc vào độ dài hay số lượng từ ngữ của mỗi đoạn.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên nộp đơn xin xét học bổng khuyến khích học tập tại phòng Công tác Sinh viên."
- Câu B: "Hồ sơ đăng ký học bổng học tập xuất sắc được tiếp nhận bởi văn phòng phụ trách công tác sinh viên."
- Tại sao tương đồng: Cả hai câu sử dụng các từ vựng và cấu trúc ngữ pháp khác nhau nhưng cùng truyền tải trọn vẹn một nội dung ngữ nghĩa: thủ tục và địa điểm tiếp nhận hồ sơ học bổng cho sinh viên.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Sinh viên nộp đơn xin xét học bổng khuyến khích học tập tại phòng Công tác Sinh viên."
- Câu B: "Thực đơn trưa nay tại nhà ăn trường bao gồm cơm gà xối mỡ và canh chua cá lóc."
- Tại sao khác: Hai câu thuộc hai miền chủ đề hoàn toàn tách biệt (thủ tục học vụ - học bổng và ẩm thực/đời sống hàng ngày), không có mối liên hệ ngữ nghĩa nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid đo khoảng cách tuyệt đối giữa hai đầu mút vector nên bị chi phối nặng nề bởi độ lớn vector (độ dài văn bản); trong khi cosine similarity chỉ đo góc lệch giữa hai vector (hướng biểu diễn ngữ nghĩa), giúp chuẩn hóa và loại trừ hoàn toàn ảnh hưởng của độ dài văn bản lên kết quả so khớp.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*
> Áp dụng công thức: `số lượng chunk = làm_tròn_lên((độ_dài_tài_liệu - độ_chồng_chéo) / (kích_thước_chunk - độ_chồng_chéo))`  
> Số lượng chunk = $\lceil (10.000 - 50) / (500 - 50) \rceil = \lceil 9.950 / 450 \rceil = \lceil 22.111... \rceil = 23$ chunks.  
> *(Kiểm chứng bằng code: `step = 500 - 50 = 450`, vòng lặp chạy qua các vị trí bắt đầu `0, 450, 900, ..., 9900` tạo đúng 23 chunks).*  
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, số lượng chunk sẽ là $\lceil (10.000 - 100) / (500 - 100) \rceil = \lceil 9.900 / 400 \rceil = 25$ chunks (tăng thêm 2 chunks). Ta muốn tăng độ chồng chéo để bảo tồn ngữ cảnh liền mạch tại ranh giới cắt giữa hai chunk (boundary context loss), đảm bảo các thực thể, con số hoặc điều kiện quan trọng không bị chia cắt làm đôi, giúp mô hình embedding nắm bắt ngữ nghĩa trọn vẹn hơn.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy với kỹ thuật positive lookbehind `(?<=[.!?])\s+` để phân tách câu tại các ký tự kết thúc câu (`. `, `! `, `? `, `.\n`) mà không làm mất dấu câu ở cuối câu. Sau khi strip khoảng trắng, gom tối đa `max_sentences_per_chunk` câu vào mỗi chunk. Trường hợp ngoại lệ (edge case): chuỗi rỗng hoặc toàn khoảng trắng trả về `[]`; các chữ viết tắt (`TS.`, `v.v.`) hoặc số thập phân có thể bị regex coi nhầm là ranh giới câu, đây là giới hạn của phương pháp rule-based regex.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán hoạt động theo mô hình chia để trị đệ quy kết hợp gom nhóm (Top-down Recursive Splitting with Bottom-up Merging). Thuật toán thử phân tách văn bản theo danh sách dấu phân tách ưu tiên `["\n\n", "\n", ". ", " ", ""]`; nếu đoạn văn bản con vượt quá `chunk_size`, nó đệ quy gọi `_split` với danh sách phân tách còn lại, sau đó gộp các đoạn nhỏ liền kề cho đến khi chạm sát ngưỡng `chunk_size` để tránh sinh ra các chunk vụn. Base cases gồm: văn bản rỗng, văn bản $\le chunk\_size$, hoặc danh sách separator đã hết (khi đó fallback cắt cứng theo `chunk_size`).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ dữ liệu dạng danh sách bản ghi in-memory (`self._store`), mỗi bản ghi chứa `id`, `content`, bản sao `metadata` (đảm bảo luôn có khóa `doc_id`), và vector `embedding`. Phương thức `search` nhúng truy vấn qua `self._embedding_fn`, sau đó gọi hàm trợ giúp `_search_records` tính tích vô hướng (dot-product) giữa vector query và từng chunk đã chuẩn hóa độ dài, sắp xếp điểm giảm dần và trả về top-k bản ghi (đã lược bỏ trường embedding để output tinh gọn).

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` bắt buộc thực hiện **lọc trước (pre-filtering)**: lọc danh sách các bản ghi trong `self._store` khớp với toàn bộ các cặp key-value trong `metadata_filter` trước khi tính similarity search; cách này đảm bảo kết quả không liên quan không chiếm dụng k-slot của tài liệu thỏa điều kiện. `delete_document` lọc bỏ tất cả các chunk có `id` hoặc `metadata['doc_id']` trùng với `doc_id` được yêu cầu, so sánh độ dài trước và sau để trả về `True`/`False`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Nhận câu hỏi, gọi `store.search(question, top_k=top_k)` để truy xuất ngữ cảnh liên quan. Ngữ cảnh được định dạng thành các khối có đánh số thứ tự trích dẫn `[1]`, `[2]`, ... kèm tên file nguồn gốc (source provenance), sau đó ghép vào prompt yêu cầu LLM trả lời nghiêm ngặt dựa trên ngữ cảnh và trích dẫn số thứ tự nguồn để chống bịa đặt (hallucination). Nếu không tìm thấy kết quả nào trong store, agent trả về thông báo lỗi rõ ràng thay vì gọi LLM vô ích.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\Admin\K4-DAY07-LeHoangDat-2A202602583
plugins: anyio-4.15.1
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.25s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên nộp đơn xin xét học bổng tại phòng Công tác Sinh viên. | Hồ sơ học bổng của sinh viên được tiếp nhận tại phòng Công tác Sinh viên. | cao | 0.1241 | Không (do mock) |
| 2 | Quy định tiêu chuẩn điểm trung bình tích lũy GPA để đạt học bổng xuất sắc. | Điều kiện điểm rèn luyện và kết quả học tập để nhận khen thưởng khuyến khích. | cao | 0.0708 | Không (do mock) |
| 3 | Hạn nộp học phí học kỳ 1 kết thúc vào ngày 15 tháng 10. | Hạn nộp học phí học kỳ 1 kết thúc vào ngày 15 tháng 10. | cao | 1.0000 | Đúng |
| 4 | Sinh viên có thể mượn tối đa năm quyển sách tại thư viện trường. | Quy định đăng ký đề tài nghiên cứu khoa học dành cho giảng viên. | thấp | 0.0972 | Đúng |
| 5 | Thực đơn bữa trưa tại nhà ăn sinh viên gồm cơm gà và rau xào. | Quy trình phúc khảo điểm thi kết thúc học phần của sinh viên. | thấp | 0.0072 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả bất ngờ nhất xuất hiện ở Cặp 1 và Cặp 2: về mặt ngữ nghĩa đời thực, hai câu có ý nghĩa gần như tương đương nhưng điểm tương tự thực tế lại rất thấp (chỉ 0.07 – 0.12), thậm chí Cặp 4 khác chủ đề lại có điểm tương tự (0.0972) cao hơn Cặp 2. Điều này giải thích rõ cơ chế của `MockEmbedder`: nó chỉ băm chuỗi ký tự bằng hàm MD5 để sinh vector ngẫu nhiên giả định chứ hoàn toàn không nắm bắt được ngữ nghĩa từ vựng. Để vector embeddings thực sự phản ánh ngữ nghĩa, ta bắt buộc phải sử dụng các mô hình ngôn ngữ được huấn luyện chuyên biệt (như Sentence-Transformers hoặc OpenAI/Gemini Embeddings).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src` (chủ đề Quy định học bổng đại học).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên cần đạt GPA và điểm rèn luyện tối thiểu là bao nhiêu để nhận học bổng Khuyến khích học tập loại Xuất sắc? | `scholarship-exchange-student`: Điều kiện ứng tuyển hoàn thành tối thiểu 30 tín chỉ... | 0.3722 | Không trực tiếp | [LLM Answer] Trích xuất từ tài liệu trao đổi sinh viên, thiếu tiêu chuẩn GPA 3.60 của học bổng KKHT. |
| 2 | Mức tài trợ của học bổng Doanh nghiệp và Hỗ trợ tài chính vượt khó là bao nhiêu tiền mỗi học kỳ? | `scholarship-exchange-student`: Quyền lợi học bổng tài trợ sinh hoạt phí 800 USD/tháng... | 0.2344 | Không | [LLM Answer] Trả lời nhầm sang mức trợ cấp trao đổi quốc tế thay vì mức 15.000.000 VNĐ. |
| 3 | Quy định nộp hồ sơ xin cấp học bổng tại phòng nào? *(Có filter `audience: student`)* | `scholarship-merit-student`: Nộp đơn điều chỉnh kèm bảng điểm về Phòng Công tác Sinh viên (CTSV)... | 0.2115 | Có | [LLM Answer] Hướng dẫn sinh viên nộp hồ sơ về Phòng Công tác Sinh viên (CTSV). |
| 4 | Sinh viên tham gia chương trình học bổng Trao đổi quốc tế cần chứng chỉ tiếng Anh IELTS tối thiểu bao nhiêu? | `scholarship-exchange-student`: Yêu cầu chứng chỉ tiếng Anh IELTS tối thiểu đạt 6.5... | 0.1933 | Có | [LLM Answer] Trả lời chính xác yêu cầu chứng chỉ IELTS 6.5 (hoặc TOEFL iBT 79). |
| 5 | Điều kiện GPA và điểm rèn luyện để duy trì học bổng toàn phần đối với tân sinh viên thủ khoa qua từng năm học là gì? | `scholarship-council-procedure`: Quy trình thẩm định và họp hội đồng xét duyệt... | 0.2131 | Không ở Top-1 (Top-3 đạt `valedictorian`) | [LLM Answer] Trả lời khái quát quy trình họp hội đồng, có đề cập điều kiện duy trì ở ngữ cảnh trích dẫn [3]. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 câu hỏi (với chiến lược Recursive Chunking và Sentence Chunking).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Tôi học được rằng việc tiền lọc siêu dữ liệu (Metadata Pre-filtering) đóng vai trò quyết định trong việc loại bỏ nhiễu giữa các nhóm đối tượng (`student` và `faculty`); nếu không có bộ lọc `audience`, tài liệu nghiên cứu của giảng viên có thể lấn át và đẩy quy định của sinh viên ra khỏi top-k. Ngoài ra, việc chia nhỏ theo tiêu đề mục (Heading-based chunking) giúp các đoạn văn bản giữ trọn vẹn ngữ cảnh của từng điều khoản học bổng tốt hơn nhiều so với chia nhỏ theo kích thước ký tự cố định.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |



