"""
Benchmark script for Lab 07 (K4-L3A): University Regulations & Services (Học bổng).
Evaluates retrieval strategies and metadata filtering on 5 benchmark queries.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.agent import KnowledgeBaseAgent
from src.chunking import FixedSizeChunker, RecursiveChunker, SentenceChunker
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore

DATA_DIR = Path("data/university")

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Sinh viên cần đạt GPA và điểm rèn luyện tối thiểu là bao nhiêu để nhận học bổng Khuyến khích học tập loại Xuất sắc?",
        "filter": None,
        "gold_answer": "GPA từ 3.60 trở lên và điểm rèn luyện từ 80 điểm trở lên (nhận 120% học phí).",
        "expected_doc": "scholarship-merit-student",
    },
    {
        "id": 2,
        "query": "Mức tài trợ của học bổng Doanh nghiệp và Hỗ trợ tài chính vượt khó là bao nhiêu tiền mỗi học kỳ?",
        "filter": None,
        "gold_answer": "15.000.000 VNĐ / suất / học kỳ.",
        "expected_doc": "scholarship-sponsor-student",
    },
    {
        "id": 3,
        "query": "Quy định nộp hồ sơ xin cấp học bổng tại phòng nào?",
        "filter": {"audience": "student"},
        "gold_answer": "Sinh viên nộp đơn xác nhận điều chỉnh kèm bảng điểm đã cập nhật về Phòng Công tác Sinh viên (CTSV) trong thời hạn 15 ngày kể từ ngày công bố điểm chính thức.",
        "expected_doc": "scholarship-merit-student",
    },
    {
        "id": 4,
        "query": "Sinh viên tham gia chương trình học bổng Trao đổi quốc tế cần chứng chỉ tiếng Anh IELTS tối thiểu bao nhiêu?",
        "filter": None,
        "gold_answer": "IELTS tối thiểu 6.5 (hoặc TOEFL iBT từ 79 điểm trở lên).",
        "expected_doc": "scholarship-exchange-student",
    },
    {
        "id": 5,
        "query": "Điều kiện GPA và điểm rèn luyện để duy trì học bổng toàn phần đối với tân sinh viên thủ khoa qua từng năm học là gì?",
        "filter": None,
        "gold_answer": "GPA tích lũy kết thúc năm học từ 3.00 trở lên và điểm rèn luyện đạt loại Tốt trở lên (tối thiểu 80 điểm).",
        "expected_doc": "scholarship-valedictorian-student",
    },
]


def load_corpus() -> list[tuple[str, dict, str]]:
    """Loads markdown files, parses frontmatter and returns (stem, metadata, body)."""
    items = []
    for md_file in sorted(DATA_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8").replace("\r\n", "\n")
        parts = text.split("---", 2)
        if len(parts) < 3:
            continue
        fm_text = parts[1]
        body = parts[2].strip()
        metadata = {}
        for line in fm_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                metadata[k.strip()] = v.strip().strip('"').strip("'")
        metadata["doc_id"] = md_file.stem
        metadata["source"] = str(md_file)
        items.append((md_file.stem, metadata, body))
    return items


def build_store(chunker_type: str = "sentence") -> EmbeddingStore:
    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=_mock_embed)
    corpus = load_corpus()

    docs: list[Document] = []
    for stem, meta, body in corpus:
        if chunker_type == "fixed":
            chunker = FixedSizeChunker(chunk_size=300, overlap=50)
            chunks = chunker.chunk(body)
        elif chunker_type == "recursive":
            chunker = RecursiveChunker(chunk_size=300)
            chunks = chunker.chunk(body)
        else:  # sentence
            chunker = SentenceChunker(max_sentences_per_chunk=3)
            chunks = chunker.chunk(body)

        for i, chunk_text in enumerate(chunks):
            doc_id = f"{stem}#{i}"
            docs.append(Document(id=doc_id, content=chunk_text, metadata=meta.copy()))

    store.add_documents(docs)
    return store


def run_benchmark():
    print("=================================================================")
    print("BÁO CÁO ĐÁNH GIÁ TRUY XUẤT (RETRIEVAL BENCHMARK) — K4-L3A")
    print("Chủ đề: Quy định Học bổng Đại học (University Scholarships)")
    print("=================================================================\n")

    strategies = ["sentence", "recursive", "fixed"]

    for strat in strategies:
        print(f"\n#################################################################")
        print(f"CHIẾN LƯỢC: {strat.upper()} CHUNKING")
        print(f"#################################################################")
        store = build_store(strat)
        print(f"Tổng số chunk trong store: {store.get_collection_size()}\n")

        agent = KnowledgeBaseAgent(store=store, llm_fn=lambda p: f"[LLM Answer] {p[:200]}...")

        for item in BENCHMARK_QUERIES:
            q_id = item["id"]
            q_text = item["query"]
            filt = item["filter"]
            gold = item["gold_answer"]
            exp_doc = item["expected_doc"]

            print(f"--- Câu hỏi {q_id} ---")
            print(f"Query: {q_text}")
            print(f"Filter: {filt}")
            print(f"Gold Answer: {gold}")

            if filt:
                results = store.search_with_filter(q_text, top_k=3, metadata_filter=filt)
            else:
                results = store.search(q_text, top_k=3)

            top1_hit = False
            top3_hit = False
            for rank, r in enumerate(results, start=1):
                doc_stem = r["metadata"].get("doc_id", "")
                is_hit = (doc_stem == exp_doc)
                if rank == 1 and is_hit:
                    top1_hit = True
                if is_hit:
                    top3_hit = True
                content_preview = r["content"][:100].replace("\n", " ")
                print(f"  Top {rank}: score={r['score']:.4f} doc={doc_stem} | {content_preview}...")

            status = "HIT_TOP1" if top1_hit else ("HIT_TOP3" if top3_hit else "MISS")
            print(f"Đánh giá: {status}")
            print()

    # A/B Testing on Query 3
    print("=================================================================")
    print("THỰC NGHIỆM A/B: METADATA FILTERING (CÂU HỎI 3)")
    print("=================================================================")
    q3 = BENCHMARK_QUERIES[2]
    store_sent = build_store("sentence")

    print("\n[A] Chạy KHÔNG có filter (metadata_filter=None):")
    res_no_filt = store_sent.search(q3["query"], top_k=3)
    for rank, r in enumerate(res_no_filt, start=1):
        print(f"  Top {rank}: score={r['score']:.4f} doc={r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')})")
        print(f"         content: {r['content'][:110].replace(chr(10), ' ')}...")

    print("\n[B] Chạy CÓ filter (metadata_filter={'audience': 'student'}):")
    res_filt = store_sent.search_with_filter(q3["query"], top_k=3, metadata_filter={"audience": "student"})
    for rank, r in enumerate(res_filt, start=1):
        print(f"  Top {rank}: score={r['score']:.4f} doc={r['metadata'].get('doc_id')} (audience: {r['metadata'].get('audience')})")
        print(f"         content: {r['content'][:110].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    import io
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    run_benchmark()
    sys.stdout = old_stdout
    content = buffer.getvalue()
    print(content)
    Path("ket_qua_benchmark.txt").write_text(content, encoding="utf-8")
