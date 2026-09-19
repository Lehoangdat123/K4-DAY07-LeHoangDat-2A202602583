from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức."

        context_blocks = []
        for idx, res in enumerate(results, start=1):
            src = res.get("metadata", {}).get("source", res.get("id", f"doc_{idx}"))
            context_blocks.append(f"[{idx}] (Nguồn: {src})\n{res['content']}")
        context_text = "\n\n".join(context_blocks)

        prompt = (
            f"Dưới đây là các đoạn văn bản trích xuất từ tài liệu liên quan:\n\n"
            f"{context_text}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
            f"Chỉ dẫn nguồn bằng số thứ tự [1], [2] tương ứng."
        )
        return self.llm_fn(prompt)
