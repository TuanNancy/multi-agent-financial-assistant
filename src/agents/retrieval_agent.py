from __future__ import annotations

from typing import List, Optional

from src.rag_news import NewsItem, NewsRAG


class RetrievalAgent:
    """Agent phụ trách tìm tin tức phù hợp dựa trên RAG backend."""

    def __init__(self, rag: NewsRAG) -> None:
        self.rag = rag

    def get_relevant_news(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: int | None = None,
    ) -> List[NewsItem]:
        """
        Ủy quyền cho `NewsRAG` để tìm tin liên quan.

        Args:
            query: Câu hỏi/từ khóa cần tìm.
            ticker: Mã cổ phiếu (nếu muốn lọc).
            top_k: Số lượng tin tối đa, fallback sang cấu hình RAG khi None.
        """
        if not query.strip():
            return []

        if top_k is None:
            return self.rag.search(query=query, ticker=ticker)

        return self.rag.search(query=query, ticker=ticker, top_k=top_k)

