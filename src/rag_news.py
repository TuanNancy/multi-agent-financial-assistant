from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.config import NEWS_INDEX_FILE, RAG_SIMILARITY_THRESHOLD, TOP_K_RESULTS
from src.embedding_service import EmbeddingService


@dataclass(slots=True)
class NewsItem:
    """Đại diện cho một bản tin tài chính trong chỉ mục."""

    id: str
    title: str
    content: str
    date: str
    ticker: str


class NewsRAG:
    """Simple JSONL-based news retriever using dense embeddings."""

    def __init__(
        self,
        embed_service: EmbeddingService,
        index_path: str | Path = NEWS_INDEX_FILE,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
    ) -> None:
        self.embed_service = embed_service
        self.index_path = Path(index_path)
        self.similarity_threshold = similarity_threshold
        self.news: List[NewsItem] = []
        self.embeddings: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        """Load news và embeddings từ file JSONL (mỗi dòng 1 object)."""
        if not self.index_path.exists():
            self.news = []
            self.embeddings = None
            return

        with self.index_path.open("r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            self.news = []
            self.embeddings = None
            return

        if raw.startswith("["):
            parsed = json.loads(raw)
        else:
            parsed = [
                json.loads(line)
                for line in raw.splitlines()
                if line.strip()
            ]

        news: List[NewsItem] = []
        texts: List[str] = []
        for obj in parsed:
            item = NewsItem(
                id=obj["id"],
                title=obj["title"],
                content=obj["content"],
                date=obj["date"],
                ticker=obj["ticker"],
            )
            news.append(item)
            texts.append(item.content)

        self.news = news
        self.embeddings = self.embed_service.encode(texts) if texts else None

    def reload(self) -> None:
        """Cho phép refresh dữ liệu tin tức từ file."""
        self._load()

    def search(
        self,
        query: str,
        ticker: Optional[str] = None,
        top_k: int = TOP_K_RESULTS,
    ) -> List[NewsItem]:
        """Tìm các bản tin phù hợp nhất với query và ticker (nếu cung cấp)."""
        if not query.strip():
            return []
        if self.embeddings is None or not self.news:
            return []

        q_emb = self.embed_service.encode([query])[0]
        sims = self.embeddings @ q_emb  # cosine similarity vì đã normalize

        ordered_idxs = np.argsort(-sims)
        results: List[NewsItem] = []
        ticker_lower = ticker.lower() if ticker else None

        for idx in ordered_idxs:
            sim = sims[idx]
            if sim < self.similarity_threshold:
                break

            item = self.news[idx]
            if ticker_lower and item.ticker.lower() != ticker_lower:
                continue

            results.append(item)
            if len(results) >= top_k:
                break

        return results

