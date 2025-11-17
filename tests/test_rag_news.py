from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from src.rag_news import NewsRAG


class DummyEmbeddingService:
    def encode(self, texts: List[str], **kwargs):
        vectors = []
        for text in texts:
            if "Tesla" in text or "TSLA" in text:
                vectors.append(np.array([1.0, 0.0]))
            else:
                vectors.append(np.array([0.0, 1.0]))
        return np.vstack(vectors)


def test_newsrag_loads_jsonl_and_searches(tmp_path: Path):
    index_path = tmp_path / "news.jsonl"
    docs = [
        {
            "id": "n1",
            "title": "Tesla beats expectations",
            "content": "Tesla reports strong growth",
            "date": "2025-01-01",
            "ticker": "TSLA",
        },
        {
            "id": "n2",
            "title": "Apple launches product",
            "content": "Apple expands services",
            "date": "2025-01-02",
            "ticker": "AAPL",
        },
    ]
    index_path.write_text(
        "\n".join(json.dumps(doc) for doc in docs), encoding="utf-8"
    )

    rag = NewsRAG(
        embed_service=DummyEmbeddingService(),
        index_path=index_path,
        similarity_threshold=0.1,
    )

    results = rag.search("Tell me about Tesla", ticker="TSLA", top_k=2)

    assert len(results) == 1
    assert results[0].ticker == "TSLA"

