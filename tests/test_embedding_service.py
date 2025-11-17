from __future__ import annotations

import numpy as np

from src.embedding_service import EmbeddingService


def test_embedding_service_encode(monkeypatch):
    """Embedding service should wrap SentenceTransformer encode outputs."""

    class DummySentenceTransformer:
        def __init__(self, model_name, device):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 3))

        def get_sentence_embedding_dimension(self):
            return 3

    monkeypatch.setattr(
        "src.embedding_service.SentenceTransformer",
        DummySentenceTransformer,
    )

    svc = EmbeddingService(model_name="fake-model", device="cpu")

    embeddings = svc.encode(["hello", "world"])

    assert embeddings.shape == (2, 3)
    assert svc.get_embedding_dimension() == 3

