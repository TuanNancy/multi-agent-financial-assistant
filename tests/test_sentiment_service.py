from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from src.sentiment_service import SentimentService


def test_sentiment_service_uses_local_checkpoint(tmp_path, monkeypatch):
    model_dir = tmp_path / "finbert"
    model_dir.mkdir()

    class DummyTokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.zeros((1, 4), dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                id2label={0: "negative", 1: "positive"}
            )

        def to(self, device):
            return self

        def eval(self):
            return None

        def forward(self, **kwargs):
            logits = torch.tensor([[0.1, 0.9]])
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(
        "src.sentiment_service.AutoTokenizer.from_pretrained",
        lambda path: DummyTokenizer(),
    )
    monkeypatch.setattr(
        "src.sentiment_service.AutoModelForSequenceClassification.from_pretrained",
        lambda path: DummyModel(),
    )

    service = SentimentService(model_dir=model_dir, device="cpu")

    result = service.analyze(["Great quarter"])

    assert result[0]["label"] == "positive"
    assert 0.0 <= result[0]["score"] <= 1.0

