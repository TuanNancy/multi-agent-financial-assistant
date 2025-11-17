"""Sentiment analysis service backed by a local FinBERT checkpoint."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import FINBERT_DEVICE, FINBERT_MAX_LENGTH, FINBERT_MODEL_PATH


class SentimentService:
    """Load a locally fine-tuned FinBERT model and run single-pass inference."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        max_length: int | None = None,
    ) -> None:
        model_path = Path(model_dir or FINBERT_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"FinBERT model path not found: {model_path}")

        self.device = torch.device(device or FINBERT_DEVICE)
        self.max_length = max_length or FINBERT_MAX_LENGTH

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()

        # Some checkpoints may not populate id2label; ensure a fallback mapping.
        config_labels = getattr(self.model.config, "id2label", None) or {}
        if config_labels:
            self.id2label = {int(idx): lbl for idx, lbl in config_labels.items()}
        else:
            num_labels = int(getattr(self.model.config, "num_labels", 0))
            self.id2label = {idx: f"LABEL_{idx}" for idx in range(num_labels)}

    def analyze(self, texts: Sequence[str]) -> List[Dict[str, float | str]]:
        """Return FinBERT sentiment label + score for each input text."""
        if not isinstance(texts, Iterable):
            raise TypeError("texts must be an iterable of strings")

        results: List[Dict[str, float | str]] = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Each item in texts must be a string")

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].cpu()

            pred_id = torch.argmax(probs).item()
            label = self.id2label.get(pred_id, str(pred_id))
            score = float(probs[pred_id].item())

            results.append(
                {
                    "text": text,
                    "label": label,
                    "score": score,
                }
            )

        return results


