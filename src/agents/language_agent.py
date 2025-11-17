from __future__ import annotations

from langdetect import detect


class LanguageAgent:
    """Simple language detection helper with user preference override."""

    def detect(self, text: str, user_preference: str | None = None) -> str:
        if user_preference in {"vi", "en"}:
            return user_preference

        text = text.strip()
        if not text:
            return "en"

        try:
            lang = detect(text)
        except Exception:
            return "en"

        return "vi" if lang.startswith("vi") else "en"

