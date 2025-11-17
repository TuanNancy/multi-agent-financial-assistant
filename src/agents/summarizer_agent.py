from __future__ import annotations

import asyncio
from typing import List, Sequence

from src.llm_client import call_llm
from src.rag_news import NewsItem


class SummarizerAgent:
    """Gọi LLM để tóm tắt tin tức + sentiment FinBERT."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = (
            system_prompt
            or "You are a helpful financial analysis assistant."
        )

    async def summarize_news_and_sentiment(
        self,
        news: List[NewsItem],
        sentiments: Sequence[dict],
        lang: str,
    ) -> str:
        lang = "vi" if lang == "vi" else "en"

        if not news:
            return (
                "Không tìm thấy tin tức phù hợp."
                if lang == "vi"
                else "No relevant news found."
            )

        bullets: List[str] = []
        for idx, item in enumerate(news):
            sentiment = sentiments[idx] if idx < len(sentiments) else {}
            label = str(sentiment.get("label", "neutral"))
            score = sentiment.get("score")
            score_str = (
                f"{float(score):.2f}"
                if isinstance(score, (float, int))
                else "n/a"
            )
            bullets.append(
                f"- [{item.date}] {item.title} ({label} / {score_str})"
            )

        news_text = "\n".join(bullets)

        prompt_vi = (
            "Bạn là chuyên gia tài chính. Dưới đây là danh sách tin tức và "
            "cảm xúc (FinBERT):\n\n"
            f"{news_text}\n\n"
            "Hãy tóm tắt ngắn gọn (5–7 câu) tình hình chung và tâm lý thị "
            "trường, bằng tiếng Việt, dễ hiểu cho nhà đầu tư cá nhân."
        )

        prompt_en = (
            "You are a financial analyst. Below are news items with their "
            "sentiment (FinBERT):\n\n"
            f"{news_text}\n\n"
            "Summarize the overall situation and market sentiment in 5–7 "
            "sentences in English, suitable for a retail investor."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": prompt_vi if lang == "vi" else prompt_en,
            },
        ]

        return await asyncio.to_thread(call_llm, messages)

