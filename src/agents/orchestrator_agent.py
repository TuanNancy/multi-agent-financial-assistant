from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Tuple

from src.agents.language_agent import LanguageAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.llm_client import call_llm
from src.sentiment_service import SentimentService
from src.session_store import SessionStore


class OrchestratorAgent:
    """Điều phối toàn bộ pipeline: detect lang → extract intent → RAG → sentiment → summary."""

    def __init__(
        self,
        session_store: SessionStore,
        language_agent: LanguageAgent,
        retrieval_agent: RetrievalAgent,
        summarizer_agent: SummarizerAgent,
        sentiment_service: SentimentService,
    ) -> None:
        self.session_store = session_store
        self.language_agent = language_agent
        self.retrieval_agent = retrieval_agent
        self.summarizer_agent = summarizer_agent
        self.sentiment_service = sentiment_service

    async def handle(self, session_id: str, user_message: str) -> str:
        session = self.session_store.get_session(session_id)
        lang = self.language_agent.detect(
            user_message, session.get("preferences", {}).get("language")
            if isinstance(session, dict)
            else None
        )

        company, time_range = await self._extract_company_and_range(user_message)

        query_for_rag = (
            user_message if not company else f"{company} {user_message}"
        )
        ticker_filter = self._normalize_ticker(company)
        news = self.retrieval_agent.get_relevant_news(
            query_for_rag, ticker=ticker_filter, top_k=5
        )

        sentiments = (
            self.sentiment_service.analyze([n.content for n in news])
            if news
            else []
        )

        summary = await self.summarizer_agent.summarize_news_and_sentiment(
            news, sentiments, lang
        )

        self.session_store.update_session(
            session_id=session_id,
            user_msg=user_message,
            assistant_msg=summary,
            summary=summary,
        )

        return summary

    async def _extract_company_and_range(
        self, user_message: str
    ) -> Tuple[str | None, str | None]:
        """Gọi LLM để extract company + time_range, fallback an toàn."""
        parse_prompt = (
            "User query:\n"
            f"{user_message}\n\n"
            "Extract:\n"
            "- company (stock ticker or organization, short form if possible)\n"
            "- time_range (e.g. '6 months', '1 year', 'YTD')\n\n"
            'Return strict JSON like: {"company": "...", "time_range": "..."}'
        )

        messages = [
            {
                "role": "system",
                "content": "You extract structured info from user queries as strict JSON.",
            },
            {"role": "user", "content": parse_prompt},
        ]

        raw_response = await asyncio.to_thread(call_llm, messages)
        return self._safe_parse_structured_response(raw_response)

    @staticmethod
    def _safe_parse_structured_response(
        raw_response: str,
    ) -> Tuple[str | None, str | None]:
        """Parse JSON safely, tolerant với JSON trong code-blocks."""
        if not raw_response:
            return None, None

        candidate = raw_response.strip()

        if candidate.startswith("```"):
            parts = candidate.split("```")
            if len(parts) >= 3:
                candidate = parts[1].strip()

        try:
            data: Dict[str, Any] = json.loads(candidate)
        except json.JSONDecodeError:
            return None, None

        company = data.get("company")
        time_range = data.get("time_range")

        if isinstance(company, str):
            company = company.strip() or None
        else:
            company = None

        if isinstance(time_range, str):
            time_range = time_range.strip() or None
        else:
            time_range = None

        return company, time_range

    @staticmethod
    def _normalize_ticker(candidate: str | None) -> str | None:
        """Return uppercase ticker if candidate looks like one (1–5 alphanum chars)."""
        if not isinstance(candidate, str):
            return None

        ticker = candidate.strip().upper()
        if not ticker or len(ticker) > 5:
            return None

        if not ticker.isalnum():
            return None

        return ticker

