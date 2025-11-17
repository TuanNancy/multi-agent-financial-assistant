from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.agents.orchestrator_agent import OrchestratorAgent
from src.rag_news import NewsItem
from src.session_store import SessionStore


class DummyLanguageAgent:
    def detect(self, text, user_pref=None):
        return "en"


class DummyRetrievalAgent:
    def get_relevant_news(self, *args, **kwargs):
        return [
            NewsItem(
                id="n1",
                title="Tesla hits target",
                content="Tesla delivered vehicles",
                date="2025-01-01",
                ticker="TSLA",
            )
        ]


class DummySummarizerAgent:
    async def summarize_news_and_sentiment(self, news, sentiments, lang):
        return "Summary ready"


class DummySentimentService:
    def analyze(self, texts):
        return [{"label": "positive", "score": 0.9}]


@pytest.mark.asyncio
async def test_orchestrator_pipeline(monkeypatch, tmp_path):
    session_path = tmp_path / "sessions.json"
    store = SessionStore(path=session_path)

    monkeypatch.setattr(
        "src.agents.orchestrator_agent.call_llm",
        lambda messages: '{"company": "TSLA", "time_range": "Q1"}',
    )

    orchestrator = OrchestratorAgent(
        session_store=store,
        language_agent=DummyLanguageAgent(),
        retrieval_agent=DummyRetrievalAgent(),
        summarizer_agent=DummySummarizerAgent(),
        sentiment_service=DummySentimentService(),
    )

    result = await orchestrator.handle("session-1", "Tell me about Tesla")

    assert result == "Summary ready"
    history = store.get_session("session-1")["history"]
    assert history[-1]["role"] == "assistant"

