from __future__ import annotations

from types import SimpleNamespace

from src import app


def test_init_dependencies_wires_services(monkeypatch):
    created = {}

    class FakeEmbeddingService:
        def __init__(self):
            created["embedding"] = self

    class FakeRAG:
        def __init__(self, embed_service):
            created["rag"] = embed_service
            self.news = [object()]

    class FakeSessionStore:
        def __init__(self):
            created["session"] = self

    class FakeSentiment:
        pass

    class FakeRetrievalAgent:
        def __init__(self, rag):
            created["retrieval"] = rag

    class FakeSummarizer:
        pass

    class FakeLanguageAgent:
        pass

    class FakeOrchestratorAgent:
        def __init__(self, **kwargs):
            created["orchestrator_args"] = kwargs

    monkeypatch.setattr(app, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(app, "NewsRAG", FakeRAG)
    monkeypatch.setattr(app, "SessionStore", FakeSessionStore)
    monkeypatch.setattr(app, "SentimentService", FakeSentiment)
    monkeypatch.setattr(app, "RetrievalAgent", FakeRetrievalAgent)
    monkeypatch.setattr(app, "SummarizerAgent", FakeSummarizer)
    monkeypatch.setattr(app, "LanguageAgent", FakeLanguageAgent)
    monkeypatch.setattr(app, "OrchestratorAgent", FakeOrchestratorAgent)

    deps = app.init_dependencies()

    assert "orchestrator" in deps
    assert created["orchestrator_args"]["session_store"] is created["session"]

