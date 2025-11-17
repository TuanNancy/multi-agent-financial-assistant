from __future__ import annotations

from types import SimpleNamespace

from src import llm_client


def test_call_llm_success(monkeypatch):
    """LLM client should hit configured endpoint and parse OpenAI-format payload."""
    llm_client.LLM_API_URL = "https://fake-llm"
    llm_client.LLM_MODEL_NAME = "mock-model"
    llm_client.LLM_API_KEY = "secret"
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "choices": [
                    {"message": {"content": "Hello investor!"}}
                ]
            },
        )

    monkeypatch.setattr(llm_client.requests, "post", fake_post)

    response = llm_client.call_llm([{"role": "user", "content": "Ping"}])

    assert response == "Hello investor!"
    assert captured["url"] == "https://fake-llm"
    assert captured["json"]["model"] == "mock-model"
    assert captured["headers"]["Authorization"] == "Bearer secret"

