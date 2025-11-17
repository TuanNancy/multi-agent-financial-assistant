from __future__ import annotations

from src.session_store import SessionStore


def test_session_store_persists_history(tmp_path):
    store_path = tmp_path / "sessions.json"
    store = SessionStore(path=store_path)

    store.update_session(
        session_id="abc",
        user_msg="Hi",
        assistant_msg="Hello",
        summary="Summary",
    )

    loaded = store.get_session("abc")

    assert loaded["history"][0]["content"] == "Hi"
    assert loaded["history"][1]["content"] == "Hello"
    assert loaded["summary"] == "Summary"

