"""Simple JSON-backed session persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.config import SESSIONS_FILE


class SessionStore:
    """Persist conversational sessions in a single JSON file on disk."""

    def __init__(self, path: str | Path = SESSIONS_FILE) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._write_empty()

    def _write_empty(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    def _load_all(self) -> Dict[str, Any]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
        except FileNotFoundError:
            self._write_empty()
            return {}

        if not content:
            self._write_empty()
            return {}

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            self._write_empty()
            return {}

    def _save_all(self, data: Dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        data = self._load_all()
        return data.get(session_id, {"history": [], "summary": ""})

    def update_session(
        self,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        summary: str | None = None,
    ) -> None:
        data = self._load_all()
        session = data.get(session_id, {"history": [], "summary": ""})
        session["history"].append({"role": "user", "content": user_msg})
        session["history"].append({"role": "assistant", "content": assistant_msg})
        if summary is not None:
            session["summary"] = summary
        data[session_id] = session
        self._save_all(data)


