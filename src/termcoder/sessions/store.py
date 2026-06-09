"""Per-chat conversation storage.

Each chat session is a JSON Lines transcript (one message per line) plus a
small JSON metadata sidecar. Plain-text, append-only storage keeps history
inspectable and easy to resume, and keeps conversation history separate from
the project-level long-term memory introduced in a later phase.

Files are written with ASCII escaping so transcripts stay portable and contain
no surprising non-keyboard characters.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path

from .models import SessionMeta

_TITLE_MAX = 60


def _now_iso() -> str:
    # Microsecond precision so sessions created within the same second still
    # order correctly by recency. ISO 8601 strings with a fixed offset sort
    # lexicographically, which is what the session list relies on.
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _new_id() -> str:
    """Return a sortable, unique session id like 20260608-153012-ab12cd."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _title_from_text(text: str) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= _TITLE_MAX:
        return flattened or "untitled"
    return flattened[: _TITLE_MAX - 3] + "..."


class Session:
    """A single chat session backed by a JSONL transcript on disk."""

    def __init__(
        self,
        meta: SessionMeta,
        messages_path: Path,
        meta_path: Path,
        messages: list[dict],
    ):
        self._meta = meta
        self._messages_path = messages_path
        self._meta_path = meta_path
        self._messages = messages

    @property
    def meta(self) -> SessionMeta:
        return self._meta

    @property
    def messages(self) -> list[dict]:
        """A copy of the message list, safe for the caller to iterate."""
        return list(self._messages)

    def append(self, message: dict) -> None:
        """Append a message to memory and to the transcript on disk."""
        self._messages.append(message)
        with self._messages_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(message, ensure_ascii=True) + "\n")
        self._meta.updated_at = _now_iso()
        self._write_meta()

    def maybe_set_title(self, text: str) -> None:
        """Set the session title from the first user message only."""
        if self._meta.title == "untitled":
            self._meta.title = _title_from_text(text)
            self._write_meta()

    def set_model(self, model: str) -> None:
        """Record a model change on the session metadata."""
        self._meta.model = model
        self._write_meta()

    def _write_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps(self._meta.to_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


class SessionStore:
    """Filesystem-backed store of chat sessions for one workspace."""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def create(self, model: str, title: str = "untitled") -> Session:
        """Create a brand new, empty session."""
        session_id = _new_id()
        now = _now_iso()
        meta = SessionMeta(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            model=model,
        )
        messages_path = self._messages_path(session_id)
        meta_path = self._meta_path(session_id)
        messages_path.touch()
        session = Session(meta, messages_path, meta_path, [])
        session._write_meta()
        return session

    def open(self, session_id: str) -> Session:
        """Load an existing session by id."""
        meta_path = self._meta_path(session_id)
        messages_path = self._messages_path(session_id)
        if not meta_path.is_file():
            raise FileNotFoundError(f"No session with id '{session_id}'.")
        meta = SessionMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
        messages = self._read_messages(messages_path)
        return Session(meta, messages_path, meta_path, messages)

    def list(self) -> list[SessionMeta]:
        """Return metadata for every session, newest first."""
        metas: list[SessionMeta] = []
        for meta_file in self._root.glob("*.meta.json"):
            try:
                metas.append(
                    SessionMeta.from_dict(
                        json.loads(meta_file.read_text(encoding="utf-8"))
                    )
                )
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        return metas

    def latest(self) -> SessionMeta | None:
        """Return the most recently updated session, if any."""
        sessions = self.list()
        return sessions[0] if sessions else None

    def _messages_path(self, session_id: str) -> Path:
        return self._root / f"{session_id}.jsonl"

    def _meta_path(self, session_id: str) -> Path:
        return self._root / f"{session_id}.meta.json"

    @staticmethod
    def _read_messages(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        messages: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                messages.append(json.loads(line))
        return messages
