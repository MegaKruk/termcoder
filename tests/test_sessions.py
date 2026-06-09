"""Tests for per-chat session storage."""

from __future__ import annotations

from termcoder.llm.messages import assistant_message_to_dict, tool_message, user_message
from termcoder.sessions.store import SessionStore


def test_create_append_and_reopen_roundtrip(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = store.create(model="ollama")
    session.maybe_set_title("Fix the parser bug")
    session.append(user_message("hello"))
    session.append({"role": "assistant", "content": "hi there"})
    session.append(tool_message("call_1", "tool output"))

    reopened = store.open(session.meta.id)
    assert reopened.meta.title == "Fix the parser bug"
    assert len(reopened.messages) == 3
    assert reopened.messages[0]["content"] == "hello"
    assert reopened.messages[2]["tool_call_id"] == "call_1"


def test_list_and_latest(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    first = store.create(model="ollama")
    second = store.create(model="openai")
    second.append(user_message("newer"))

    listed = store.list()
    ids = {meta.id for meta in listed}
    assert first.meta.id in ids
    assert second.meta.id in ids

    latest = store.latest()
    assert latest is not None
    assert latest.id == second.meta.id


def test_title_defaults_until_first_user_message(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    session = store.create(model="ollama")
    assert session.meta.title == "untitled"
    session.maybe_set_title("First question here")
    session.maybe_set_title("Second question should not override")
    assert session.meta.title == "First question here"
