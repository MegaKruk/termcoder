"""Tests for conversation compaction.

A scripted stand-in for the model returns a fixed summary, so the tests stay
offline and deterministic. They cover the turn-boundary cut, the automatic and
forced paths, and that tool-call and tool-result messages are never split.
"""

from __future__ import annotations

from termcoder.context import ContextManager
from termcoder.sessions.store import SessionStore


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeResult:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)
        self.raw = None


class ScriptedLLM:
    """A model stand-in that records calls and returns a fixed summary."""

    def __init__(self, summary: str = "SUMMARY"):
        self._summary = summary
        self.calls = 0

    def complete(self, messages, tools=None, on_text=None):
        self.calls += 1
        return _FakeResult(self._summary)


def _session(tmp_path):
    return SessionStore(tmp_path / "sessions").create(model="test")


def _add_turns(session, count):
    for index in range(count):
        session.append({"role": "user", "content": f"question {index} " + "x" * 40})
        session.append({"role": "assistant", "content": f"answer {index} " + "y" * 40})


def test_cut_index_keeps_recent_turns():
    manager = ContextManager(
        auto_compact=True,
        compact_threshold=0.8,
        keep_recent_turns=2,
        context_window=1000,
    )
    messages = [
        {"role": "user", "content": "t1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "t2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "t3"},
        {"role": "assistant", "content": "a3"},
    ]
    assert manager._cut_index(messages) == 2


def test_cut_index_zero_when_few_turns():
    manager = ContextManager(
        auto_compact=True,
        compact_threshold=0.8,
        keep_recent_turns=3,
        context_window=1000,
    )
    messages = [
        {"role": "user", "content": "t1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert manager._cut_index(messages) == 0


def test_maybe_compact_summarizes_when_over_budget(tmp_path):
    session = _session(tmp_path)
    _add_turns(session, 6)
    llm = ScriptedLLM("THE SUMMARY")
    manager = ContextManager(
        auto_compact=True,
        compact_threshold=0.01,
        keep_recent_turns=2,
        context_window=100,
        summarizer=llm,
    )

    result = manager.maybe_compact(session)

    assert result is not None
    assert session.meta.summary == "THE SUMMARY"
    assert session.meta.summary_through == 8
    tail = manager.tail_messages(session)
    assert tail[0]["role"] == "user"
    assert len(tail) == 4
    assert llm.calls == 1


def test_no_compaction_below_budget(tmp_path):
    session = _session(tmp_path)
    _add_turns(session, 1)
    llm = ScriptedLLM()
    manager = ContextManager(
        auto_compact=True,
        compact_threshold=0.8,
        keep_recent_turns=2,
        context_window=100000,
        summarizer=llm,
    )

    assert manager.maybe_compact(session) is None
    assert llm.calls == 0
    assert session.meta.summary is None


def test_force_compact_ignores_budget(tmp_path):
    session = _session(tmp_path)
    _add_turns(session, 5)
    manager = ContextManager(
        auto_compact=False,
        compact_threshold=0.8,
        keep_recent_turns=2,
        context_window=100000,
        summarizer=ScriptedLLM("FORCED"),
    )

    result = manager.force_compact(session, instructions="focus on the goal")

    assert result is not None
    assert session.meta.summary == "FORCED"
    assert session.meta.summary_through == 6


def test_compaction_preserves_tool_pairing(tmp_path):
    session = _session(tmp_path)
    session.append({"role": "user", "content": "do x"})
    session.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }
    )
    session.append({"role": "tool", "tool_call_id": "1", "content": "file body"})
    session.append({"role": "user", "content": "next"})
    session.append({"role": "assistant", "content": "done"})
    session.append({"role": "user", "content": "more"})
    session.append({"role": "assistant", "content": "ok"})

    manager = ContextManager(
        auto_compact=True,
        compact_threshold=0.01,
        keep_recent_turns=1,
        context_window=50,
        summarizer=ScriptedLLM("S"),
    )
    manager.maybe_compact(session)

    assert session.meta.summary_through == 5
    tail = manager.tail_messages(session)
    assert tail[0]["role"] == "user"
    assert all(message["role"] != "tool" for message in tail)


def test_no_summarizer_means_no_compaction(tmp_path):
    session = _session(tmp_path)
    _add_turns(session, 6)
    manager = ContextManager(
        auto_compact=True,
        compact_threshold=0.01,
        keep_recent_turns=2,
        context_window=100,
    )

    assert manager.maybe_compact(session) is None
    assert session.meta.summary is None
