"""Tests for converting LiteLLM assistant messages into storable dicts."""

from __future__ import annotations

from types import SimpleNamespace

from termcoder.llm.messages import assistant_message_to_dict


def _fake_tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_plain_text_message():
    message = SimpleNamespace(content="hello", tool_calls=None)
    data = assistant_message_to_dict(message)
    assert data == {"role": "assistant", "content": "hello"}


def test_message_with_tool_calls_is_preserved():
    message = SimpleNamespace(
        content="",
        tool_calls=[_fake_tool_call("call_1", "read_file", '{"path": "a.txt"}')],
    )
    data = assistant_message_to_dict(message)
    assert data["role"] == "assistant"
    assert data["tool_calls"][0]["id"] == "call_1"
    assert data["tool_calls"][0]["function"]["name"] == "read_file"
    assert data["tool_calls"][0]["function"]["arguments"] == '{"path": "a.txt"}'


def test_none_content_becomes_empty_string():
    message = SimpleNamespace(content=None, tool_calls=None)
    data = assistant_message_to_dict(message)
    assert data["content"] == ""
