"""Tests for token counting.

These hold whether or not tiktoken is installed, since they only assert on
emptiness and monotonicity rather than exact counts.
"""

from __future__ import annotations

from termcoder.context.tokens import TokenCounter


def test_empty_text_is_zero():
    assert TokenCounter().count_text("") == 0


def test_nonempty_text_is_positive():
    assert TokenCounter().count_text("hello world") > 0


def test_messages_count_grows_with_content():
    counter = TokenCounter()
    base = [{"role": "user", "content": "hi"}]
    longer = base + [{"role": "assistant", "content": "a noticeably longer reply"}]
    assert counter.count_messages(longer) > counter.count_messages(base)


def test_tool_calls_add_to_count():
    counter = TokenCounter()
    plain = [{"role": "assistant", "content": "x"}]
    with_call = [
        {
            "role": "assistant",
            "content": "x",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
            ],
        }
    ]
    assert counter.count_messages(with_call) > counter.count_messages(plain)
