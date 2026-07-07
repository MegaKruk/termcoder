"""Tests for the streaming think-tag filter.

These cover the tricky parts: splitting reasoning from visible text when the
``<think>`` and ``</think>`` markers arrive in pieces across chunk boundaries,
releasing plain text without lag, and stripping complete or unterminated
reasoning from a stored message.
"""

from __future__ import annotations

from termcoder.llm.thinking import ThinkingFilter, strip_thinking


def _drive(chunks: list[str]) -> tuple[str, str]:
    """Feed chunks through a fresh filter and return the (visible, thinking) totals."""
    thinking_filter = ThinkingFilter()
    visible_parts: list[str] = []
    thinking_parts: list[str] = []
    for chunk in chunks:
        visible, thinking = thinking_filter.feed(chunk)
        visible_parts.append(visible)
        thinking_parts.append(thinking)
    visible, thinking = thinking_filter.flush()
    visible_parts.append(visible)
    thinking_parts.append(thinking)
    return "".join(visible_parts), "".join(thinking_parts)


def test_plain_text_passes_through_unchanged():
    visible, thinking = _drive(["hello world"])
    assert visible == "hello world"
    assert thinking == ""


def test_plain_text_emits_immediately_without_buffering():
    thinking_filter = ThinkingFilter()
    visible, thinking = thinking_filter.feed("abc")
    assert visible == "abc"
    assert thinking == ""


def test_complete_inline_thinking_is_separated():
    visible, thinking = _drive(["answer <think>secret</think> done"])
    assert visible == "answer  done"
    assert thinking == "secret"


def test_open_tag_split_across_chunks():
    visible, thinking = _drive(["ans <thi", "nk>rea", "soning</thi", "nk> done"])
    assert visible == "ans  done"
    assert thinking == "reasoning"


def test_only_a_genuine_tag_prefix_is_withheld():
    # A trailing "<" could start a tag, so it is held until the next chunk
    # proves it is just a less-than sign, then released.
    thinking_filter = ThinkingFilter()
    visible, _ = thinking_filter.feed("a <")
    assert visible == "a "
    visible, _ = thinking_filter.feed("b")
    assert visible == "<b"


def test_unterminated_thinking_goes_entirely_to_reasoning():
    visible, thinking = _drive(["visible <think>still thinking"])
    assert visible == "visible "
    assert thinking == "still thinking"


def test_multiple_thinking_sections():
    visible, thinking = _drive(["<think>a</think>X<think>b</think>Y"])
    assert visible == "XY"
    assert thinking == "ab"


def test_strip_thinking_removes_complete_section():
    assert strip_thinking("keep <think>drop</think> keep") == "keep  keep"


def test_strip_thinking_drops_unterminated_remainder():
    assert strip_thinking("keep <think>drop the rest") == "keep "


def test_strip_thinking_without_tags_is_identity():
    assert strip_thinking("nothing to remove") == "nothing to remove"
