"""Tests for reasoning handling in the streaming LLM client.

A fake litellm module is injected so no network or real provider is needed.
These check that reasoning arriving either as a dedicated ``reasoning_content``
field or as inline ``<think>`` tags is routed to the thinking callback, that
visible text still reaches the text callback, and that the stored message has
its reasoning stripped so it is never replayed to the model.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from termcoder.config import ModelConfig
from termcoder.providers.llm_client import LLMClient


def _chunk(content=None, reasoning=None) -> SimpleNamespace:
    """Build a streamed chunk with optional visible and reasoning deltas."""
    delta = SimpleNamespace(content=content)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _install_fake_litellm(monkeypatch, chunks, final_content) -> None:
    """Register a fake litellm module that replays chunks and a final message."""
    fake = ModuleType("litellm")

    def completion(messages=None, stream=False, **kwargs):
        return iter(chunks)

    def stream_chunk_builder(collected, messages=None):
        message = SimpleNamespace(content=final_content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=None
        )

    fake.completion = completion
    fake.stream_chunk_builder = stream_chunk_builder
    monkeypatch.setitem(sys.modules, "litellm", fake)


def _client() -> LLMClient:
    return LLMClient(
        ModelConfig(name="test", model="ollama_chat/qwen3", cache_prompts=False)
    )


def test_reasoning_content_field_routes_to_thinking(monkeypatch):
    chunks = [
        _chunk(reasoning="I should "),
        _chunk(reasoning="plan first. "),
        _chunk(content="The answer is 42."),
    ]
    _install_fake_litellm(monkeypatch, chunks, "The answer is 42.")
    visible: list[str] = []
    thinking: list[str] = []
    result = _client().complete(
        [{"role": "user", "content": "q"}],
        on_text=visible.append,
        on_thinking=thinking.append,
    )
    assert "".join(visible) == "The answer is 42."
    assert "".join(thinking) == "I should plan first. "
    assert result.message.content == "The answer is 42."


def test_inline_think_tags_route_to_thinking_and_are_stripped(monkeypatch):
    chunks = [
        _chunk(content="<thi"),
        _chunk(content="nk>inside</think>"),
        _chunk(content="visible answer"),
    ]
    _install_fake_litellm(
        monkeypatch, chunks, "<think>inside</think>visible answer"
    )
    visible: list[str] = []
    thinking: list[str] = []
    result = _client().complete(
        [{"role": "user", "content": "q"}],
        on_text=visible.append,
        on_thinking=thinking.append,
    )
    assert "".join(visible) == "visible answer"
    assert "".join(thinking) == "inside"
    assert result.message.content == "visible answer"


def test_mixed_reasoning_sources_merge_into_one_channel(monkeypatch):
    chunks = [
        _chunk(reasoning="field part. "),
        _chunk(content="<think>tag part</think>done"),
    ]
    _install_fake_litellm(monkeypatch, chunks, "<think>tag part</think>done")
    thinking: list[str] = []
    visible: list[str] = []
    _client().complete(
        [{"role": "user", "content": "q"}],
        on_text=visible.append,
        on_thinking=thinking.append,
    )
    assert "".join(visible) == "done"
    assert "".join(thinking) == "field part. tag part"


def test_thinking_callback_is_optional(monkeypatch):
    chunks = [_chunk(reasoning="ignored"), _chunk(content="hi")]
    _install_fake_litellm(monkeypatch, chunks, "hi")
    visible: list[str] = []
    # No on_thinking passed: reasoning is simply dropped, visible text remains.
    result = _client().complete(
        [{"role": "user", "content": "q"}], on_text=visible.append
    )
    assert "".join(visible) == "hi"
    assert result.message.content == "hi"
