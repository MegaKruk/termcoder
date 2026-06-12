"""Tests for the provider layer helpers.

These cover the pieces that run before and after the actual model call: usage
accounting, prompt-cache breakpoint preparation, and reading cached-token
counts from either provider's usage shape. No network and no LiteLLM import is
needed because the client only touches LiteLLM inside the call methods.
"""

from __future__ import annotations

from types import SimpleNamespace

from termcoder.config import ModelConfig
from termcoder.providers.llm_client import LLMClient, _cached_tokens
from termcoder.providers.usage import UsageTracker


def _model(model: str, cache_prompts: bool = True) -> ModelConfig:
    return ModelConfig(name="test", model=model, cache_prompts=cache_prompts)


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "tool", "tool_call_id": "1", "content": "tool output"},
    ]


def test_tracker_accumulates_turn_and_session():
    tracker = UsageTracker()
    tracker.begin_turn()
    tracker.record(prompt_tokens=100, completion_tokens=20, cached_tokens=50, cost_usd=0.01)
    tracker.record(prompt_tokens=200, completion_tokens=30)

    assert tracker.turn.calls == 2
    assert tracker.turn.prompt_tokens == 300
    assert tracker.turn.completion_tokens == 50
    assert tracker.turn.cached_tokens == 50
    assert tracker.session.prompt_tokens == 300

    tracker.begin_turn()
    tracker.record(prompt_tokens=10, completion_tokens=1)

    assert tracker.turn.calls == 1
    assert tracker.turn.prompt_tokens == 10
    assert tracker.session.calls == 3
    assert tracker.session.prompt_tokens == 310
    assert abs(tracker.session.cost_usd - 0.01) < 1e-9


def test_tracker_reset_clears_everything():
    tracker = UsageTracker()
    tracker.record(prompt_tokens=5, completion_tokens=5)
    tracker.reset()
    assert tracker.session.calls == 0
    assert tracker.turn.calls == 0


def test_tracker_clamps_negative_values():
    tracker = UsageTracker()
    tracker.record(prompt_tokens=-5, completion_tokens=-1, cached_tokens=-2, cost_usd=-0.5)
    assert tracker.session.prompt_tokens == 0
    assert tracker.session.completion_tokens == 0
    assert tracker.session.cached_tokens == 0
    assert tracker.session.cost_usd == 0.0


def test_cache_breakpoints_added_for_anthropic_models():
    client = LLMClient(_model("anthropic/claude-sonnet-4-6"), stream=False)
    original = _messages()

    prepared = client._prepare_messages(original)

    system = prepared[0]["content"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    assert system[0]["text"] == "You are a coding assistant."

    last_user = prepared[3]["content"]
    assert isinstance(last_user, list)
    assert last_user[0]["cache_control"] == {"type": "ephemeral"}

    # Earlier user message and other roles stay plain strings.
    assert prepared[1]["content"] == "first question"
    assert prepared[2]["content"] == "first answer"
    assert prepared[4]["content"] == "tool output"

    # The caller's message list is never mutated.
    assert original[0]["content"] == "You are a coding assistant."
    assert original[3]["content"] == "second question"


def test_cache_breakpoints_skipped_for_other_providers():
    client = LLMClient(_model("ollama_chat/llama3.1"), stream=False)
    messages = _messages()
    assert client._prepare_messages(messages) is messages


def test_cache_breakpoints_respect_config_toggle():
    client = LLMClient(
        _model("anthropic/claude-sonnet-4-6", cache_prompts=False), stream=False
    )
    messages = _messages()
    assert client._prepare_messages(messages) is messages


def test_cached_tokens_reads_both_provider_shapes():
    anthropic_usage = SimpleNamespace(cache_read_input_tokens=42)
    openai_usage = SimpleNamespace(
        cache_read_input_tokens=None,
        prompt_tokens_details=SimpleNamespace(cached_tokens=7),
    )
    assert _cached_tokens(anthropic_usage) == 42
    assert _cached_tokens(openai_usage) == 7
    assert _cached_tokens(SimpleNamespace()) == 0
    assert _cached_tokens(None) == 0
