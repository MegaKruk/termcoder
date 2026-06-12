"""Provider-agnostic chat client built on LiteLLM.

A single :class:`LLMClient` works across OpenAI, Anthropic and local Ollama
models because LiteLLM normalizes them to the OpenAI chat format. When
streaming, text chunks are surfaced live through a callback and then
reassembled with LiteLLM's stream-chunk builder so any tool calls in the
response are reconstructed correctly.

This is the one chokepoint every model call passes through, which makes it the
right place for two cross-cutting concerns:

* Usage metering. Provider-reported token usage, cache hits, and an estimated
  cost are recorded into a shared :class:`UsageTracker` after every call, with
  a local token estimate as the fallback when a provider reports nothing.
* Prompt caching. For Anthropic-family models, ``cache_control`` breakpoints
  are added to the system prompt and the last user message, so the stable
  prefix (system + history) is served from the provider cache at a fraction of
  the input price. Other providers either cache automatically on stable
  prefixes (OpenAI) or ignore the field, so requests stay portable.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..config import ModelConfig
from ..context.tokens import TokenCounter
from ..errors import MalformedModelOutputError, ProviderError
from .usage import UsageTracker

_CACHE_CONTROL = {"type": "ephemeral"}


class CompletionResult:
    """The outcome of a single model call."""

    def __init__(self, message: Any, raw: Any):
        self.message = message
        self.raw = raw


def _chunk_text(chunk: Any) -> str | None:
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, IndexError):
        return None
    return getattr(delta, "content", None)


def _is_malformed_output(exc: Exception) -> bool:
    """Decide whether a provider failure was unparseable model output.

    Small local models sometimes emit invalid tool-call JSON; LiteLLM then
    raises a JSONDecodeError (possibly wrapped) while parsing the stream. The
    exception chain and message are both checked because wrapping varies by
    provider path and LiteLLM version.
    """
    seen: Any = exc
    while seen is not None:
        if isinstance(seen, json.JSONDecodeError):
            return True
        seen = seen.__cause__ or seen.__context__
    text = str(exc)
    return "JSONDecodeError" in text or "Expecting value:" in text


class LLMClient:
    """A thin completion client bound to one model configuration."""

    def __init__(
        self,
        config: ModelConfig,
        stream: bool = True,
        usage_tracker: UsageTracker | None = None,
    ):
        self._config = config
        self._stream = stream
        self._usage = usage_tracker
        self._estimator = TokenCounter()

    @property
    def model_name(self) -> str:
        return self._config.model

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_text: Callable[[str], None] | None = None,
    ) -> CompletionResult:
        """Run a completion, streaming text through ``on_text`` when enabled."""
        kwargs = self._config.to_completion_kwargs()
        if tools:
            kwargs["tools"] = tools
        prepared = self._prepare_messages(messages)
        try:
            if self._stream:
                result = self._complete_streaming(prepared, kwargs, on_text)
            else:
                result = self._complete_blocking(prepared, kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # surface any provider failure uniformly
            if _is_malformed_output(exc):
                raise MalformedModelOutputError(
                    "The model produced output that could not be parsed "
                    f"(often malformed tool-call JSON): {exc}"
                ) from exc
            raise ProviderError(f"Model request failed: {exc}") from exc
        self._record_usage(prepared, result)
        return result

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Return messages with prompt-cache breakpoints where they help.

        Breakpoints go on the system prompt and on the last user message, so
        the provider caches the stable prefix (system plus all history up to
        the current request). Only Anthropic-family models receive them; other
        providers cache automatically or have no equivalent.
        """
        if not self._config.cache_prompts or not self._is_anthropic_family():
            return messages
        prepared = list(messages)
        for index, message in enumerate(prepared):
            if message.get("role") == "system":
                prepared[index] = self._with_cache_control(message)
                break
        for index in range(len(prepared) - 1, -1, -1):
            if prepared[index].get("role") == "user":
                prepared[index] = self._with_cache_control(prepared[index])
                break
        return prepared

    def _is_anthropic_family(self) -> bool:
        model = self._config.model.lower()
        return "claude" in model or model.startswith("anthropic")

    @staticmethod
    def _with_cache_control(message: dict) -> dict:
        content = message.get("content")
        if not isinstance(content, str) or not content:
            return message
        updated = dict(message)
        updated["content"] = [
            {"type": "text", "text": content, "cache_control": dict(_CACHE_CONTROL)}
        ]
        return updated

    def _complete_streaming(
        self,
        messages: list[dict],
        kwargs: dict,
        on_text: Callable[[str], None] | None,
    ) -> CompletionResult:
        import litellm

        chunks: list[Any] = []
        stream = litellm.completion(messages=messages, stream=True, **kwargs)
        for chunk in stream:
            chunks.append(chunk)
            text = _chunk_text(chunk)
            if text and on_text is not None:
                on_text(text)
        if not chunks:
            raise ProviderError("The model returned an empty response.")
        assembled = litellm.stream_chunk_builder(chunks, messages=messages)
        if assembled is None:
            raise ProviderError("Could not assemble the streamed response.")
        return CompletionResult(assembled.choices[0].message, assembled)

    def _complete_blocking(
        self, messages: list[dict], kwargs: dict
    ) -> CompletionResult:
        import litellm

        response = litellm.completion(messages=messages, stream=False, **kwargs)
        return CompletionResult(response.choices[0].message, response)

    def _record_usage(self, messages: list[dict], result: CompletionResult) -> None:
        """Record this call's usage, estimating anything the provider omitted."""
        if self._usage is None:
            return
        usage = getattr(result.raw, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        cached_tokens = _cached_tokens(usage)
        if prompt_tokens == 0:
            prompt_tokens = self._estimator.count_messages(messages)
        if completion_tokens == 0:
            completion_tokens = self._estimator.count_text(
                getattr(result.message, "content", None) or ""
            )
        self._usage.record(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cost_usd=_estimate_cost(result.raw),
        )


def _cached_tokens(usage: Any) -> int:
    """Extract cached prompt tokens from either provider's usage shape."""
    if usage is None:
        return 0
    cached = getattr(usage, "cache_read_input_tokens", None)
    if cached:
        return int(cached)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None)
    return int(cached or 0)


def _estimate_cost(response: Any) -> float:
    """Estimate the call cost in USD; zero for unknown or local models."""
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response) or 0.0)
    except Exception:
        return 0.0
