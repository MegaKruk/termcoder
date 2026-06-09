"""Provider-agnostic chat client built on LiteLLM.

A single :class:`LLMClient` works across OpenAI, Anthropic and local Ollama
models because LiteLLM normalizes them to the OpenAI chat format. When
streaming, text chunks are surfaced live through a callback and then
reassembled with LiteLLM's stream-chunk builder so any tool calls in the
response are reconstructed correctly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import ModelConfig
from ..errors import ProviderError


class CompletionResult:
    """The outcome of a single model call."""

    def __init__(self, message: Any, raw: Any):
        self.message = message
        self.raw = raw


def _chunk_text(chunk: Any) -> str:
    """Extract the text delta from a streaming chunk, if any."""
    try:
        delta = chunk.choices[0].delta
    except (AttributeError, IndexError):
        return ""
    content = getattr(delta, "content", None)
    return content or ""


class LLMClient:
    """Send chat requests to a configured model and return the reply."""

    def __init__(self, model_config: ModelConfig, stream: bool = True):
        self._config = model_config
        self._stream = stream

    @property
    def model_name(self) -> str:
        """The friendly name of the configured model."""
        return self._config.name

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
        try:
            if self._stream:
                return self._complete_streaming(messages, kwargs, on_text)
            return self._complete_blocking(messages, kwargs)
        except ProviderError:
            raise
        except Exception as exc:  # surface any provider failure uniformly
            raise ProviderError(f"Model request failed: {exc}") from exc

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
