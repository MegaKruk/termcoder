"""Token counting.

Used to decide when a conversation is approaching the model's context window.
If tiktoken is installed it is used for a closer estimate; otherwise a simple
character-based heuristic is used. The heuristic is deliberately rough because
it only needs to be good enough to trigger compaction ahead of the limit, not
to bill tokens.
"""

from __future__ import annotations

# Per-message overhead (role, separators) added on top of content tokens.
_MESSAGE_OVERHEAD_TOKENS = 4
# Average characters per token used by the fallback heuristic.
_CHARS_PER_TOKEN = 4


class TokenCounter:
    """Estimate the number of tokens in text and message lists."""

    def __init__(self) -> None:
        self._encoder = self._load_encoder()

    @staticmethod
    def _load_encoder():
        try:
            import tiktoken
        except Exception:
            return None
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    def count_text(self, text: str) -> int:
        """Estimate the token count of a single string."""
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text))
        return max(1, (len(text) + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN)

    def count_messages(self, messages: list[dict]) -> int:
        """Estimate the token count of a list of chat messages."""
        total = 0
        for message in messages:
            total += _MESSAGE_OVERHEAD_TOKENS
            total += self.count_text(message.get("content") or "")
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                total += self.count_text(function.get("name", ""))
                total += self.count_text(function.get("arguments", ""))
        return total
