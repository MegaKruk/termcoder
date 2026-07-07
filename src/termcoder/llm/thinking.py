"""Separate a model's inline reasoning from its visible answer.

Some local models do not use a dedicated reasoning field; instead they wrap
their private thinking in ``<think>`` and ``</think>`` tags inside the normal
content stream. This module splits such a stream into two channels: the visible
answer and the hidden reasoning.

The filter is stateful because tags can be split across streaming chunk
boundaries. Feed it text as it arrives; it returns the visible and thinking
fragments recognized so far and buffers only the shortest tail that might be
the start of a tag, so visible text is released as promptly as possible.
"""

from __future__ import annotations

_OPEN = "<think>"
_CLOSE = "</think>"
# The longest tag prefix we might need to hold back while awaiting more text.
_MAX_TAG = max(len(_OPEN), len(_CLOSE))


class ThinkingFilter:
    """Route ``<think>`` sections to a reasoning channel, the rest to output.

    Usage: call :meth:`feed` for each streamed fragment, then :meth:`flush`
    once the stream ends to release any trailing buffered text.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_thinking = False

    def feed(self, text: str) -> tuple[str, str]:
        """Consume a fragment and return its (visible, thinking) parts.

        Any suffix that might be the start of a tag is held back until enough
        characters arrive to decide, so a tag split across chunks is handled
        correctly.
        """
        self._buffer += text
        return self._drain(final=False)

    def flush(self) -> tuple[str, str]:
        """Release any buffered text once the stream is complete."""
        return self._drain(final=True)

    def _drain(self, final: bool) -> tuple[str, str]:
        """Emit every unambiguous fragment from the buffer.

        When ``final`` is true no tail is withheld, since no more text will
        arrive to complete a partial tag.
        """
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        while self._buffer:
            marker = _CLOSE if self._in_thinking else _OPEN
            index = self._buffer.find(marker)
            if index != -1:
                self._route(self._buffer[:index], visible_parts, thinking_parts)
                self._buffer = self._buffer[index + len(marker) :]
                self._in_thinking = not self._in_thinking
                continue
            keep = 0 if final else self._partial_tail_len(marker)
            cut = len(self._buffer) - keep
            if cut > 0:
                self._route(self._buffer[:cut], visible_parts, thinking_parts)
                self._buffer = self._buffer[cut:]
            break
        return "".join(visible_parts), "".join(thinking_parts)

    def _partial_tail_len(self, marker: str) -> int:
        """Length of the buffer suffix that could be a prefix of ``marker``.

        If the buffer ends with characters that might grow into a tag once more
        text arrives, that many characters are withheld; otherwise nothing is.
        """
        limit = min(len(self._buffer), _MAX_TAG)
        for size in range(limit, 0, -1):
            if marker.startswith(self._buffer[-size:]):
                return size
        return 0

    def _route(
        self, text: str, visible_parts: list[str], thinking_parts: list[str]
    ) -> None:
        """Append ``text`` to the channel selected by the current state."""
        if not text:
            return
        if self._in_thinking:
            thinking_parts.append(text)
        else:
            visible_parts.append(text)


def strip_thinking(text: str) -> str:
    """Return ``text`` with any complete ``<think>`` sections removed.

    Used when persisting an assistant message so stored history never contains
    the model's private reasoning. An unterminated section drops the remainder.
    """
    result: list[str] = []
    rest = text
    while True:
        start = rest.find(_OPEN)
        if start == -1:
            result.append(rest)
            break
        result.append(rest[:start])
        end = rest.find(_CLOSE, start + len(_OPEN))
        if end == -1:
            break
        rest = rest[end + len(_CLOSE) :]
    return "".join(result)


__all__ = ["ThinkingFilter", "strip_thinking"]
