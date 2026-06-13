"""Conversation context management and compaction.

A long session eventually fills the model's context window. This module keeps
the working context within budget by summarizing older turns into a compact
summary and keeping only the most recent turns verbatim. The full transcript is
never altered on disk; the summary lives in the session metadata, so nothing is
lost and resuming a session still has the complete record.

Compaction always cuts on turn boundaries (a turn starts with a user message),
which guarantees that an assistant message carrying tool calls is never
separated from its tool-result messages.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import ProviderError
from ..sessions.store import Session
from .tokens import TokenCounter

# Rough allowance for the system prompt, which is not stored in the transcript.
_SYSTEM_PROMPT_TOKEN_ESTIMATE = 500

_SUMMARY_SYSTEM_PROMPT = (
    "You write concise, factual summaries of an in-progress software "
    "engineering session between a user and a coding assistant. Capture the "
    "user's goals, decisions made, files created or changed and how, important "
    "facts discovered, and any open tasks. Prefer specifics over generalities. "
    "Do not invent details. Write plain prose with no preamble."
)


@dataclass
class CompactionResult:
    """A record of one compaction, for reporting to the user."""

    summarized_turns: int
    before_tokens: int
    after_tokens: int


def _is_turn_start(message: dict) -> bool:
    return message.get("role") == "user"


def _render_transcript(messages: list[dict]) -> str:
    """Render messages as readable text for the summarization request."""
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = (message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments", "")
            lines.append(f"assistant called tool {name} with {arguments}")
    return "\n".join(lines)


class ContextManager:
    """Builds the working context and compacts it when it grows too large.

    The manager owns its summarizer client. That keeps the call sites simple
    and lets the summarizer be a different, cheaper model than the active one:
    the summarization request carries the whole region being folded away, so it
    is the largest single prompt termcoder ever sends. With no summarizer the
    manager still tracks sizes but cannot compact.
    """

    def __init__(
        self,
        auto_compact: bool,
        compact_threshold: float,
        keep_recent_turns: int,
        context_window: int | None,
        summarizer=None,
        token_counter: TokenCounter | None = None,
        system_prompt_tokens: int | None = None,
    ):
        self._auto_compact = auto_compact
        self._threshold = compact_threshold
        self._keep_recent_turns = max(1, keep_recent_turns)
        self._context_window = context_window
        self._summarizer = summarizer
        self._tokens = token_counter or TokenCounter()
        self._system_prompt_tokens = system_prompt_tokens

    def summary_text(self, session: Session) -> str | None:
        """Return the stored summary for the session, if any."""
        return session.meta.summary

    def tail_messages(self, session: Session) -> list[dict]:
        """Return the verbatim messages not covered by the summary."""
        return session.messages[session.meta.summary_through :]

    def estimate_tokens(self, session: Session) -> int:
        """Estimate the token size of the current working context."""
        if self._system_prompt_tokens is not None:
            total = self._system_prompt_tokens
        else:
            total = _SYSTEM_PROMPT_TOKEN_ESTIMATE
        summary = session.meta.summary
        if summary:
            total += self._tokens.count_text(summary)
        total += self._tokens.count_messages(self.tail_messages(session))
        return total

    def maybe_compact(self, session: Session) -> CompactionResult | None:
        """Compact automatically if enabled and the budget is exceeded."""
        if not self._auto_compact or not self._context_window:
            return None
        budget = int(self._context_window * self._threshold)
        if self.estimate_tokens(session) <= budget:
            return None
        try:
            return self._compact(session, instructions=None)
        except ProviderError:
            return None

    def force_compact(
        self, session: Session, instructions: str | None = None
    ) -> CompactionResult | None:
        """Compact on demand, regardless of the current size.

        Unlike the automatic path, provider errors are allowed to propagate so
        the caller can report that a manual compaction failed.
        """
        return self._compact(session, instructions=instructions)

    def _compact(
        self, session: Session, instructions: str | None
    ) -> CompactionResult | None:
        if self._summarizer is None:
            return None
        messages = session.messages
        start = session.meta.summary_through
        pending = messages[start:]
        cut = self._cut_index(pending)
        if cut <= 0:
            return None

        to_summarize = pending[:cut]
        before_tokens = self.estimate_tokens(session)
        summary = self._summarize(session.meta.summary, to_summarize, instructions)
        if not summary:
            return None

        session.set_summary(summary, start + cut)
        after_tokens = self.estimate_tokens(session)
        summarized_turns = sum(1 for message in to_summarize if _is_turn_start(message))
        return CompactionResult(
            summarized_turns=summarized_turns,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
        )

    def _cut_index(self, pending: list[dict]) -> int:
        """Index in ``pending`` up to which messages should be summarized.

        Keeps the last ``keep_recent_turns`` turns verbatim. Returns 0 when
        there are not enough older turns to compact.
        """
        turn_starts = [i for i, message in enumerate(pending) if _is_turn_start(message)]
        if len(turn_starts) <= self._keep_recent_turns:
            return 0
        return turn_starts[-self._keep_recent_turns]

    def _summarize(
        self,
        previous_summary: str | None,
        messages: list[dict],
        instructions: str | None,
    ) -> str:
        parts: list[str] = []
        if previous_summary:
            parts.append("Summary so far:\n" + previous_summary)
        parts.append("New conversation to fold into the summary:\n" + _render_transcript(messages))
        if instructions:
            parts.append("Pay particular attention to: " + instructions)
        parts.append("Write the updated summary.")
        request = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]
        result = self._summarizer.complete(request, tools=None, on_text=None)
        return (result.message.content or "").strip()
