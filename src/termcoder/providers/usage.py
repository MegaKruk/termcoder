"""Token and cost metering for model calls.

Every model call in termcoder flows through one client chokepoint, so a single
tracker sees everything, including compaction summaries. The numbers come from
provider-reported usage when available and from a local estimate otherwise, so
they are directionally accurate rather than exact bills. Cached prompt tokens
are tracked separately because prompt caching is the main cost lever for agent
loops: cached input is heavily discounted by cloud providers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UsageStats:
    """Accumulated usage over some scope (one turn, one session)."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    def add(
        self, prompt_tokens: int, completion_tokens: int, cached_tokens: int, cost_usd: float
    ) -> None:
        """Fold one model call into this scope."""
        self.calls += 1
        self.prompt_tokens += max(0, prompt_tokens)
        self.completion_tokens += max(0, completion_tokens)
        self.cached_tokens += max(0, cached_tokens)
        self.cost_usd += max(0.0, cost_usd)


class UsageTracker:
    """Tracks model usage at turn and session granularity.

    The tracker is shared by every client in a REPL so totals stay coherent
    across model switches and compaction calls. ``begin_turn`` resets only the
    turn scope; ``reset`` clears everything for a new or resumed session.
    """

    def __init__(self) -> None:
        self.session = UsageStats()
        self.turn = UsageStats()

    def begin_turn(self) -> None:
        """Start a fresh turn scope."""
        self.turn = UsageStats()

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        """Record one model call in both scopes."""
        self.turn.add(prompt_tokens, completion_tokens, cached_tokens, cost_usd)
        self.session.add(prompt_tokens, completion_tokens, cached_tokens, cost_usd)

    def reset(self) -> None:
        """Clear all accumulated usage, for a new or resumed session."""
        self.session = UsageStats()
        self.turn = UsageStats()
