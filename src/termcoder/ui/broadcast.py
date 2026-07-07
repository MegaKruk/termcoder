"""A renderer that mirrors everything it prints onto the session event bus.

This is the bridge that keeps the terminal and remote clients showing the same
session. It subclasses the plain :class:`Renderer`, so the terminal output is
byte-for-byte identical to a non-remote run; each method additionally publishes
a semantic event that the embedded server relays to connected clients.

Because higher-level renderer methods (usage reports, undo summaries, command
output) are built from the primitive ones (info, plain, status, warning), those
flows broadcast automatically without further wiring.

Reasoning fragments are always published, even while the terminal hides them:
a remote client keeps them and can reveal the whole trail the moment the
thinking toggle is turned on.
"""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console

from ..context.compaction import CompactionResult
from ..providers.usage import UsageStats
from ..remote.bus import EventBus
from ..remote.events import (
    AssistantTextEvent,
    StatusEvent,
    ThinkingEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
)
from ..tools.base import ToolResult
from .renderer import Renderer, _shorten, compacted_line, usage_line


class BroadcastingRenderer(Renderer):
    """Render to the terminal and publish the same output as events."""

    def __init__(
        self,
        bus: EventBus,
        console: Console | None = None,
        thinking_visible: Callable[[], bool] | None = None,
    ):
        super().__init__(console=console, thinking_visible=thinking_visible)
        self._bus = bus

    # Plain text lines

    def info(self, text: str) -> None:
        self._bus.publish(StatusEvent(text=text, level="info"))
        super().info(text)

    def plain(self, text: str) -> None:
        self._bus.publish(StatusEvent(text=text, level="info"))
        super().plain(text)

    def status(self, text: str) -> None:
        self._bus.publish(StatusEvent(text=text, level="dim"))
        super().status(text)

    def warning(self, text: str) -> None:
        self._bus.publish(StatusEvent(text=text, level="warning"))
        super().warning(text)

    def error(self, text: str) -> None:
        self._bus.publish(StatusEvent(text=text, level="error"))
        super().error(text)

    def tool_progress(self, text: str) -> None:
        self._bus.publish(StatusEvent(text=text, level="dim"))
        super().tool_progress(text)

    # Streamed assistant output

    def stream_assistant(self, text: str) -> None:
        self._bus.publish(AssistantTextEvent(text=text))
        super().stream_assistant(text)

    def stream_thinking(self, text: str) -> None:
        self._bus.publish(ThinkingEvent(text=text))
        super().stream_thinking(text)

    # Tool activity

    def tool_started(self, name: str, raw_args: str) -> None:
        self._bus.publish(
            ToolStartedEvent(tool_name=name, arguments_preview=_shorten(raw_args))
        )
        super().tool_started(name, raw_args)

    def tool_finished(self, name: str, result: ToolResult) -> None:
        summary = result.display or ("ok" if result.ok else "failed")
        self._bus.publish(
            ToolFinishedEvent(tool_name=name, summary=summary, ok=result.ok)
        )
        super().tool_finished(name, result)

    # Session bookkeeping lines

    def compacted(self, result: CompactionResult) -> None:
        self._bus.publish(StatusEvent(text=compacted_line(result), level="dim"))
        super().compacted(result)

    def usage(self, turn: UsageStats, session: UsageStats) -> None:
        self._bus.publish(StatusEvent(text=usage_line(turn, session), level="dim"))
        super().usage(turn, session)


__all__ = ["BroadcastingRenderer"]
