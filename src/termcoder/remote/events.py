"""Semantic events broadcast from a running session to remote observers.

These events describe what the agent is doing in terms the user interface can
render directly, rather than raw terminal bytes. Both the terminal and any
connected remote client consume the same events, so what the phone shows always
matches the session on the PC.

Every event is a small frozen dataclass with a ``kind`` discriminator and a
``to_payload`` method that returns a JSON-serializable dict. Keeping the wire
format in one place makes the protocol easy to extend in later phases: add a
new event class, give it a unique ``kind``, and both server and client can grow
to handle it without touching the transport.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def _now_ms() -> int:
    """Return the current wall-clock time in milliseconds."""
    return int(time.time() * 1000)


@dataclass(frozen=True)
class Event:
    """Base class for all broadcast events.

    The ``kind`` string identifies the event type on the wire. The timestamp is
    filled in automatically so clients can order or age events if they wish.
    """

    kind: str = field(init=False, default="event")
    at_ms: int = field(default_factory=_now_ms)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this event."""
        return {"kind": self.kind, "at_ms": self.at_ms}


@dataclass(frozen=True)
class AssistantTextEvent(Event):
    """A streamed fragment of visible assistant output."""

    kind: str = field(init=False, default="assistant_text")
    text: str = ""

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["text"] = self.text
        return data


@dataclass(frozen=True)
class ThinkingEvent(Event):
    """A streamed fragment of the model's internal reasoning.

    Reasoning is hidden by default. Clients decide whether to display it based
    on their own toggle, but the event is always broadcast so a client can show
    it the moment the toggle is turned on, and so a slow model does not look
    frozen to a remote observer.
    """

    kind: str = field(init=False, default="thinking")
    text: str = ""

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["text"] = self.text
        return data


@dataclass(frozen=True)
class ToolStartedEvent(Event):
    """A tool call has started, with a preview of its arguments."""

    kind: str = field(init=False, default="tool_started")
    tool_name: str = ""
    arguments_preview: str = ""

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["tool_name"] = self.tool_name
        data["arguments_preview"] = self.arguments_preview
        return data


@dataclass(frozen=True)
class ToolFinishedEvent(Event):
    """A tool call has finished, with a short result summary."""

    kind: str = field(init=False, default="tool_finished")
    tool_name: str = ""
    summary: str = ""
    ok: bool = True

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["tool_name"] = self.tool_name
        data["summary"] = self.summary
        data["ok"] = self.ok
        return data


@dataclass(frozen=True)
class StatusEvent(Event):
    """A general status or informational line, such as a warning or notice."""

    kind: str = field(init=False, default="status")
    text: str = ""
    level: str = "info"  # one of: "info", "warning", "error", "dim"

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["text"] = self.text
        data["level"] = self.level
        return data


@dataclass(frozen=True)
class UserMessageEvent(Event):
    """A user message accepted by the session, echoed to all clients.

    Broadcasting the accepted input keeps every observer in sync about what was
    asked, no matter which client (terminal or a phone) submitted it.
    """

    kind: str = field(init=False, default="user_message")
    text: str = ""
    source: str = "terminal"  # "terminal" or "remote"

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["text"] = self.text
        data["source"] = self.source
        return data


@dataclass(frozen=True)
class ApprovalRequestedEvent(Event):
    """An action is waiting for approval.

    Carries everything a client needs to show the pending action and let the
    user decide: the summary, the detail body and its kind (plain text, a diff,
    or a shell command), and whether the action is destructive. The
    ``request_id`` ties a later resolution back to this request.
    """

    kind: str = field(init=False, default="approval_requested")
    request_id: str = ""
    tool_name: str = ""
    summary: str = ""
    detail: str | None = None
    detail_kind: str = "text"
    destructive: bool = False
    note: str | None = None

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["request_id"] = self.request_id
        data["tool_name"] = self.tool_name
        data["summary"] = self.summary
        data["detail"] = self.detail
        data["detail_kind"] = self.detail_kind
        data["destructive"] = self.destructive
        data["note"] = self.note
        return data


@dataclass(frozen=True)
class ApprovalResolvedEvent(Event):
    """A pending approval has been decided, by whichever client answered first.

    Broadcast to every client so the prompt can be cleared everywhere and the
    observers learn how it was resolved and by whom.
    """

    kind: str = field(init=False, default="approval_resolved")
    request_id: str = ""
    decision: str = ""  # "approve", "approve_for_session", or "reject"
    resolved_by: str = "terminal"  # "terminal" or "remote"

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["request_id"] = self.request_id
        data["decision"] = self.decision
        data["resolved_by"] = self.resolved_by
        return data


@dataclass(frozen=True)
class TurnEndedEvent(Event):
    """The agent has finished handling a turn and is ready for new input."""

    kind: str = field(init=False, default="turn_ended")

    def to_payload(self) -> dict[str, Any]:
        return super().to_payload()


@dataclass(frozen=True)
class SessionStateEvent(Event):
    """A snapshot of high-level session state sent to a client on connect.

    A newly connected client has missed the events so far, so it is given the
    essentials up front: which workspace and model are active, and whether a
    turn is currently running.
    """

    kind: str = field(init=False, default="session_state")
    workspace: str = ""
    model: str = ""
    busy: bool = False
    show_thinking: bool = False

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["workspace"] = self.workspace
        data["model"] = self.model
        data["busy"] = self.busy
        data["show_thinking"] = self.show_thinking
        return data


@dataclass(frozen=True)
class ThinkingVisibilityEvent(Event):
    """The session-wide reasoning display toggle changed.

    The toggle is shared: switching it from the terminal or from any client
    changes it for everyone, and this event keeps all views in sync.
    """

    kind: str = field(init=False, default="thinking_visibility")
    enabled: bool = False

    def to_payload(self) -> dict[str, Any]:
        data = super().to_payload()
        data["enabled"] = self.enabled
        return data
