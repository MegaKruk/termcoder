"""Optional remote control and observation for a running termcoder session.

This package lets a phone or another device on the same local network attach
to the terminal session running on the PC, watch what the agent is doing in
real time, send messages, and answer approval prompts. The terminal remains the
source of truth: the remote clients and the terminal both subscribe to one
in-process event bus and both can inject input.

The feature is optional and off by default. When it is disabled, nothing in
this package is imported by the running session.
"""

from __future__ import annotations

import importlib.util

from .bus import EventBus
from .events import (
    ApprovalRequestedEvent,
    ApprovalResolvedEvent,
    AssistantTextEvent,
    Event,
    SessionStateEvent,
    StatusEvent,
    ThinkingEvent,
    ThinkingVisibilityEvent,
    ToolFinishedEvent,
    ToolStartedEvent,
    TurnEndedEvent,
    UserMessageEvent,
)

_SERVER_PACKAGES = ("starlette", "uvicorn", "websockets")


def remote_available() -> bool:
    """Return True when the optional server dependencies are installed.

    The event bus and event types above are dependency-free and always usable;
    only the embedded web server needs the ``remote`` extra. Checked with
    ``find_spec`` so nothing heavy is imported until the server actually starts.
    """
    return all(
        importlib.util.find_spec(name) is not None for name in _SERVER_PACKAGES
    )


__all__ = [
    "EventBus",
    "Event",
    "AssistantTextEvent",
    "ThinkingEvent",
    "ThinkingVisibilityEvent",
    "ToolStartedEvent",
    "ToolFinishedEvent",
    "StatusEvent",
    "ApprovalRequestedEvent",
    "ApprovalResolvedEvent",
    "SessionStateEvent",
    "TurnEndedEvent",
    "UserMessageEvent",
    "remote_available",
]
