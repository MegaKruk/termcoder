"""Tests for the renderer that mirrors terminal output onto the event bus.

These check that each rendering method also publishes the matching semantic
event, that reasoning is always published even while the terminal keeps it
hidden, and that tool activity carries the expected payload fields.
"""

from __future__ import annotations

import io

from rich.console import Console

from termcoder.remote.bus import EventBus
from termcoder.tools.base import ToolResult
from termcoder.ui.broadcast import BroadcastingRenderer


def _renderer(bus: EventBus, thinking: bool = False) -> BroadcastingRenderer:
    """Build a broadcasting renderer writing to an in-memory console."""
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    return BroadcastingRenderer(
        bus, console=console, thinking_visible=lambda: thinking
    )


def _kinds(bus: EventBus) -> list[str]:
    return [event.to_payload()["kind"] for event in bus.history()]


def test_info_publishes_status_event():
    bus = EventBus()
    _renderer(bus).info("hello")
    payload = bus.history()[-1].to_payload()
    assert payload["kind"] == "status"
    assert payload["level"] == "info"
    assert payload["text"] == "hello"


def test_status_publishes_dim_level():
    bus = EventBus()
    _renderer(bus).status("working")
    assert bus.history()[-1].to_payload()["level"] == "dim"


def test_stream_assistant_publishes_assistant_text():
    bus = EventBus()
    _renderer(bus).stream_assistant("chunk")
    payload = bus.history()[-1].to_payload()
    assert payload["kind"] == "assistant_text"
    assert payload["text"] == "chunk"


def test_thinking_is_published_even_when_hidden_in_terminal():
    bus = EventBus()
    sink = io.StringIO()
    renderer = BroadcastingRenderer(
        bus, console=Console(file=sink, force_terminal=False, width=100),
        thinking_visible=lambda: False,
    )
    renderer.stream_thinking("reasoning")
    # Published for remote clients...
    payload = bus.history()[-1].to_payload()
    assert payload["kind"] == "thinking"
    assert payload["text"] == "reasoning"
    # ...but not printed to the terminal while hidden.
    assert sink.getvalue() == ""


def test_tool_started_payload_fields():
    bus = EventBus()
    _renderer(bus).tool_started("write_file", '{"path": "a.txt"}')
    payload = bus.history()[-1].to_payload()
    assert payload["kind"] == "tool_started"
    assert payload["tool_name"] == "write_file"
    assert "a.txt" in payload["arguments_preview"]


def test_tool_finished_uses_display_and_ok_flag():
    bus = EventBus()
    _renderer(bus).tool_finished(
        "read_file", ToolResult(content="body", ok=True, display="read a.txt")
    )
    payload = bus.history()[-1].to_payload()
    assert payload["kind"] == "tool_finished"
    assert payload["tool_name"] == "read_file"
    assert payload["summary"] == "read a.txt"
    assert payload["ok"] is True


def test_tool_finished_falls_back_to_ok_or_failed_summary():
    bus = EventBus()
    _renderer(bus).tool_finished("run", ToolResult(content="oops", ok=False))
    payload = bus.history()[-1].to_payload()
    assert payload["summary"] == "failed"
    assert payload["ok"] is False
