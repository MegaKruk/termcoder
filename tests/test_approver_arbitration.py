"""Tests for approval arbitration between the terminal and remote clients.

The approver must let either side answer first and honor whichever wins. These
drive a real prompt_toolkit session over a pipe so both paths are exercised:
a remote client answering while the terminal prompt is blocked, the terminal
answering by keypress, and a remote session-wide approval populating the set of
tools that no longer need asking. The event history is checked for the expected
request and resolution pair.
"""

from __future__ import annotations

import io
import threading
import time

from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from termcoder.approval.types import ApprovalOutcome, ApprovalRequest, Decision
from termcoder.remote.bus import EventBus
from termcoder.ui.approver import ConsoleApprover
from termcoder.ui.renderer import Renderer


def _request() -> ApprovalRequest:
    return ApprovalRequest(
        tool_name="write_file",
        summary="Create notes.txt",
        detail="+ hello",
        detail_kind="diff",
    )


def _make(pipe):
    """Build an approver wired to a bus, plus its console sink for assertions."""
    sink = io.StringIO()
    renderer = Renderer(console=Console(file=sink, force_terminal=False, width=100))
    bus = EventBus()
    session = PromptSession(input=pipe, output=DummyOutput())
    approver = ConsoleApprover(renderer, prompt_session=session, bus=bus)
    return approver, bus, sink


def _run_async(approver, request) -> dict:
    """Run approver.request on a thread, returning a dict that will hold the outcome."""
    box: dict = {}
    thread = threading.Thread(
        target=lambda: box.__setitem__("outcome", approver.request(request))
    )
    thread.start()
    box["thread"] = thread
    return box


def test_remote_answer_wins_while_terminal_prompt_blocks():
    with create_pipe_input() as pipe:
        approver, bus, sink = _make(pipe)
        box = _run_async(approver, _request())
        time.sleep(0.4)  # let the terminal prompt block
        request_id = bus.current_pending_id()
        assert request_id is not None
        bus.resolve_approval(
            request_id, ApprovalOutcome(Decision.APPROVE), resolved_by="remote"
        )
        box["thread"].join(timeout=3)
        assert not box["thread"].is_alive()
        assert box["outcome"].decision is Decision.APPROVE
        assert _kinds(bus) == ["approval_requested", "approval_resolved"]
        assert "Resolved from remote" in sink.getvalue()


def test_terminal_keypress_wins():
    with create_pipe_input() as pipe:
        approver, bus, _sink = _make(pipe)
        box = _run_async(approver, _request())
        time.sleep(0.3)
        pipe.send_text("a\n")  # allow for the session
        box["thread"].join(timeout=3)
        assert not box["thread"].is_alive()
        assert box["outcome"].decision is Decision.APPROVE_FOR_SESSION
        resolved = bus.history()[-1].to_payload()
        assert resolved["resolved_by"] == "terminal"
        assert resolved["decision"] == "approve_for_session"


def test_session_approval_skips_future_prompts():
    with create_pipe_input() as pipe:
        approver, bus, _sink = _make(pipe)
        box = _run_async(approver, _request())
        time.sleep(0.3)
        pipe.send_text("a\n")
        box["thread"].join(timeout=3)
        assert box["outcome"].decision is Decision.APPROVE_FOR_SESSION
        # A second request for the same tool returns without prompting.
        again = approver.request(_request())
        assert again.decision is Decision.APPROVE_FOR_SESSION


def test_remote_session_approval_populates_the_set():
    with create_pipe_input() as pipe:
        approver, bus, _sink = _make(pipe)
        box = _run_async(approver, _request())
        time.sleep(0.4)
        request_id = bus.current_pending_id()
        bus.resolve_approval(
            request_id,
            ApprovalOutcome(Decision.APPROVE_FOR_SESSION),
            resolved_by="remote",
        )
        box["thread"].join(timeout=3)
        assert box["outcome"].decision is Decision.APPROVE_FOR_SESSION
        # Approved for the session from the remote side, so no prompt next time.
        again = approver.request(_request())
        assert again.decision is Decision.APPROVE_FOR_SESSION


def _kinds(bus: EventBus) -> list[str]:
    return [event.to_payload()["kind"] for event in bus.history()]
