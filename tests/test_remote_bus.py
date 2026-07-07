"""Tests for the session event bus.

The bus is the single point that both the terminal and remote clients talk
through, so these cover history and subscriber fan-out, the remote input queue
and its wake-up listener, and the first-responder-wins approval arbitration
including the watcher that dismisses a losing prompt.
"""

from __future__ import annotations

import threading

from termcoder.approval.types import ApprovalOutcome, Decision
from termcoder.remote.bus import EventBus
from termcoder.remote.events import AssistantTextEvent, StatusEvent


def test_history_records_published_events_in_order():
    bus = EventBus()
    bus.publish(AssistantTextEvent(text="one"))
    bus.publish(StatusEvent(text="two", level="info"))
    kinds = [event.to_payload()["kind"] for event in bus.history()]
    assert kinds == ["assistant_text", "status"]


def test_history_is_capped_at_the_limit():
    bus = EventBus(history_limit=3)
    for index in range(5):
        bus.publish(AssistantTextEvent(text=str(index)))
    texts = [event.to_payload()["text"] for event in bus.history()]
    assert texts == ["2", "3", "4"]


def test_submit_input_is_polled_in_order():
    bus = EventBus()
    bus.submit_input("first")
    bus.submit_input("second")
    assert bus.poll_input().text == "first"
    assert bus.poll_input().text == "second"
    assert bus.poll_input() is None


def test_input_listener_fires_on_submit_and_can_be_removed():
    bus = EventBus()
    calls: list[int] = []
    remove = bus.add_input_listener(lambda: calls.append(1))
    bus.submit_input("hello")
    assert calls == [1]
    remove()
    bus.submit_input("again")
    assert calls == [1]


def test_resolve_approval_first_responder_wins():
    bus = EventBus()
    request_id = bus.open_approval()
    first = bus.resolve_approval(
        request_id, ApprovalOutcome(Decision.APPROVE), resolved_by="remote"
    )
    second = bus.resolve_approval(
        request_id, ApprovalOutcome(Decision.REJECT), resolved_by="terminal"
    )
    assert first is True
    assert second is False


def test_wait_for_decision_returns_outcome_and_resolver():
    bus = EventBus()
    request_id = bus.open_approval()
    results: list[tuple] = []
    waiter = threading.Thread(
        target=lambda: results.append(bus.wait_for_decision(request_id))
    )
    waiter.start()
    bus.resolve_approval(
        request_id, ApprovalOutcome(Decision.APPROVE_FOR_SESSION), resolved_by="remote"
    )
    waiter.join(timeout=2)
    assert not waiter.is_alive()
    outcome, resolved_by = results[0]
    assert outcome.decision is Decision.APPROVE_FOR_SESSION
    assert resolved_by == "remote"


def test_watch_approval_invokes_callback_on_resolution():
    bus = EventBus()
    request_id = bus.open_approval()
    fired = threading.Event()
    bus.watch_approval(request_id, fired.set)
    assert not fired.is_set()
    bus.resolve_approval(
        request_id, ApprovalOutcome(Decision.REJECT), resolved_by="terminal"
    )
    assert fired.wait(timeout=2) is True


def test_watch_approval_fires_immediately_for_unknown_id():
    bus = EventBus()
    fired = threading.Event()
    bus.watch_approval("no-such-id", fired.set)
    assert fired.is_set()


def test_current_pending_id_tracks_open_and_resolved():
    bus = EventBus()
    assert bus.current_pending_id() is None
    request_id = bus.open_approval()
    assert bus.current_pending_id() == request_id
    bus.resolve_approval(
        request_id, ApprovalOutcome(Decision.APPROVE), resolved_by="remote"
    )
    assert bus.current_pending_id() is None
