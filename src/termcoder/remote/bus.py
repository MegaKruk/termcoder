"""The in-process event bus that connects the session to remote clients.

The bus is the single meeting point between two worlds that run on different
threads:

* the agent loop, which runs on the main thread and publishes events (assistant
  text, tool activity, approval requests) and consumes remote input;
* the embedded web server, which runs on a background asyncio event loop, fans
  events out to connected WebSocket clients, and feeds their messages back in.

It owns three responsibilities, each kept separate:

* broadcast: deliver every published event to all current subscribers, and
  keep a bounded backlog so a client that connects mid-session can catch up;
* input: collect messages submitted by remote clients into a queue the terminal
  loop can poll, so remote input is handled on the same thread as typed input;
* approval arbitration: hold at most one pending approval at a time and let the
  first responder (terminal or a remote client) decide it, ignoring the rest.

Thread-safety is explicit. Publishing and resolving may be called from any
thread; delivery to async subscribers is marshalled onto the server's event
loop with ``run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ..approval.types import ApprovalOutcome, Decision
from .events import Event

_DEFAULT_HISTORY = 200


@dataclass
class RemoteInput:
    """A message submitted by a remote client for the session to handle."""

    text: str


@dataclass
class _PendingApproval:
    """A single in-flight approval awaiting a decision from any client."""

    request_id: str
    event: threading.Event
    outcome: ApprovalOutcome | None = None
    resolved_by: str = "terminal"


class EventBus:
    """Broadcast session events and collect remote input and decisions.

    A single instance is shared between the terminal session and the embedded
    web server for the lifetime of one run.
    """

    def __init__(self, history_limit: int = _DEFAULT_HISTORY):
        self._subscribers: set[asyncio.Queue] = set()
        self._history: deque[Event] = deque(maxlen=history_limit)
        self._input: queue.Queue = queue.Queue()
        self._input_listeners: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: _PendingApproval | None = None
        self._pending_counter = 0

    # Server loop wiring

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Record the server's event loop used to deliver events to clients.

        Called once by the server thread after its loop starts. Delivery to
        async subscribers is scheduled onto this loop from the publishing
        thread.
        """
        with self._lock:
            self._loop = loop

    # Subscription (called on the server loop)

    def subscribe(self) -> asyncio.Queue:
        """Register a new subscriber queue and return it.

        Must be called from the server event loop. The returned queue receives
        every event published from now on.
        """
        subscriber: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: asyncio.Queue) -> None:
        """Remove a subscriber that has disconnected."""
        with self._lock:
            self._subscribers.discard(subscriber)

    def history(self) -> list[Event]:
        """Return the recent event backlog for catching up a new client."""
        with self._lock:
            return list(self._history)

    # Publishing (may be called from any thread)

    def publish(self, event: Event) -> None:
        """Broadcast an event to all subscribers and record it in history.

        Safe to call from the agent thread. If the server loop is running,
        delivery to each subscriber queue is scheduled on that loop; if no loop
        is bound yet (no server, or not started), the event is still kept in
        history so a later subscriber can see it.
        """
        with self._lock:
            self._history.append(event)
            loop = self._loop
            subscribers = list(self._subscribers)
        if loop is None or loop.is_closed():
            return
        for subscriber in subscribers:
            self._schedule_put(loop, subscriber, event)

    @staticmethod
    def _schedule_put(
        loop: asyncio.AbstractEventLoop, subscriber: asyncio.Queue, event: Event
    ) -> None:
        """Schedule ``subscriber.put(event)`` on the server loop, safely."""

        def _put() -> None:
            subscriber.put_nowait(event)

        try:
            loop.call_soon_threadsafe(_put)
        except RuntimeError:
            # The loop was stopping or closed between the check and the call.
            pass

    # Remote input (producer: server loop; consumer: terminal thread)

    def submit_input(self, text: str) -> None:
        """Queue a message from a remote client for the session to handle.

        Registered input listeners are notified after the message is queued so
        a blocked terminal prompt can wake up and collect it.
        """
        self._input.put(RemoteInput(text=text))
        with self._lock:
            listeners = list(self._input_listeners)
        for listener in listeners:
            listener()

    def add_input_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Register a callback fired whenever remote input arrives.

        Returns a function that removes the listener again. The callback runs
        on whichever thread submitted the input, so it must be quick.
        """
        with self._lock:
            self._input_listeners.append(listener)

        def remove() -> None:
            with self._lock:
                if listener in self._input_listeners:
                    self._input_listeners.remove(listener)

        return remove

    def poll_input(self) -> RemoteInput | None:
        """Return the next queued remote message, or None if there is none.

        Non-blocking, intended to be polled by the terminal loop between typed
        inputs.
        """
        try:
            return self._input.get_nowait()
        except queue.Empty:
            return None

    def wait_input(self, timeout: float) -> RemoteInput | None:
        """Block up to ``timeout`` seconds for the next remote message."""
        try:
            return self._input.get(timeout=timeout)
        except queue.Empty:
            return None

    # Approval arbitration

    def open_approval(self) -> str:
        """Open a pending approval and return its request id.

        Called on the agent thread just before it blocks waiting for a
        decision. Only one approval is open at a time because the agent handles
        tool calls one by one.
        """
        with self._lock:
            self._pending_counter += 1
            request_id = f"appr-{self._pending_counter}"
            self._pending = _PendingApproval(
                request_id=request_id, event=threading.Event()
            )
        return request_id

    def wait_for_decision(self, request_id: str) -> tuple[ApprovalOutcome, str]:
        """Block until the given approval is resolved, then return the outcome.

        Returns the decided outcome and a label for who resolved it
        ("terminal" or "remote"). Runs on the agent thread.
        """
        with self._lock:
            pending = self._pending
        if pending is None or pending.request_id != request_id:
            # Already resolved or unknown; default to a safe rejection.
            return ApprovalOutcome(Decision.REJECT), "terminal"
        pending.event.wait()
        with self._lock:
            outcome = pending.outcome or ApprovalOutcome(Decision.REJECT)
            resolved_by = pending.resolved_by
            self._pending = None
        return outcome, resolved_by

    def resolve_approval(
        self, request_id: str, outcome: ApprovalOutcome, resolved_by: str
    ) -> bool:
        """Resolve a pending approval if it is still open and matches the id.

        Returns True if this call is the one that resolved it, False if there
        was nothing to resolve or another responder won first. This is the
        first-responder-wins arbitration between the terminal and remote
        clients.
        """
        with self._lock:
            pending = self._pending
            if pending is None or pending.request_id != request_id:
                return False
            if pending.event.is_set():
                return False
            pending.outcome = outcome
            pending.resolved_by = resolved_by
            pending.event.set()
            return True

    def watch_approval(self, request_id: str, callback: Callable[[], None]) -> None:
        """Invoke ``callback`` once the given approval is resolved.

        The callback runs on a small helper thread (or immediately, if the
        approval is already resolved or unknown). The terminal approver uses
        this to dismiss its blocking prompt when a remote client answers first.
        """
        with self._lock:
            pending = self._pending
        if pending is None or pending.request_id != request_id:
            callback()
            return

        def _wait() -> None:
            pending.event.wait()
            callback()

        threading.Thread(
            target=_wait, daemon=True, name="termcoder-approval-watch"
        ).start()

    def current_pending_id(self) -> str | None:
        """Return the id of the open approval, if any (for late joiners)."""
        with self._lock:
            if self._pending is None or self._pending.event.is_set():
                return None
            return self._pending.request_id


__all__ = ["EventBus", "RemoteInput"]
