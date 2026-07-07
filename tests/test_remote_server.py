"""Tests for the embedded remote web server.

These drive the Starlette application with the test client (as a context
manager so the lifespan runs) rather than a live uvicorn process. They cover
serving the client page, the token and Origin checks, the connection snapshot
order, live events crossing from another thread, and each kind of message a
client can send back: input, approval decisions, and the thinking toggle.
"""

from __future__ import annotations

import threading
import time

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from termcoder.approval.types import Decision
from termcoder.remote.bus import EventBus
from termcoder.remote.events import AssistantTextEvent, SessionStateEvent
from termcoder.remote.server import build_app, make_client_message_handler

_TOKEN = "secret-token"


def _build(bus: EventBus, thinking: dict):
    """Assemble the app with a handler that records the thinking toggle."""
    handler = make_client_message_handler(
        bus, lambda value: thinking.__setitem__("value", value)
    )

    def state() -> SessionStateEvent:
        return SessionStateEvent(
            workspace="/work",
            model="test-model",
            busy=False,
            show_thinking=thinking["value"],
        )

    return build_app(bus, _TOKEN, state, handler, notify=lambda text: None)


def test_index_serves_the_client_page():
    bus = EventBus()
    app = _build(bus, {"value": False})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "termcoder remote" in response.text


def test_websocket_rejects_a_bad_token():
    bus = EventBus()
    app = _build(bus, {"value": False})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws?token=wrong") as ws:
                ws.receive_json()


def test_websocket_rejects_cross_origin():
    bus = EventBus()
    app = _build(bus, {"value": False})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                f"/ws?token={_TOKEN}", headers={"origin": "http://evil.example"}
            ) as ws:
                ws.receive_json()


def test_snapshot_sends_state_then_backlog():
    bus = EventBus()
    app = _build(bus, {"value": False})
    bus.publish(AssistantTextEvent(text="earlier"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        with client.websocket_connect(f"/ws?token={_TOKEN}") as ws:
            first = ws.receive_json()
            assert first["kind"] == "session_state"
            assert first["workspace"] == "/work"
            second = ws.receive_json()
            assert second["kind"] == "assistant_text"
            assert second["text"] == "earlier"


def test_live_event_from_another_thread_reaches_client():
    bus = EventBus()
    app = _build(bus, {"value": False})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        with client.websocket_connect(f"/ws?token={_TOKEN}") as ws:
            ws.receive_json()  # session_state snapshot
            threading.Thread(
                target=lambda: bus.publish(AssistantTextEvent(text="live"))
            ).start()
            message = ws.receive_json()
            assert message["text"] == "live"


def test_client_user_input_reaches_the_input_queue():
    bus = EventBus()
    app = _build(bus, {"value": False})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        with client.websocket_connect(f"/ws?token={_TOKEN}") as ws:
            ws.receive_json()
            ws.send_json({"type": "user_input", "text": "do the thing"})
            time.sleep(0.2)
            queued = bus.poll_input()
            assert queued is not None
            assert queued.text == "do the thing"


def test_client_approval_resolves_as_remote():
    bus = EventBus()
    app = _build(bus, {"value": False})
    with TestClient(app, base_url="http://127.0.0.1") as client:
        with client.websocket_connect(f"/ws?token={_TOKEN}") as ws:
            ws.receive_json()
            request_id = bus.open_approval()
            ws.send_json(
                {
                    "type": "approval",
                    "request_id": request_id,
                    "decision": "approve_for_session",
                    "feedback": "",
                }
            )
            outcome, resolved_by = bus.wait_for_decision(request_id)
            assert outcome.decision is Decision.APPROVE_FOR_SESSION
            assert resolved_by == "remote"


def test_client_set_thinking_calls_the_callback():
    bus = EventBus()
    thinking = {"value": False}
    app = _build(bus, thinking)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        with client.websocket_connect(f"/ws?token={_TOKEN}") as ws:
            ws.receive_json()
            ws.send_json({"type": "set_thinking", "enabled": True})
            time.sleep(0.2)
            assert thinking["value"] is True
