"""The embedded web server that lets remote clients attach to a session.

The server runs uvicorn on a background thread with its own asyncio loop while
the terminal session keeps the main thread. It serves two things:

* ``GET /`` returns the single-page client (plain HTML and JavaScript, no build
  step), which a phone opens by visiting the printed LAN address;
* ``/ws`` is the WebSocket that carries session events to the client and the
  client's messages (inputs, approval decisions, the thinking toggle) back.

Security is sized for a trusted local network: the socket binds to the
configured interface and every WebSocket connection must present the session
token, with an Origin consistency check on top for browsers.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import secrets
import socket
import threading
from collections.abc import Callable
from urllib.parse import urlsplit

from ..approval.types import ApprovalOutcome, Decision
from .bus import EventBus
from .events import SessionStateEvent

_START_TIMEOUT_S = 5.0
_STOP_TIMEOUT_S = 5.0
_TOKEN_BYTES = 16

_DECISIONS = {
    "approve": Decision.APPROVE,
    "approve_for_session": Decision.APPROVE_FOR_SESSION,
    "reject": Decision.REJECT,
}


def generate_token() -> str:
    """Return a fresh URL-safe session token."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def lan_ip() -> str:
    """Best-effort discovery of this machine's LAN address.

    Opening a UDP socket toward a public address selects the outbound
    interface without sending any packet; its local name is the LAN IP. Falls
    back to the loopback address when the machine has no route at all.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def _hostname_of(value: str) -> str:
    """Extract the bare hostname from a ``host[:port]`` header value."""
    if not value:
        return ""
    parsed = urlsplit(f"//{value}")
    return parsed.hostname or ""


def build_app(
    bus: EventBus,
    token: str,
    session_state: Callable[[], SessionStateEvent],
    on_client_message: Callable[[dict], None],
    notify: Callable[[str], None],
):
    """Assemble the Starlette application serving the client page and socket.

    Kept as a factory so tests can drive it with Starlette's TestClient without
    running uvicorn. The callables connect the server to the session: a state
    snapshot for newly connected clients, a handler for their messages, and a
    notifier for connection lifecycle lines.
    """
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocket, WebSocketDisconnect

    from .webpage import PAGE_HTML

    async def index(request):
        return HTMLResponse(PAGE_HTML)

    def _authorized(websocket: WebSocket) -> bool:
        """Gate the socket on the session token, plus an Origin sanity check.

        The token, compared in constant time, is the real protection: a hostile
        web page can neither know it nor read it cross-origin, which is what
        defeats drive-by connections and DNS rebinding. When a browser does
        send an Origin header it must agree with the Host header, so a page
        served from elsewhere cannot ride along even with a leaked address.
        """
        supplied = websocket.query_params.get("token", "")
        if not hmac.compare_digest(supplied, token):
            return False
        origin = websocket.headers.get("origin", "")
        if origin:
            host = _hostname_of(websocket.headers.get("host", ""))
            origin_host = urlsplit(origin).hostname or ""
            if not host or origin_host != host:
                return False
        return True

    async def _send_snapshot(websocket: WebSocket) -> None:
        await websocket.send_json(session_state().to_payload())
        for event in bus.history():
            await websocket.send_json(event.to_payload())

    async def _pump_events(websocket: WebSocket, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            await websocket.send_json(event.to_payload())

    async def ws_endpoint(websocket: WebSocket) -> None:
        if not _authorized(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        client = websocket.client
        label = f"{client.host}:{client.port}" if client else "unknown"
        notify(f"remote: client connected ({label})")
        queue = bus.subscribe()
        sender = asyncio.create_task(_pump_events(websocket, queue))
        try:
            await _send_snapshot(websocket)
            while True:
                try:
                    message = await websocket.receive_json()
                except WebSocketDisconnect:
                    break
                if isinstance(message, dict):
                    on_client_message(message)
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
            bus.unsubscribe(queue)
            notify(f"remote: client disconnected ({label})")

    @contextlib.asynccontextmanager
    async def lifespan(app):
        bus.bind_loop(asyncio.get_running_loop())
        yield

    return Starlette(
        routes=[
            Route("/", index),
            WebSocketRoute("/ws", ws_endpoint),
        ],
        lifespan=lifespan,
    )


def make_client_message_handler(
    bus: EventBus, set_thinking: Callable[[bool], None]
) -> Callable[[dict], None]:
    """Build the dispatcher for messages arriving from remote clients.

    Kept separate from the app factory so the message protocol can be tested
    directly. Unknown or malformed messages are ignored: a personal LAN tool
    gains nothing from being strict with its only user.
    """

    def handle(message: dict) -> None:
        kind = message.get("type")
        if kind == "user_input":
            text = str(message.get("text", "")).strip()
            if text:
                bus.submit_input(text)
        elif kind == "approval":
            decision = _DECISIONS.get(str(message.get("decision", "")))
            if decision is None:
                return
            feedback = str(message.get("feedback", "")).strip() or None
            bus.resolve_approval(
                str(message.get("request_id", "")),
                ApprovalOutcome(decision, feedback=feedback),
                resolved_by="remote",
            )
        elif kind == "set_thinking":
            set_thinking(bool(message.get("enabled", False)))

    return handle


class RemoteServer:
    """Run the embedded server on a background thread for one session."""

    def __init__(
        self,
        bus: EventBus,
        host: str,
        port: int,
        token: str,
        session_state: Callable[[], SessionStateEvent],
        set_thinking: Callable[[bool], None],
        notify: Callable[[str], None],
    ):
        self._bus = bus
        self._host = host
        self._port = port
        self._token = token
        self._session_state = session_state
        self._set_thinking = set_thinking
        self._notify = notify
        self._server = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    @property
    def token(self) -> str:
        return self._token

    @property
    def url(self) -> str:
        """The address to open on a phone, including the access token."""
        host = self._host if self._host not in {"0.0.0.0", ""} else lan_ip()
        return f"http://{host}:{self._port}/?token={self._token}"

    def start(self) -> bool:
        """Start serving in the background; True when the socket is listening."""
        import uvicorn

        handler = make_client_message_handler(self._bus, self._set_thinking)
        app = build_app(
            self._bus, self._token, self._session_state, handler, self._notify
        )
        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="error",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="termcoder-remote-server"
        )
        self._thread.start()
        deadline = _START_TIMEOUT_S
        while deadline > 0:
            if self._server.started:
                return True
            if not self._thread.is_alive():
                break
            threading.Event().wait(0.05)
            deadline -= 0.05
        if self.error is None:
            self.error = f"could not listen on {self._host}:{self._port}"
        return False

    def _run(self) -> None:
        try:
            self._server.run()
        except SystemExit:
            self.error = f"could not listen on {self._host}:{self._port}"
        except Exception as exc:
            self.error = str(exc)

    def stop(self) -> None:
        """Ask the server to exit and wait briefly for the thread to finish."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=_STOP_TIMEOUT_S)


__all__ = [
    "RemoteServer",
    "build_app",
    "make_client_message_handler",
    "generate_token",
    "lan_ip",
]
