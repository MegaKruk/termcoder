"""A terminal prompt that a background event can cancel.

When a remote client is attached, two prompts must be answerable from either
side: the main input prompt and the approval prompt. The terminal side blocks
in prompt_toolkit; when the remote side acts first, the blocked prompt must be
dismissed so the session can move on.

prompt_toolkit applications run on an asyncio loop in the prompting thread. The
supported way to stop one from another thread is to schedule ``app.exit`` on
that loop with ``call_soon_threadsafe``. This module wraps that dance behind a
small :class:`CancelToken` so callers never touch prompt_toolkit internals.

If anything about the cancellation plumbing fails (an unusual terminal, a
prompt_toolkit change), the prompt simply behaves like a normal blocking
prompt: the feature degrades, the session does not break.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from prompt_toolkit import PromptSession

# How long a cancel attempt keeps looking for a running prompt application.
_CANCEL_RETRIES = 50
_CANCEL_INTERVAL_S = 0.02


class PromptInterrupted(Exception):
    """Raised by an interruptible prompt when it was cancelled externally."""


class CancelToken:
    """A one-shot, thread-safe signal with listeners.

    ``trip`` may be called from any thread. Listeners added after the token has
    tripped are invoked immediately. Each listener is invoked at most once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tripped = False
        self._listeners: list[Callable[[], None]] = []

    @property
    def tripped(self) -> bool:
        with self._lock:
            return self._tripped

    def trip(self) -> None:
        """Fire the token, notifying all current listeners once."""
        with self._lock:
            if self._tripped:
                return
            self._tripped = True
            listeners = list(self._listeners)
            self._listeners.clear()
        for listener in listeners:
            listener()

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Register a listener; returns a function that removes it again.

        If the token already tripped, the listener runs immediately on the
        calling thread and the returned remover is a no-op.
        """
        with self._lock:
            if not self._tripped:
                self._listeners.append(listener)

                def remove() -> None:
                    with self._lock:
                        if listener in self._listeners:
                            self._listeners.remove(listener)

                return remove
        listener()
        return lambda: None


def prompt_interruptible(
    session: PromptSession, message: str, cancel: CancelToken | None = None
) -> str:
    """Prompt for one line of input, unless ``cancel`` trips first.

    Behaves exactly like ``session.prompt(message)`` when no token is given.
    With a token, a trip from any thread dismisses the prompt and this function
    raises :class:`PromptInterrupted`. EOFError and KeyboardInterrupt propagate
    unchanged so callers keep their usual handling.
    """
    if cancel is None:
        return session.prompt(message)
    if cancel.tripped:
        raise PromptInterrupted()

    def on_cancel() -> None:
        # Never block the tripping thread (it may be the server's event loop);
        # the wait-for-app-and-exit dance runs on its own small thread.
        threading.Thread(
            target=_dismiss_prompt, args=(session,), daemon=True,
            name="termcoder-prompt-cancel",
        ).start()

    remove = cancel.add_listener(on_cancel)
    try:
        return session.prompt(message)
    finally:
        remove()


def _dismiss_prompt(session: PromptSession) -> None:
    """Ask the session's running prompt application to exit, from any thread.

    Retries briefly because the trip can race the prompt starting up. Gives up
    quietly if no prompt is running; in that case there is nothing to dismiss.
    """
    for _ in range(_CANCEL_RETRIES):
        app = getattr(session, "app", None)
        loop = getattr(app, "loop", None)
        if app is not None and loop is not None and app.is_running:
            try:
                loop.call_soon_threadsafe(_exit_app, app)
            except RuntimeError:
                pass
            return
        time.sleep(_CANCEL_INTERVAL_S)


def _exit_app(app) -> None:
    """Exit a prompt application with PromptInterrupted, tolerating races."""
    try:
        if app.is_running:
            app.exit(exception=PromptInterrupted(), style="class:aborting")
    except Exception:
        # The prompt finished in the same instant; nothing to do.
        pass


__all__ = ["CancelToken", "PromptInterrupted", "prompt_interruptible"]
