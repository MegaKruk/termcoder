"""The agent loop.

This is the deterministic core: assemble the message list, call the model,
render any streamed text, then either finish (no tool calls) or run each
requested tool and feed the results back. Every tool call produces a tool
result message, including on error, so the conversation stays valid for the
next model turn.

Two Phase 2 concerns are woven in here without changing that shape:

* Before each model call the context manager may compact older turns so the
  conversation stays within the model's context window. The summary it produces
  is folded into the system message; the transcript on disk is untouched.
* At the start of every turn a snapshot group is opened so the file tools can
  record prior file states, which powers undo.

The loop talks to the terminal through a small :class:`AgentUI` protocol rather
than importing the UI package directly. This keeps the dependency one-way (the
UI imports the agent, not the reverse) and makes the loop easy to test.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..context.compaction import CompactionResult, ContextManager
from ..errors import ToolError, WorkspaceViolationError
from ..llm.messages import (
    assistant_message_to_dict,
    system_message,
    tool_message,
    user_message,
)
from ..providers.llm_client import LLMClient
from ..sessions.store import Session
from ..tools.base import ToolContext, ToolRegistry, ToolResult

_TURN_LABEL_LIMIT = 50


def _turn_label(text: str) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= _TURN_LABEL_LIMIT:
        return flattened or "change"
    return flattened[: _TURN_LABEL_LIMIT - 3] + "..."


@runtime_checkable
class AgentUI(Protocol):
    """The terminal behaviors the agent loop relies on."""

    def begin_assistant(self) -> None: ...
    def stream_assistant(self, text: str) -> None: ...
    def end_assistant(self) -> None: ...
    def tool_started(self, name: str, raw_args: str) -> None: ...
    def tool_finished(self, name: str, result: ToolResult) -> None: ...
    def warning(self, text: str) -> None: ...
    def compacted(self, result: CompactionResult) -> None: ...


class Agent:
    """Drives one conversation: user message in, assistant actions out."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        context: ToolContext,
        session: Session,
        ui: AgentUI,
        system_prompt: str,
        max_iterations: int = 25,
        context_manager: ContextManager | None = None,
    ):
        self._llm = llm
        self._tools = tools
        self._context = context
        self._session = session
        self._ui = ui
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations
        self._context_manager = context_manager

    def run_turn(self, user_text: str) -> None:
        """Handle one user message, running tools until the model is done."""
        self._session.maybe_set_title(user_text)
        self._context.snapshots.start_turn(_turn_label(user_text))
        self._session.append(user_message(user_text))
        self._loop()

    def _loop(self) -> None:
        for _ in range(self._max_iterations):
            self._maybe_compact()
            messages = self._build_messages()
            self._ui.begin_assistant()
            result = self._llm.complete(
                messages,
                tools=self._tools.schemas(),
                on_text=self._ui.stream_assistant,
            )
            self._ui.end_assistant()

            stored = assistant_message_to_dict(result.message)
            self._session.append(stored)

            tool_calls = stored.get("tool_calls")
            if not tool_calls:
                return
            for call in tool_calls:
                self._handle_tool_call(call)
        self._ui.warning(
            "Reached the maximum number of tool steps for this turn. "
            "Send another message to continue."
        )

    def _maybe_compact(self) -> None:
        if self._context_manager is None:
            return
        result = self._context_manager.maybe_compact(self._session, self._llm)
        if result is not None:
            self._ui.compacted(result)

    def _build_messages(self) -> list[dict]:
        system_content = self._system_prompt
        if self._context_manager is not None:
            summary = self._context_manager.summary_text(self._session)
            tail = self._context_manager.tail_messages(self._session)
        else:
            summary = None
            tail = self._session.messages
        if summary:
            system_content = (
                f"{system_content}\n\nSummary of earlier conversation:\n{summary}"
            )
        return [system_message(system_content), *tail]

    def _handle_tool_call(self, call: dict) -> None:
        name = call["function"]["name"]
        raw_args = call["function"]["arguments"]
        self._ui.tool_started(name, raw_args)
        result = self._run_tool(name, raw_args)
        self._ui.tool_finished(name, result)
        self._session.append(tool_message(call["id"], result.content))

    def _run_tool(self, name: str, raw_args: str) -> ToolResult:
        try:
            tool = self._tools.get(name)
            args = tool.parse_args(raw_args)
            return tool.execute(args, self._context)
        except WorkspaceViolationError as exc:
            return ToolResult(content=f"Blocked for safety: {exc}", ok=False)
        except ToolError as exc:
            return ToolResult(content=f"Tool error: {exc}", ok=False)
