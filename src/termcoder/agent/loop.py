"""The agent loop.

This is the deterministic core: assemble the message list, call the model,
render any streamed text, then either finish (no tool calls) or run each
requested tool and feed the results back. Every tool call produces a tool
result message, including on error, so the conversation stays valid for the
next model turn.

The loop talks to the terminal through a small :class:`AgentUI` protocol rather
than importing the UI package directly. This keeps the dependency one-way (the
UI imports the agent, not the reverse) and makes the loop easy to test with a
fake UI.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

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


@runtime_checkable
class AgentUI(Protocol):
    """The terminal behaviors the agent loop relies on."""

    def begin_assistant(self) -> None: ...
    def stream_assistant(self, text: str) -> None: ...
    def end_assistant(self) -> None: ...
    def tool_started(self, name: str, raw_args: str) -> None: ...
    def tool_finished(self, name: str, result: ToolResult) -> None: ...
    def warning(self, text: str) -> None: ...


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
    ):
        self._llm = llm
        self._tools = tools
        self._context = context
        self._session = session
        self._ui = ui
        self._system_prompt = system_prompt
        self._max_iterations = max_iterations

    def run_turn(self, user_text: str) -> None:
        """Handle one user message, running tools until the model is done."""
        self._session.maybe_set_title(user_text)
        self._session.append(user_message(user_text))
        self._loop()

    def _loop(self) -> None:
        for _ in range(self._max_iterations):
            messages = [system_message(self._system_prompt), *self._session.messages]
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
