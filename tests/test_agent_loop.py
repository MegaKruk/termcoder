"""Offline integration test for the agent loop.

A scripted fake model first asks to write a file (approved automatically), then
returns a final answer. This exercises tool dispatch, approval, session storage
and loop termination without any network or real model.
"""

from __future__ import annotations

from types import SimpleNamespace

from termcoder.agent.loop import Agent
from termcoder.agent.system_prompt import build_system_prompt
from termcoder.approval.auto import AutoApprover
from termcoder.providers.llm_client import CompletionResult
from termcoder.sessions.store import SessionStore
from termcoder.tools import build_default_registry
from termcoder.tools.base import ToolContext
from termcoder.workspace.paths import WorkspaceGuard
from termcoder.config import AppConfig, default_models


class RecordingUI:
    """Minimal AgentUI implementation that records nothing but stays silent."""

    def begin_assistant(self):
        pass

    def stream_assistant(self, text):
        pass

    def end_assistant(self):
        pass

    def tool_started(self, name, raw_args):
        pass

    def tool_finished(self, name, result):
        pass

    def warning(self, text):
        pass


class ScriptedLLM:
    """Returns pre-baked completions in order, ignoring the input messages."""

    model_name = "scripted"

    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages, tools=None, on_text=None):
        message = self._replies.pop(0)
        if on_text and message.content:
            on_text(message.content)
        return CompletionResult(message=message, raw=None)


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _config(tmp_path):
    return AppConfig(
        workspace=tmp_path,
        config_dir=tmp_path / ".termcoder",
        active_model="ollama",
        models=default_models(),
    )


def test_agent_runs_tool_then_finishes(tmp_path):
    write_request = SimpleNamespace(
        content="",
        tool_calls=[
            _tool_call(
                "call_1",
                "write_file",
                '{"path": "greeting.txt", "content": "hi\\n"}',
            )
        ],
    )
    final_answer = SimpleNamespace(content="Done, I created the file.", tool_calls=None)

    config = _config(tmp_path)
    registry = build_default_registry(config)
    context = ToolContext(workspace=WorkspaceGuard(tmp_path), approver=AutoApprover())
    store = SessionStore(config.sessions_dir)
    session = store.create(model="scripted")
    prompt = build_system_prompt(tmp_path, registry.names(), "TestOS")

    agent = Agent(
        llm=ScriptedLLM([write_request, final_answer]),
        tools=registry,
        context=context,
        session=session,
        ui=RecordingUI(),
        system_prompt=prompt,
    )

    agent.run_turn("Please create greeting.txt")

    # The file was written through the tool.
    assert (tmp_path / "greeting.txt").read_text(encoding="utf-8") == "hi\n"

    # The transcript holds: user, assistant(tool call), tool result, assistant(final).
    roles = [message["role"] for message in session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.messages[-1]["content"] == "Done, I created the file."
