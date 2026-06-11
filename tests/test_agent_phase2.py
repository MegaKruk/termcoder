"""Offline integration test for the Phase 2 agent wiring.

A scripted model writes a file through the tool while a real snapshot store is
attached. The test then undoes the turn and confirms the newly created file is
removed. It also confirms the renderer satisfies the agent UI protocol now that
a compaction notice method exists.
"""

from __future__ import annotations

from types import SimpleNamespace

from termcoder.agent.loop import Agent, AgentUI
from termcoder.agent.system_prompt import build_system_prompt
from termcoder.approval.auto import AutoApprover
from termcoder.config import AppConfig, default_models
from termcoder.context import ContextManager
from termcoder.providers.llm_client import CompletionResult
from termcoder.sessions.store import SessionStore
from termcoder.snapshots.store import SnapshotStore
from termcoder.tools import build_default_registry
from termcoder.tools.base import ToolContext
from termcoder.ui.renderer import Renderer
from termcoder.workspace.paths import WorkspaceGuard


class RecordingUI:
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

    def compacted(self, result):
        pass


class ScriptedLLM:
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
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def _config(tmp_path):
    return AppConfig(
        workspace=tmp_path,
        config_dir=tmp_path / ".termcoder",
        active_model="ollama",
        models=default_models(),
    )


def test_renderer_satisfies_agent_ui():
    assert isinstance(Renderer(), AgentUI)


def test_turn_snapshots_then_undo_removes_created_file(tmp_path):
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
    final_answer = SimpleNamespace(content="Created it.", tool_calls=None)

    config = _config(tmp_path)
    registry = build_default_registry(config)
    snapshots = SnapshotStore(config.snapshots_dir / "session", WorkspaceGuard(tmp_path))
    context = ToolContext(
        workspace=WorkspaceGuard(tmp_path),
        approver=AutoApprover(),
        snapshots=snapshots,
    )
    session = SessionStore(config.sessions_dir).create(model="scripted")
    prompt = build_system_prompt(tmp_path, registry.names(), "TestOS")
    manager = ContextManager(
        auto_compact=False,
        compact_threshold=0.8,
        keep_recent_turns=3,
        context_window=100000,
    )

    agent = Agent(
        llm=ScriptedLLM([write_request, final_answer]),
        tools=registry,
        context=context,
        session=session,
        ui=RecordingUI(),
        system_prompt=prompt,
        context_manager=manager,
    )

    agent.run_turn("Please create greeting.txt")

    target = tmp_path / "greeting.txt"
    assert target.read_text(encoding="utf-8") == "hi\n"
    assert snapshots.has_undo()

    result = snapshots.undo_last()
    assert result is not None
    assert not target.exists()
    assert "greeting.txt" in result.deleted
