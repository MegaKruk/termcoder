"""Tests for the agent's resilience to malformed model output.

Small local models occasionally emit invalid tool-call JSON; the provider
layer surfaces that as MalformedModelOutputError and the agent retries the
request a bounded number of times. A flaky scripted model simulates the
failure deterministically.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from termcoder.agent.loop import Agent
from termcoder.agent.system_prompt import build_system_prompt
from termcoder.approval.auto import AutoApprover
from termcoder.config import AppConfig, default_models
from termcoder.errors import MalformedModelOutputError
from termcoder.providers.llm_client import CompletionResult
from termcoder.sessions.store import SessionStore
from termcoder.tools import build_default_registry
from termcoder.tools.base import ToolContext
from termcoder.workspace.paths import WorkspaceGuard


class RecordingUI:
    """AgentUI implementation that records warnings and stream lifecycle."""

    def __init__(self):
        self.warnings = []
        self.begin_count = 0
        self.end_count = 0

    def begin_assistant(self):
        self.begin_count += 1

    def stream_assistant(self, text):
        pass

    def end_assistant(self):
        self.end_count += 1

    def tool_started(self, name, raw_args):
        pass

    def tool_finished(self, name, result):
        pass

    def warning(self, text):
        self.warnings.append(text)

    def compacted(self, result):
        pass


class FlakyLLM:
    """Raises MalformedModelOutputError a set number of times, then answers."""

    model_name = "flaky"

    def __init__(self, failures: int, answer: str = "Recovered fine."):
        self._failures = failures
        self._answer = answer
        self.calls = 0

    def complete(self, messages, tools=None, on_text=None):
        self.calls += 1
        if self.calls <= self._failures:
            raise MalformedModelOutputError("bad tool-call JSON")
        message = SimpleNamespace(content=self._answer, tool_calls=None)
        return CompletionResult(message=message, raw=None)


def _agent(tmp_path, llm, ui):
    config = AppConfig(
        workspace=tmp_path,
        config_dir=tmp_path / ".termcoder",
        active_model="ollama",
        models=default_models(),
    )
    registry = build_default_registry(config)
    context = ToolContext(workspace=WorkspaceGuard(tmp_path), approver=AutoApprover())
    session = SessionStore(config.sessions_dir).create(model="flaky")
    prompt = build_system_prompt(tmp_path, registry.names(), "TestOS")
    agent = Agent(
        llm=llm,
        tools=registry,
        context=context,
        session=session,
        ui=ui,
        system_prompt=prompt,
    )
    return agent, session


def test_turn_recovers_after_two_malformed_responses(tmp_path):
    llm = FlakyLLM(failures=2)
    ui = RecordingUI()
    agent, session = _agent(tmp_path, llm, ui)

    agent.run_turn("hello")

    assert llm.calls == 3
    assert len(ui.warnings) == 2
    assert "retrying (1 of 2)" in ui.warnings[0]
    assert "retrying (2 of 2)" in ui.warnings[1]
    assert ui.begin_count == 3
    assert ui.end_count == 3
    assert session.messages[-1]["content"] == "Recovered fine."


def test_turn_fails_after_retries_exhausted(tmp_path):
    llm = FlakyLLM(failures=3)
    ui = RecordingUI()
    agent, session = _agent(tmp_path, llm, ui)

    with pytest.raises(MalformedModelOutputError):
        agent.run_turn("hello")

    assert llm.calls == 3
    assert len(ui.warnings) == 2
    # Only the user message was stored; no half-finished assistant message.
    assert [message["role"] for message in session.messages] == ["user"]
