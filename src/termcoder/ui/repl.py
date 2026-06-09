"""The interactive read-eval-print loop.

This module wires the pieces together: configuration, the model client, the
workspace guard, the tool registry, the session store, the approver and the
agent. It also handles slash commands. It is the one place that knows about all
the parts, which keeps every other module small and independent.
"""

from __future__ import annotations

import platform

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from ..agent.loop import Agent
from ..agent.system_prompt import build_system_prompt
from ..config import AppConfig
from ..errors import ConfigError, ProviderError, TermcoderError
from ..providers.llm_client import LLMClient
from ..sessions.store import SessionStore
from ..tools import build_default_registry
from ..tools.base import ToolContext
from ..workspace.paths import WorkspaceGuard
from .approver import ConsoleApprover
from .renderer import Renderer

_HELP = """Commands:
  /help            Show this help.
  /new             Start a new chat session.
  /sessions        List chat sessions for this workspace.
  /resume <id>     Resume a previous session by id.
  /model [name]    Show the active model, or switch to another configured one.
  /tools           List the available tools.
  /clear           Clear the screen.
  /exit, /quit     Leave termcoder.
"""


class Repl:
    """Run an interactive coding session in the terminal."""

    def __init__(self, config: AppConfig):
        self._config = config
        config.config_dir.mkdir(parents=True, exist_ok=True)

        self._renderer = Renderer()
        self._prompt = PromptSession(history=FileHistory(str(config.history_path)))
        self._approver = ConsoleApprover(self._renderer)
        self._store = SessionStore(config.sessions_dir)
        self._workspace = WorkspaceGuard(config.workspace)
        self._tools = build_default_registry(config)
        self._context = ToolContext(
            workspace=self._workspace,
            approver=self._approver,
            emit=self._renderer.tool_progress,
        )

        self._llm = self._build_client()
        self._session = self._store.create(model=config.active_model)
        self._agent = self._build_agent()

    def run(self) -> None:
        """Start the main input loop."""
        self._renderer.banner(self._config.workspace, self._llm.model_name)
        while True:
            try:
                line = self._prompt.prompt("you> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                continue
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                if self._handle_command(line):
                    break
                continue
            self._run_turn(line)
        self._renderer.info("Goodbye.")

    def _run_turn(self, line: str) -> None:
        try:
            self._agent.run_turn(line)
        except ProviderError as exc:
            self._renderer.error(str(exc))
            self._renderer.warning(
                "Check that the selected model is reachable. For the default "
                "Ollama model, make sure Ollama is running and the model is "
                "pulled. For cloud models, check the API key environment "
                "variable."
            )
        except KeyboardInterrupt:
            self._renderer.warning("Interrupted this turn.")

    def _build_client(self) -> LLMClient:
        return LLMClient(self._config.model(), stream=self._config.stream)

    def _build_agent(self) -> Agent:
        prompt = build_system_prompt(
            self._workspace.root, self._tools.names(), platform.system()
        )
        return Agent(
            llm=self._llm,
            tools=self._tools,
            context=self._context,
            session=self._session,
            ui=self._renderer,
            system_prompt=prompt,
            max_iterations=self._config.max_tool_iterations,
        )

    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command in {"/exit", "/quit"}:
            return True
        if command == "/help":
            self._renderer.plain(_HELP)
        elif command == "/new":
            self._cmd_new()
        elif command == "/sessions":
            self._cmd_sessions()
        elif command == "/resume":
            self._cmd_resume(argument)
        elif command == "/model":
            self._cmd_model(argument)
        elif command == "/tools":
            self._cmd_tools()
        elif command == "/clear":
            self._renderer.console.clear()
        else:
            self._renderer.warning(f"Unknown command: {command}. Try /help.")
        return False

    def _cmd_new(self) -> None:
        self._session = self._store.create(model=self._config.active_model)
        self._agent = self._build_agent()
        self._renderer.info(f"Started a new session ({self._session.meta.id}).")

    def _cmd_sessions(self) -> None:
        sessions = self._store.list()
        if not sessions:
            self._renderer.info("No sessions yet.")
            return
        for meta in sessions:
            marker = "*" if meta.id == self._session.meta.id else " "
            self._renderer.plain(
                f"{marker} {meta.id}  {meta.model:10s}  {meta.title}"
            )

    def _cmd_resume(self, session_id: str) -> None:
        if not session_id:
            self._renderer.warning("Usage: /resume <id>. See /sessions for ids.")
            return
        try:
            self._session = self._store.open(session_id)
        except FileNotFoundError as exc:
            self._renderer.error(str(exc))
            return
        self._agent = self._build_agent()
        self._renderer.info(
            f"Resumed session {self._session.meta.id} "
            f"({len(self._session.messages)} messages)."
        )

    def _cmd_model(self, name: str) -> None:
        if not name:
            self._renderer.info(f"Active model: {self._config.active_model}")
            available = ", ".join(sorted(self._config.models))
            self._renderer.plain(f"Available: {available}")
            return
        try:
            self._config = self._config.with_active_model(name)
        except ConfigError as exc:
            self._renderer.error(str(exc))
            return
        self._llm = self._build_client()
        self._session.set_model(name)
        self._agent = self._build_agent()
        self._renderer.info(f"Switched to model '{name}'.")

    def _cmd_tools(self) -> None:
        for tool in self._tools:
            kind = "read-only" if tool.is_read_only else "needs approval"
            self._renderer.plain(f"  {tool.name:16s} {kind}")


def run_repl(config: AppConfig) -> None:
    """Build and run a REPL, reporting configuration errors cleanly."""
    try:
        Repl(config).run()
    except TermcoderError as exc:
        Renderer().error(str(exc))
