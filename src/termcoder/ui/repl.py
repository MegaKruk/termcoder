"""The interactive read-eval-print loop.

This module wires the pieces together: configuration, the model client, the
workspace guard, the tool registry with its command runner, the session store,
the snapshot store, the context manager, the approver and the agent. It also
handles slash commands. It is the one place that knows about all the parts,
which keeps every other module small and independent.
"""

from __future__ import annotations

import platform

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from ..agent.loop import Agent
from ..agent.system_prompt import build_system_prompt
from ..config import AppConfig
from ..context import ContextManager, TokenCounter
from ..errors import ConfigError, ProviderError, TermcoderError
from ..mcp import MCPClient, register_mcp_tools
from ..memory.loader import ProjectMemory, load_project_memory
from ..providers.llm_client import LLMClient
from ..providers.usage import UsageTracker
from ..repomap.builder import RepoMapBuilder, RepoMapResult
from ..sandbox.runner import build_command_runner
from ..sessions.store import SessionStore
from ..skills import SkillRegistry
from ..snapshots.store import NullSnapshotStore, SnapshotStore
from ..tools import build_default_registry
from ..tools.base import ToolContext
from ..workspace.paths import WorkspaceGuard
from .approver import ConsoleApprover
from .renderer import Renderer

_HELP = """Commands:
  /help                Show this help.
  /new                 Start a new chat session.
  /sessions            List chat sessions for this workspace.
  /resume <id>         Resume a previous session by id.
  /model [name]        Show the active model, or switch to another configured one.
  /compact [focus]     Summarize older turns now to free context space.
  /usage               Show token and cost usage for this session.
  /map [refresh]       Show the repository map, or rebuild it from the files.
  /memory [reload]     Show the project memory file, or reload it from disk.
  /skills              List the loaded skills.
  /undo                Revert the file changes from the most recent turn.
  /tools               List the available tools.
  /clear               Clear the screen.
  /exit, /quit         Leave termcoder.
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
        self._runner = build_command_runner(config.sandbox, config.workspace)
        self._skills = self._load_skills()
        self._tools = build_default_registry(
            config, command_runner=self._runner, skills=self._skills
        )
        self._mcp_client: MCPClient | None = None
        self._mcp_tool_count = self._connect_mcp_servers()
        self._context = ToolContext(
            workspace=self._workspace,
            approver=self._approver,
            emit=self._renderer.tool_progress,
            snapshots=NullSnapshotStore(),
        )

        self._usage = UsageTracker()
        self._llm = self._build_client()
        self._token_counter = TokenCounter()
        self._memory = self._load_memory()
        self._repo_map = self._build_repo_map()
        self._system_prompt = self._compose_system_prompt()
        self._context_manager = self._build_context_manager()
        self._session = self._store.create(model=config.active_model)
        self._snapshots = self._build_snapshots()
        self._context.snapshots = self._snapshots
        self._agent = self._build_agent()

    def run(self) -> None:
        """Start the main input loop."""
        self._renderer.banner(
            self._config.workspace, self._llm.model_name, sandbox=self._sandbox_label()
        )
        self._show_understanding_status()
        try:
            self._loop()
        finally:
            self._close()
        self._renderer.info("Goodbye.")

    def _loop(self) -> None:
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

    def _close(self) -> None:
        """Release resources held for the session, such as MCP servers."""
        if self._mcp_client is not None:
            self._mcp_client.close()
            self._mcp_client = None

    def _run_turn(self, line: str) -> None:
        self._usage.begin_turn()
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
        finally:
            if self._config.show_usage and self._usage.turn.calls:
                self._renderer.usage(self._usage.turn, self._usage.session)

    def _sandbox_label(self) -> str:
        if self._runner.is_sandboxed:
            return f"{self._runner.backend} container"
        return "host (no sandbox)"

    def _build_client(self) -> LLMClient:
        return LLMClient(
            self._config.model(),
            stream=self._config.stream,
            usage_tracker=self._usage,
        )

    def _build_summarizer(self) -> LLMClient:
        """Build the client that writes compaction summaries.

        Uses ``context.summary_model`` when configured, so a cheap model can
        handle the largest prompt termcoder sends; otherwise the active model.
        """
        name = self._config.context.summary_model
        if name:
            model_config = self._config.models.get(name)
            if model_config is None:
                raise ConfigError(
                    f"context.summary_model '{name}' is not a configured model. "
                    "Add it under [models] or remove the setting."
                )
        else:
            model_config = self._config.model()
        return LLMClient(model_config, stream=False, usage_tracker=self._usage)

    def _load_memory(self) -> ProjectMemory | None:
        settings = self._config.memory
        if not settings.enabled or not settings.files:
            return None
        return load_project_memory(self._config.workspace, settings.files)

    def _load_skills(self) -> SkillRegistry:
        settings = self._config.skills
        if not settings.enabled or not settings.directories:
            return SkillRegistry()
        directories = [
            self._config.workspace / name for name in settings.directories
        ]
        return SkillRegistry.from_directories(directories)

    def _connect_mcp_servers(self) -> int:
        """Connect configured MCP servers and register their tools.

        Returns the number of tools registered. A failure to reach one server
        is reported and skipped so the session still starts.
        """
        servers = [server for server in self._config.mcp_servers if server.enabled]
        if not servers:
            return 0
        self._mcp_client = MCPClient()
        return register_mcp_tools(
            self._tools, self._mcp_client, servers, self._renderer.status
        )

    def _build_repo_map(self) -> RepoMapResult | None:
        settings = self._config.repomap
        if not settings.enabled:
            return None
        builder = RepoMapBuilder(
            root=self._config.workspace,
            cache_path=self._config.cache_dir / "repomap.json",
            budget_tokens=settings.tokens,
            token_counter=self._token_counter,
        )
        return builder.build()

    def _compose_system_prompt(self) -> str:
        """Compose the full, session-stable system prompt."""
        map_text = self._repo_map.text if self._repo_map else None
        return build_system_prompt(
            self._workspace.root,
            self._tools.names(),
            platform.system(),
            memory=self._memory,
            repo_map=map_text,
            skill_catalog=self._skills.catalog(),
        )

    def _show_understanding_status(self) -> None:
        """Report what context the agent starts with, in dim status lines."""
        if self._memory is not None:
            note = " (truncated)" if self._memory.truncated else ""
            self._renderer.status(f"memory: {self._memory.path.name}{note}")
        if self._repo_map is not None:
            if self._repo_map.text:
                self._renderer.status(
                    f"repo map: {self._repo_map.tag_count} symbols from "
                    f"{self._repo_map.file_count} files, "
                    f"about {self._repo_map.tokens} tokens"
                )
            else:
                self._renderer.status(f"repo map: off ({self._repo_map.reason})")
        if len(self._skills) > 0:
            self._renderer.status(f"skills: {len(self._skills)} loaded")
        if self._config.web_search.enabled:
            self._renderer.status(
                f"web search: on ({self._config.web_search.provider})"
            )
        if self._mcp_tool_count > 0:
            self._renderer.status(f"mcp: {self._mcp_tool_count} tool(s)")

    def _build_context_manager(self) -> ContextManager:
        settings = self._config.context
        return ContextManager(
            auto_compact=settings.auto_compact,
            compact_threshold=settings.compact_threshold,
            keep_recent_turns=settings.keep_recent_turns,
            context_window=self._config.model().context_window,
            summarizer=self._build_summarizer(),
            token_counter=self._token_counter,
            system_prompt_tokens=self._token_counter.count_text(self._system_prompt),
        )

    def _build_snapshots(self):
        if not self._config.enable_undo:
            return NullSnapshotStore()
        root = self._config.snapshots_dir / self._session.meta.id
        return SnapshotStore(root, self._workspace)

    def _build_agent(self) -> Agent:
        return Agent(
            llm=self._llm,
            tools=self._tools,
            context=self._context,
            session=self._session,
            ui=self._renderer,
            system_prompt=self._system_prompt,
            max_iterations=self._config.max_tool_iterations,
            context_manager=self._context_manager,
        )

    def _rebuild_prompt_dependents(self) -> None:
        """Recompose the prompt and rebuild what depends on its size."""
        self._system_prompt = self._compose_system_prompt()
        self._context_manager = self._build_context_manager()
        self._agent = self._build_agent()

    def _rebind_session(self) -> None:
        """Refresh per-session state after the active session changes."""
        self._usage.reset()
        self._snapshots = self._build_snapshots()
        self._context.snapshots = self._snapshots
        self._agent = self._build_agent()

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
        elif command == "/compact":
            self._cmd_compact(argument)
        elif command == "/usage":
            self._renderer.usage_report(self._usage.session)
        elif command == "/map":
            self._cmd_map(argument)
        elif command == "/memory":
            self._cmd_memory(argument)
        elif command == "/skills":
            self._cmd_skills()
        elif command == "/undo":
            self._cmd_undo()
        elif command == "/tools":
            self._cmd_tools()
        elif command == "/clear":
            self._renderer.console.clear()
        else:
            self._renderer.warning(f"Unknown command: {command}. Try /help.")
        return False

    def _cmd_new(self) -> None:
        self._session = self._store.create(model=self._config.active_model)
        self._rebind_session()
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
        self._rebind_session()
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
        self._context_manager = self._build_context_manager()
        self._session.set_model(name)
        self._agent = self._build_agent()
        self._renderer.info(f"Switched to model '{name}'.")

    def _cmd_compact(self, instructions: str) -> None:
        try:
            result = self._context_manager.force_compact(
                self._session, instructions or None
            )
        except ProviderError as exc:
            self._renderer.error(str(exc))
            return
        if result is None:
            self._renderer.info("Nothing to compact yet.")
            return
        self._renderer.compacted(result)

    def _cmd_undo(self) -> None:
        if not self._config.enable_undo:
            self._renderer.warning(
                "Undo is turned off in configuration (set enable_undo = true)."
            )
            return
        self._renderer.undone(self._snapshots.undo_last())

    def _cmd_map(self, argument: str) -> None:
        if argument and argument.lower() != "refresh":
            self._renderer.warning("Usage: /map or /map refresh.")
            return
        if not self._config.repomap.enabled:
            self._renderer.info("The repo map is disabled (repomap.enabled = false).")
            return
        if argument:
            self._repo_map = self._build_repo_map()
            self._rebuild_prompt_dependents()
            self._renderer.info("Rebuilt the repository map from the current files.")
        if self._repo_map is None or self._repo_map.text is None:
            reason = self._repo_map.reason if self._repo_map else "not built"
            self._renderer.info(f"No repository map: {reason}.")
            return
        self._renderer.plain(self._repo_map.text)
        self._renderer.status(
            f"({self._repo_map.tag_count} symbols from {self._repo_map.file_count} "
            f"files, about {self._repo_map.tokens} of "
            f"{self._config.repomap.tokens} budgeted tokens)"
        )

    def _cmd_memory(self, argument: str) -> None:
        if argument and argument.lower() != "reload":
            self._renderer.warning("Usage: /memory or /memory reload.")
            return
        if not self._config.memory.enabled:
            self._renderer.info("Project memory is disabled (memory.enabled = false).")
            return
        if argument:
            self._memory = self._load_memory()
            self._rebuild_prompt_dependents()
            self._renderer.info("Reloaded project memory from disk.")
        if self._memory is None:
            names = ", ".join(self._config.memory.files)
            self._renderer.info(
                f"No project memory file found (looked for: {names}). Create "
                "one to give the agent durable project context."
            )
            return
        self._renderer.status(f"memory file: {self._memory.path}")
        self._renderer.plain(self._memory.text)

    def _cmd_tools(self) -> None:
        for tool in self._tools:
            kind = "read-only" if tool.is_read_only else "needs approval"
            self._renderer.plain(f"  {tool.name:16s} {kind}")

    def _cmd_skills(self) -> None:
        if len(self._skills) == 0:
            directories = ", ".join(self._config.skills.directories) or "(none)"
            self._renderer.info(
                "No skills loaded. Add SKILL.md folders under one of these "
                f"directories: {directories}."
            )
            return
        self._renderer.info(f"{len(self._skills)} skill(s) loaded:")
        for name in self._skills.names():
            skill = self._skills.get(name)
            self._renderer.plain(f"  {name}: {skill.description}")


def run_repl(config: AppConfig) -> None:
    """Build and run a REPL, reporting configuration errors cleanly."""
    try:
        Repl(config).run()
    except TermcoderError as exc:
        Renderer().error(str(exc))
