"""Application configuration.

Configuration is local-first and file-based. If a workspace contains a
``.termcoder/config.toml`` file it is merged over a small set of built-in
defaults. The defaults favor privacy: the active model is a local Ollama model
so the tool works without any cloud API key.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError

CONFIG_DIRNAME = ".termcoder"
CONFIG_FILENAME = "config.toml"
SESSIONS_DIRNAME = "sessions"
SNAPSHOTS_DIRNAME = "snapshots"
CACHE_DIRNAME = "cache"
HISTORY_FILENAME = "repl_history"


@dataclass(frozen=True)
class ModelConfig:
    """Connection settings for a single language model, in LiteLLM terms.

    The ``model`` field is the LiteLLM model string, for example
    ``"ollama_chat/llama3.1"``, ``"gpt-4o"`` or
    ``"claude-3-5-sonnet-20241022"``. ``api_key_env`` names an environment
    variable rather than holding a secret directly, so keys never live in
    config files or memory dumps.
    """

    name: str
    model: str
    api_base: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.2
    max_tokens: int | None = None
    context_window: int | None = None
    cache_prompts: bool = True

    def resolve_api_key(self) -> str | None:
        """Return the API key from the configured environment variable, if any."""
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    def to_completion_kwargs(self) -> dict:
        """Build the keyword arguments passed to the LiteLLM completion call."""
        kwargs: dict = {"model": self.model, "temperature": self.temperature}
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.api_base:
            kwargs["api_base"] = self.api_base
        key = self.resolve_api_key()
        if key:
            kwargs["api_key"] = key
        return kwargs


@dataclass(frozen=True)
class SandboxSettings:
    """Settings for running agent commands inside a sandbox container.

    ``backend`` is one of 'auto', 'podman', 'docker' or 'host'. With 'auto', a
    container engine is used when available and the host is the last resort.
    Network access is off by default so commands cannot reach out unless the
    user opts in (for example to install packages). ``read_only`` additionally
    mounts the container root filesystem read-only (with a writable /tmp),
    which hardens runs that only need to touch the workspace.
    """

    backend: str = "auto"
    image: str = "python:3.14-slim"
    network: bool = False
    memory: str = "1g"
    cpus: float = 2.0
    pids_limit: int = 256
    read_only: bool = False


@dataclass(frozen=True)
class ContextSettings:
    """Settings for token budgeting and conversation compaction.

    ``summary_model`` optionally names a configured model to write compaction
    summaries. The summary request is the largest single prompt the tool ever
    sends, so pointing it at a cheap model saves real money on cloud providers.
    When unset, the active model summarizes.
    """

    auto_compact: bool = True
    compact_threshold: float = 0.8
    keep_recent_turns: int = 3
    summary_model: str | None = None


@dataclass(frozen=True)
class RepoMapSettings:
    """Settings for the repository map injected into the system prompt.

    The map is built once per session (and on /map refresh) so the prompt
    prefix stays stable and cache-friendly. ``tokens`` is the budget the map
    is fitted into; Aider's long-standing default of about 1k tokens is a good
    balance of orientation value and cost.
    """

    enabled: bool = True
    tokens: int = 1024


@dataclass(frozen=True)
class MemorySettings:
    """Settings for project memory loaded from a markdown file.

    The first existing file from ``files`` is used. TERMCODER.md is the native
    name; AGENTS.md is recognized as a cross-tool fallback.
    """

    enabled: bool = True
    files: tuple[str, ...] = ("TERMCODER.md", "AGENTS.md")


@dataclass(frozen=True)
class WebSearchSettings:
    """Settings for the optional web search tool.

    Web search is off by default because it reaches the network. When enabled,
    the default provider is SearXNG, a self-hostable meta-search engine, so a
    local instance keeps queries private. ``api_base`` points at that instance;
    when empty, LiteLLM reads SEARXNG_API_BASE from the environment.
    """

    enabled: bool = False
    provider: str = "searxng"
    api_base: str = ""


@dataclass(frozen=True)
class SkillSettings:
    """Settings for Agent-Skills-style SKILL.md loading.

    Skills are discovered under each directory in ``directories``, resolved
    relative to the workspace. The default puts them in the workspace config
    directory so they travel with the project.
    """

    enabled: bool = True
    directories: tuple[str, ...] = (".termcoder/skills",)


@dataclass(frozen=True)
class MCPServerConfig:
    """Connection settings for a single MCP server (stdio transport).

    The server runs as a local subprocess: ``command`` plus ``args``, with an
    optional ``env`` and working directory. Keeping servers local is the
    privacy-preserving default; only configure servers from trusted sources,
    since their tools are model-controlled.
    """

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    startup_timeout: float = 30.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level runtime configuration for a single workspace."""

    workspace: Path
    config_dir: Path
    active_model: str
    models: dict[str, ModelConfig] = field(default_factory=dict)
    stream: bool = True
    max_tool_iterations: int = 25
    allow_run_command: bool = True
    enable_undo: bool = True
    show_usage: bool = True
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    context: ContextSettings = field(default_factory=ContextSettings)
    repomap: RepoMapSettings = field(default_factory=RepoMapSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    web_search: WebSearchSettings = field(default_factory=WebSearchSettings)
    skills: SkillSettings = field(default_factory=SkillSettings)
    mcp_servers: tuple[MCPServerConfig, ...] = ()

    def model(self) -> ModelConfig:
        """Return the currently selected model configuration."""
        try:
            return self.models[self.active_model]
        except KeyError as exc:
            available = ", ".join(sorted(self.models)) or "(none)"
            raise ConfigError(
                f"Active model '{self.active_model}' is not defined. "
                f"Available models: {available}."
            ) from exc

    def with_active_model(self, name: str) -> "AppConfig":
        """Return a copy of this config with a different active model selected."""
        if name not in self.models:
            available = ", ".join(sorted(self.models)) or "(none)"
            raise ConfigError(
                f"Unknown model '{name}'. Available models: {available}."
            )
        return AppConfig(
            workspace=self.workspace,
            config_dir=self.config_dir,
            active_model=name,
            models=self.models,
            stream=self.stream,
            max_tool_iterations=self.max_tool_iterations,
            allow_run_command=self.allow_run_command,
            enable_undo=self.enable_undo,
            show_usage=self.show_usage,
            sandbox=self.sandbox,
            context=self.context,
            repomap=self.repomap,
            memory=self.memory,
            web_search=self.web_search,
            skills=self.skills,
            mcp_servers=self.mcp_servers,
        )

    @property
    def sessions_dir(self) -> Path:
        """Directory holding per-chat session files for this workspace."""
        return self.config_dir / SESSIONS_DIRNAME

    @property
    def snapshots_dir(self) -> Path:
        """Directory holding file snapshots used for undo."""
        return self.config_dir / SNAPSHOTS_DIRNAME

    @property
    def cache_dir(self) -> Path:
        """Directory holding rebuildable caches such as the repo map tags."""
        return self.config_dir / CACHE_DIRNAME

    @property
    def history_path(self) -> Path:
        """File backing the REPL input history for this workspace."""
        return self.config_dir / HISTORY_FILENAME


def default_models() -> dict[str, ModelConfig]:
    """Return the built-in model registry.

    Ollama is listed first and used as the default so the tool runs fully
    locally without any cloud credentials. Cloud providers are opt-in and only
    activate when their API key environment variable is set.
    """
    return {
        "ollama": ModelConfig(
            name="ollama",
            model="ollama_chat/llama3.1",
            api_base="http://localhost:11434",
            context_window=8192,
        ),
        "openai": ModelConfig(
            name="openai",
            model="gpt-4o",
            api_key_env="OPENAI_API_KEY",
            context_window=128000,
        ),
        "anthropic": ModelConfig(
            name="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key_env="ANTHROPIC_API_KEY",
            context_window=200000,
        ),
    }


def _model_from_toml(name: str, raw: dict, fallback: ModelConfig | None) -> ModelConfig:
    """Build a ModelConfig from a TOML table, inheriting fields from a fallback."""
    base = fallback or ModelConfig(name=name, model=raw.get("model", ""))
    if not raw.get("model") and not base.model:
        raise ConfigError(f"Model '{name}' is missing the required 'model' field.")
    return ModelConfig(
        name=name,
        model=raw.get("model", base.model),
        api_base=raw.get("api_base", base.api_base),
        api_key_env=raw.get("api_key_env", base.api_key_env),
        temperature=raw.get("temperature", base.temperature),
        max_tokens=raw.get("max_tokens", base.max_tokens),
        context_window=raw.get("context_window", base.context_window),
        cache_prompts=bool(raw.get("cache_prompts", base.cache_prompts)),
    )


def _sandbox_from_toml(raw: dict) -> SandboxSettings:
    """Build SandboxSettings from a TOML table, falling back to defaults."""
    base = SandboxSettings()
    return SandboxSettings(
        backend=str(raw.get("backend", base.backend)),
        image=str(raw.get("image", base.image)),
        network=bool(raw.get("network", base.network)),
        memory=str(raw.get("memory", base.memory)),
        cpus=float(raw.get("cpus", base.cpus)),
        pids_limit=int(raw.get("pids_limit", base.pids_limit)),
        read_only=bool(raw.get("read_only", base.read_only)),
    )


def _repomap_from_toml(raw: dict) -> RepoMapSettings:
    """Build RepoMapSettings from a TOML table, falling back to defaults."""
    base = RepoMapSettings()
    return RepoMapSettings(
        enabled=bool(raw.get("enabled", base.enabled)),
        tokens=int(raw.get("tokens", base.tokens)),
    )


def _memory_from_toml(raw: dict) -> MemorySettings:
    """Build MemorySettings from a TOML table, falling back to defaults."""
    base = MemorySettings()
    files = raw.get("files", list(base.files))
    if isinstance(files, str):
        files = [files]
    return MemorySettings(
        enabled=bool(raw.get("enabled", base.enabled)),
        files=tuple(str(name) for name in files if str(name).strip()),
    )


def _context_from_toml(raw: dict) -> ContextSettings:
    """Build ContextSettings from a TOML table, falling back to defaults."""
    base = ContextSettings()
    summary_model = raw.get("summary_model", base.summary_model)
    return ContextSettings(
        auto_compact=bool(raw.get("auto_compact", base.auto_compact)),
        compact_threshold=float(raw.get("compact_threshold", base.compact_threshold)),
        keep_recent_turns=int(raw.get("keep_recent_turns", base.keep_recent_turns)),
        summary_model=str(summary_model) if summary_model else None,
    )


def _web_search_from_toml(raw: dict) -> WebSearchSettings:
    """Build WebSearchSettings from a TOML table, falling back to defaults."""
    base = WebSearchSettings()
    return WebSearchSettings(
        enabled=bool(raw.get("enabled", base.enabled)),
        provider=str(raw.get("provider", base.provider)),
        api_base=str(raw.get("api_base", base.api_base)),
    )


def _skills_from_toml(raw: dict) -> SkillSettings:
    """Build SkillSettings from a TOML table, falling back to defaults."""
    base = SkillSettings()
    directories = raw.get("directories", list(base.directories))
    if isinstance(directories, str):
        directories = [directories]
    return SkillSettings(
        enabled=bool(raw.get("enabled", base.enabled)),
        directories=tuple(
            str(name) for name in directories if str(name).strip()
        ),
    )


def _mcp_servers_from_toml(raw: object) -> tuple[MCPServerConfig, ...]:
    """Build MCP server configs from a list of TOML tables.

    Each table needs a ``name`` and a ``command``; malformed entries raise a
    ConfigError so problems surface at load time rather than during a session.
    """
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise ConfigError("mcp_servers must be a list of server tables.")
    servers: list[MCPServerConfig] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ConfigError("Each MCP server must be a table of settings.")
        name = str(entry.get("name", "")).strip()
        command = str(entry.get("command", "")).strip()
        if not name or not command:
            raise ConfigError("Each MCP server needs a name and a command.")
        args = entry.get("args", [])
        if isinstance(args, str):
            args = [args]
        env = entry.get("env", {})
        if not isinstance(env, dict):
            raise ConfigError(f"MCP server '{name}' env must be a table.")
        servers.append(
            MCPServerConfig(
                name=name,
                command=command,
                args=tuple(str(item) for item in args),
                env={str(key): str(value) for key, value in env.items()},
                cwd=str(entry["cwd"]) if entry.get("cwd") else None,
                enabled=bool(entry.get("enabled", True)),
                startup_timeout=float(entry.get("startup_timeout", 30.0)),
            )
        )
    return tuple(servers)


def load_config(workspace: Path, model_override: str | None = None) -> AppConfig:
    """Load configuration for a workspace, merging file settings over defaults.

    The workspace directory must already exist. The ``.termcoder`` config
    directory is created on demand by the callers that need it.
    """
    workspace = Path(workspace).resolve(strict=True)
    config_dir = workspace / CONFIG_DIRNAME

    data: dict = {}
    config_path = config_dir / CONFIG_FILENAME
    if config_path.is_file():
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)

    models = default_models()
    for name, raw in (data.get("models") or {}).items():
        if not isinstance(raw, dict):
            raise ConfigError(f"Model '{name}' must be a table of settings.")
        models[name] = _model_from_toml(name, raw, models.get(name))

    active = model_override or data.get("active_model") or "ollama"

    return AppConfig(
        workspace=workspace,
        config_dir=config_dir,
        active_model=active,
        models=models,
        stream=bool(data.get("stream", True)),
        max_tool_iterations=int(data.get("max_tool_iterations", 25)),
        allow_run_command=bool(data.get("allow_run_command", True)),
        enable_undo=bool(data.get("enable_undo", True)),
        show_usage=bool(data.get("show_usage", True)),
        sandbox=_sandbox_from_toml(data.get("sandbox") or {}),
        context=_context_from_toml(data.get("context") or {}),
        repomap=_repomap_from_toml(data.get("repomap") or {}),
        memory=_memory_from_toml(data.get("memory") or {}),
        web_search=_web_search_from_toml(data.get("web_search") or {}),
        skills=_skills_from_toml(data.get("skills") or {}),
        mcp_servers=_mcp_servers_from_toml(data.get("mcp_servers")),
    )
