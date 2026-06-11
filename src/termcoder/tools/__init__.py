"""Tool framework and the built-in tool set.

``build_default_registry`` is the single place that decides which tools the
agent starts with. Future phases register additional tools (sandboxed runners,
web search, MCP-backed tools) here or alongside this function.
"""

from __future__ import annotations

from ..config import AppConfig
from ..sandbox.runner import CommandRunner, build_command_runner
from .base import (
    MutatingTool,
    ReadOnlyTool,
    Tool,
    ToolContext,
    ToolPreview,
    ToolRegistry,
    ToolResult,
)
from .edit_file import EditFileTool
from .list_directory import ListDirectoryTool
from .read_file import ReadFileTool
from .run_command import RunCommandTool
from .search_text import SearchTextTool
from .write_file import WriteFileTool

__all__ = [
    "Tool",
    "ReadOnlyTool",
    "MutatingTool",
    "ToolContext",
    "ToolResult",
    "ToolPreview",
    "ToolRegistry",
    "ReadFileTool",
    "ListDirectoryTool",
    "SearchTextTool",
    "WriteFileTool",
    "EditFileTool",
    "RunCommandTool",
    "build_default_registry",
]


def build_default_registry(
    config: AppConfig, command_runner: CommandRunner | None = None
) -> ToolRegistry:
    """Create a registry with the standard tool set.

    The run_command tool is wired to a command runner. When one is not provided,
    it is built from the sandbox configuration, so the same call works in tests
    and at runtime.
    """
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ListDirectoryTool())
    registry.register(SearchTextTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    if config.allow_run_command:
        runner = command_runner or build_command_runner(config.sandbox, config.workspace)
        registry.register(RunCommandTool(runner=runner))
    return registry
