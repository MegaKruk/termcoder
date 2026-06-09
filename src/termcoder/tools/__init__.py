"""Tool framework and the built-in tool set.

``build_default_registry`` is the single place that decides which tools the
agent starts with. Future phases register additional tools (sandboxed runners,
web search, MCP-backed tools) here or alongside this function.
"""

from __future__ import annotations

from ..config import AppConfig
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


def build_default_registry(config: AppConfig) -> ToolRegistry:
    """Create a registry with the standard Phase 1 tool set."""
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(ListDirectoryTool())
    registry.register(SearchTextTool())
    registry.register(WriteFileTool())
    registry.register(EditFileTool())
    if config.allow_run_command:
        registry.register(RunCommandTool())
    return registry
