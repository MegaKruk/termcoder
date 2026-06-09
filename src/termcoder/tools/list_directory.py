"""List files and folders inside the workspace."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from .base import ReadOnlyTool, ToolContext, ToolResult

# Folders that add noise to listings and are skipped during recursive walks.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".termcoder",
    }
)


class ListDirectoryArgs(BaseModel):
    """Arguments for the list_directory tool."""

    path: str = Field(
        default=".", description="Directory path relative to the workspace root."
    )
    recursive: bool = Field(
        default=False, description="List nested entries instead of only the top level."
    )
    max_entries: int = Field(
        default=200, ge=1, le=2000, description="Maximum number of entries to return."
    )


class ListDirectoryTool(ReadOnlyTool):
    """List the contents of a workspace directory."""

    name = "list_directory"
    description = (
        "List files and folders inside a workspace directory. Set recursive to "
        "true to include nested entries. Common noise folders such as .git and "
        "node_modules are skipped. Folders are marked with a trailing slash."
    )
    args_model = ListDirectoryArgs

    def _run(self, args: ListDirectoryArgs, context: ToolContext) -> ToolResult:
        base = context.workspace.resolve(args.path)
        if not base.exists():
            return ToolResult(content=f"Path not found: {args.path}", ok=False)
        if not base.is_dir():
            return ToolResult(content=f"Not a directory: {args.path}", ok=False)

        entries = (
            self._walk(base, context) if args.recursive else self._top_level(base)
        )
        total = len(entries)
        shown = entries[: args.max_entries]
        body = "\n".join(shown) if shown else "(empty directory)"
        if total > args.max_entries:
            body += f"\n(showing first {args.max_entries} of {total} entries)"
        return ToolResult(content=body, ok=True, display=f"{total} entries")

    def _top_level(self, base: Path) -> list[str]:
        children = sorted(
            base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
        )
        return [self._format(child) for child in children]

    def _walk(self, base: Path, context: ToolContext) -> list[str]:
        lines: list[str] = []
        for root, dirs, files in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
            root_path = Path(root)
            for name in sorted(files):
                rel = context.workspace.relative(root_path / name)
                lines.append(rel)
            for name in dirs:
                rel = context.workspace.relative(root_path / name)
                lines.append(rel + "/")
        return sorted(lines)

    @staticmethod
    def _format(entry: Path) -> str:
        return entry.name + "/" if entry.is_dir() else entry.name
