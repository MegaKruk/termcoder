"""Find files by name with a glob pattern.

Completes the agentic search quartet (search, find, read, list): search_text
finds content, find_files finds files. Matching uses fnmatch semantics against
workspace-relative paths, where '*' also crosses directory separators, so
'*.py' finds Python files at any depth and 'tests/test_*.py' narrows by
location. Noise directories are skipped via the shared ignore set.
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path

from pydantic import BaseModel, Field

from ..workspace.ignore import IGNORED_DIRS
from .base import ReadOnlyTool, ToolContext, ToolResult


class FindFilesArgs(BaseModel):
    """Arguments for the find_files tool."""

    pattern: str = Field(
        description=(
            "Glob pattern matched against workspace-relative paths and file "
            "names, for example '*.py' or 'tests/test_*.py'. '*' matches any "
            "characters, including '/'."
        )
    )
    path: str = Field(
        default=".", description="Directory to search, relative to the workspace."
    )
    max_results: int = Field(
        default=200, ge=1, le=2000, description="Maximum number of paths to return."
    )


class FindFilesTool(ReadOnlyTool):
    """Find files whose names or paths match a glob pattern."""

    name = "find_files"
    description = (
        "Find files by name with a glob pattern, returning workspace-relative "
        "paths sorted alphabetically. Use this to locate files when you know "
        "part of the name; use search_text to look inside files instead."
    )
    args_model = FindFilesArgs

    def _run(self, args: FindFilesArgs, context: ToolContext) -> ToolResult:
        base = context.workspace.resolve(args.path)
        if not base.exists():
            return ToolResult(content=f"Path not found: {args.path}", ok=False)
        if not base.is_dir():
            return ToolResult(content=f"Not a directory: {args.path}", ok=False)

        matches: list[str] = []
        for current, dirs, names in os.walk(base):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRS)
            for name in names:
                rel = context.workspace.relative(Path(current) / name)
                # Use forward slashes everywhere so glob patterns that contain
                # '/' (for example 'tests/test_*.py') match on Windows too, and
                # the returned paths are consistent across platforms.
                rel = rel.replace("\\", "/")
                if fnmatch(rel, args.pattern) or fnmatch(name, args.pattern):
                    matches.append(rel)
        matches.sort()

        if not matches:
            return ToolResult(content="No files matched.", ok=True, display="0 files")
        shown = matches[: args.max_results]
        body = "\n".join(shown)
        if len(matches) > args.max_results:
            body += f"\n(showing first {args.max_results} of {len(matches)} files)"
        return ToolResult(content=body, ok=True, display=f"{len(matches)} files")
