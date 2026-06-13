"""Search file contents for a regular expression.

Uses ripgrep when it is installed because it is fast and honors .gitignore.
Falls back to a built-in Python scanner so the tool always works, even without
ripgrep on the host.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from .base import ReadOnlyTool, ToolContext, ToolResult
from .list_directory import IGNORED_DIRS

RIPGREP_TIMEOUT_SECONDS = 30
HARD_MATCH_CAP = 5000


class SearchTextArgs(BaseModel):
    """Arguments for the search_text tool."""

    pattern: str = Field(description="Regular expression to search for in file contents.")
    path: str = Field(
        default=".", description="File or directory to search, relative to the workspace."
    )
    case_insensitive: bool = Field(
        default=False, description="Match without regard to letter case."
    )
    max_results: int = Field(
        default=200, ge=1, le=2000, description="Maximum number of matches to return."
    )
    include: str | None = Field(
        default=None,
        description=(
            "Optional glob limiting which files are searched, matched against "
            "file names and workspace-relative paths, for example '*.py'."
        ),
    )


class SearchTextTool(ReadOnlyTool):
    """Search file contents and return matching lines with their locations."""

    name = "search_text"
    description = (
        "Search file contents for a regular expression and return matching lines "
        "as 'path:line: text'. Uses ripgrep when available, otherwise a built-in "
        "scanner. Use include to limit the search to matching files, for example "
        "'*.py'. Prefer this over reading whole files when looking for something."
    )
    args_model = SearchTextArgs

    def _run(self, args: SearchTextArgs, context: ToolContext) -> ToolResult:
        base = context.workspace.resolve(args.path)
        if not base.exists():
            return ToolResult(content=f"Path not found: {args.path}", ok=False)

        if shutil.which("rg"):
            matches = self._ripgrep(args, base, context)
        else:
            matches = self._python_search(args, base, context)

        if matches is None:
            return ToolResult(
                content="The search pattern is not a valid regular expression.",
                ok=False,
            )
        if not matches:
            return ToolResult(content="No matches found.", ok=True, display="0 matches")

        shown = matches[: args.max_results]
        body = "\n".join(shown)
        if len(matches) > args.max_results:
            body += f"\n(showing first {args.max_results} of {len(matches)} matches)"
        return ToolResult(content=body, ok=True, display=f"{len(matches)} matches")

    def _ripgrep(
        self, args: SearchTextArgs, base: Path, context: ToolContext
    ) -> list[str] | None:
        command = ["rg", "--line-number", "--no-heading", "--color", "never"]
        if args.case_insensitive:
            command.append("--ignore-case")
        if args.include:
            command += ["--glob", args.include]
        command += ["--regexp", args.pattern, str(base)]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=RIPGREP_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return []
        # ripgrep exit codes: 0 found, 1 no matches, 2 error (such as bad regex).
        if completed.returncode == 2:
            return None
        results: list[str] = []
        for line in completed.stdout.splitlines():
            results.append(self._relativize(line, context))
        return results

    def _python_search(
        self, args: SearchTextArgs, base: Path, context: ToolContext
    ) -> list[str] | None:
        flags = re.IGNORECASE if args.case_insensitive else 0
        try:
            regex = re.compile(args.pattern, flags)
        except re.error:
            return None

        files = [base] if base.is_file() else self._iter_files(base)
        results: list[str] = []
        for file_path in files:
            rel = context.workspace.relative(file_path)
            if args.include and not self._include_matches(
                args.include, rel, file_path.name
            ):
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    results.append(f"{rel}:{number}: {line.strip()}")
                    if len(results) >= HARD_MATCH_CAP:
                        return results
        return results

    @staticmethod
    def _include_matches(include: str, rel: str, name: str) -> bool:
        from fnmatch import fnmatch

        return fnmatch(rel, include) or fnmatch(name, include)

    @staticmethod
    def _iter_files(base: Path):
        import os

        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for name in files:
                yield Path(root) / name

    @staticmethod
    def _relativize(line: str, context: ToolContext) -> str:
        # ripgrep output is "path:line:content"; relativize the path portion.
        parts = line.split(":", 2)
        if len(parts) < 3:
            return line
        path_text, number, content = parts
        rel = context.workspace.relative(Path(path_text))
        return f"{rel}:{number}: {content.strip()}"
