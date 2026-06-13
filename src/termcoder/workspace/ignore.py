"""Directory names skipped during workspace walks.

Shared by the listing and search tools and the repository map so they all
agree on what counts as noise. Kept in the workspace package because it is a
property of how termcoder views the workspace, not of any single tool.
"""

from __future__ import annotations

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
