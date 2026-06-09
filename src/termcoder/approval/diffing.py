"""Unified diff generation.

Diffs are shown to the user before any file write so they can accept or reject
the exact change, mirroring the approval flow of GitHub Copilot.
"""

from __future__ import annotations

import difflib


def make_unified_diff(old: str, new: str, path: str) -> str:
    """Return a unified diff between two strings, or an empty string if equal.

    The ``path`` is used only to label the diff headers. The result is safe to
    render with a diff syntax highlighter.
    """
    if old == new:
        return ""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    text = "".join(diff)
    if text and not text.endswith("\n"):
        text += "\n"
    return text
