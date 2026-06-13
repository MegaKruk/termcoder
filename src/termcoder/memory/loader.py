"""Load project memory from a markdown file in the workspace.

Project memory holds durable facts about the project: conventions, build and
test commands, architecture notes, preferences. It is a plain markdown file
the user owns and can read, edit and commit. termcoder reads the first file
found from a configurable list (TERMCODER.md by default, with AGENTS.md as a
cross-tool fallback) and folds it into the system prompt as a stable block, so
it is cache-friendly and survives compaction.

The agent has no special write path to memory: when asked to remember
something, it proposes an edit to the memory file through the normal approved
file tools.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Memory is meant to be small and curated; anything beyond this is truncated
# to protect the context budget.
_MAX_MEMORY_CHARS = 16_000

_TRUNCATION_NOTICE = "\n\n[memory truncated; keep this file short]"


@dataclass(frozen=True)
class ProjectMemory:
    """The loaded memory file and its content."""

    path: Path
    text: str
    truncated: bool = False


def load_project_memory(
    root: Path, filenames: Sequence[str]
) -> ProjectMemory | None:
    """Return the first existing, non-empty memory file from ``filenames``.

    Files are looked up relative to the workspace root. Returns None when no
    candidate exists or the first existing one is empty or unreadable.
    """
    for name in filenames:
        path = Path(root) / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
        if not text:
            return None
        truncated = len(text) > _MAX_MEMORY_CHARS
        if truncated:
            text = text[:_MAX_MEMORY_CHARS] + _TRUNCATION_NOTICE
        return ProjectMemory(path=path, text=text, truncated=truncated)
    return None
