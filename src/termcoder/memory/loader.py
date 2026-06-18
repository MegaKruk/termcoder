"""Load and structure project memory from a markdown file.

Project memory holds durable facts about a project: conventions, build and test
commands, architecture notes, glossary, decisions, preferences. It is a plain
markdown file the user owns and can read, edit and commit (TERMCODER.md by
default, with AGENTS.md as a cross-tool fallback). It is folded into the system
prompt as a stable block, so it is cache-friendly and survives compaction.

Why structured, not a graph database. A mind map for a language model is not a
picture; it is text with clear structure the model can navigate. Flat memory
grows into one undifferentiated blob that must be loaded whole. Splitting it by
markdown heading turns each topic into an addressable section, which keeps the
plaintext, git-friendly, human-editable properties while letting the agent and
the user refer to and update one section at a time. The full text is still kept
for prompt injection and full backward compatibility with a flat file that has
no headings.

The agent has no special write path to memory: when asked to remember
something, it proposes an edit to the memory file through the normal approved
file tools, ideally under the right section heading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Memory is meant to be small and curated; anything beyond this is truncated to
# protect the context budget.
_MAX_MEMORY_CHARS = 16_000

_TRUNCATION_NOTICE = "\n\n[memory truncated; keep this file short]"

# Headings up to this level start a new section; deeper headings stay within
# their parent section body, so a section can have sub-structure.
_SECTION_HEADING_MAX_LEVEL = 2


@dataclass(frozen=True)
class MemorySection:
    """One titled section of project memory."""

    title: str
    body: str
    level: int  # markdown heading level, 1 for '#', 2 for '##'

    def is_empty(self) -> bool:
        return not self.body.strip()


@dataclass(frozen=True)
class ProjectMemory:
    """The loaded memory file, its full text, and its parsed sections."""

    path: Path
    text: str
    truncated: bool = False
    sections: tuple[MemorySection, ...] = field(default_factory=tuple)

    def section_titles(self) -> list[str]:
        """Return the titles of the parsed sections, in document order."""
        return [section.title for section in self.sections]

    def find_section(self, query: str) -> MemorySection | None:
        """Return a section whose title matches ``query`` case-insensitively.

        An exact match wins; otherwise the first section whose title contains
        the query is returned, so "convention" finds "Conventions".
        """
        lowered = query.strip().lower()
        for section in self.sections:
            if section.title.lower() == lowered:
                return section
        for section in self.sections:
            if lowered in section.title.lower():
                return section
        return None


def parse_sections(text: str) -> tuple[MemorySection, ...]:
    """Split markdown into sections by top-level (``#``/``##``) headings.

    Content before the first qualifying heading becomes a "Preamble" section so
    no text is lost. A file with no headings yields a single section, which is
    why a flat memory file keeps working unchanged.
    """
    lines = text.splitlines()
    sections: list[MemorySection] = []
    current_title = "Preamble"
    current_level = 0
    current_body: list[str] = []

    def flush() -> None:
        body = "\n".join(current_body).strip()
        if body or current_title != "Preamble":
            sections.append(
                MemorySection(title=current_title, body=body, level=current_level)
            )

    for line in lines:
        heading = _heading(line)
        if heading is not None and heading[0] <= _SECTION_HEADING_MAX_LEVEL:
            flush()
            current_level, current_title = heading
            current_body = []
        else:
            current_body.append(line)
    flush()
    return tuple(sections)


def _heading(line: str) -> tuple[int, str] | None:
    """Return (level, title) if the line is an ATX heading, else None."""
    stripped = line.lstrip()
    if not stripped.startswith("#"):
        return None
    hashes = len(stripped) - len(stripped.lstrip("#"))
    rest = stripped[hashes:]
    if not rest.startswith(" "):
        return None
    return hashes, rest.strip()


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
        sections = parse_sections(text)
        truncated = len(text) > _MAX_MEMORY_CHARS
        if truncated:
            text = text[:_MAX_MEMORY_CHARS] + _TRUNCATION_NOTICE
        return ProjectMemory(
            path=path, text=text, truncated=truncated, sections=sections
        )
    return None
