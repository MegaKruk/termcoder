"""Load Agent-Skills-style SKILL.md folders with progressive disclosure.

A skill is a folder containing a ``SKILL.md`` file: YAML-style frontmatter with
at least ``name`` and ``description``, followed by a markdown body. The folder
may also hold ``scripts/``, ``references/`` and ``assets/`` used while the skill
runs. This mirrors Anthropic's Agent Skills open standard.

Progressive disclosure keeps the context cheap. Three tiers:

1. Startup: only each skill's name and one-line description (tens of tokens)
   are listed in the system prompt, so the model knows what exists.
2. On demand: when a skill is relevant the model calls a tool to read its full
   body, which loads the instructions.
3. During execution: bundled scripts and references are read or run through the
   ordinary file and command tools, so large files never sit in context.

Frontmatter is parsed without a YAML dependency because the standard only needs
simple ``key: value`` lines; values may be quoted and a leading list marker is
tolerated. Anything more exotic is ignored rather than failing the load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# A skill body longer than this is truncated when loaded, matching the Agent
# Skills guidance to keep SKILL.md small (~500 lines / ~5k tokens).
_MAX_BODY_CHARS = 20_000

_TRUNCATION_NOTICE = "\n\n[skill body truncated; keep SKILL.md short]"


@dataclass(frozen=True)
class Skill:
    """One loaded skill and its location."""

    name: str
    description: str
    body: str
    path: Path  # the SKILL.md file
    root: Path  # the skill folder

    def body_text(self) -> str:
        """Return the body, truncated to protect the context budget."""
        if len(self.body) <= _MAX_BODY_CHARS:
            return self.body
        return self.body[:_MAX_BODY_CHARS] + _TRUNCATION_NOTICE


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a document into a frontmatter mapping and the remaining body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text.strip()
    meta: dict[str, str] = {}
    body_start = len(lines)
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            body_start = index + 1
            break
        key, sep, value = lines[index].partition(":")
        if sep:
            meta[key.strip().lower()] = _clean_value(value)
    body = "\n".join(lines[body_start:]).strip()
    return meta, body


def _clean_value(value: str) -> str:
    cleaned = value.strip()
    if cleaned.startswith(("- ", "* ")):
        cleaned = cleaned[2:].strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1]
    return cleaned


def load_skill(skill_md: Path) -> Skill | None:
    """Load one SKILL.md file, or None when it is invalid or unreadable.

    A skill must declare a name and a description; without both it cannot take
    part in progressive disclosure and is skipped.
    """
    try:
        text = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    meta, body = _split_frontmatter(text)
    name = meta.get("name", "").strip()
    description = meta.get("description", "").strip()
    if not name or not description:
        return None
    return Skill(
        name=name,
        description=description,
        body=body,
        path=skill_md,
        root=skill_md.parent,
    )


def discover_skills(directories: Sequence[Path]) -> list[Skill]:
    """Find and load skills from a list of directories.

    Each immediate subdirectory that contains a SKILL.md is one skill, and a
    directory may itself be a single skill if it holds a SKILL.md directly.
    Results are sorted by name and de-duplicated so earlier directories win,
    which lets a workspace override a shared skill.
    """
    found: dict[str, Skill] = {}
    for directory in directories:
        for skill_md in _iter_skill_files(directory):
            skill = load_skill(skill_md)
            if skill is not None and skill.name not in found:
                found[skill.name] = skill
    return sorted(found.values(), key=lambda skill: skill.name)


def _iter_skill_files(directory: Path):
    if not directory.is_dir():
        return
    direct = directory / "SKILL.md"
    if direct.is_file():
        yield direct
    for child in sorted(directory.iterdir()):
        if child.is_dir():
            candidate = child / "SKILL.md"
            if candidate.is_file():
                yield candidate
