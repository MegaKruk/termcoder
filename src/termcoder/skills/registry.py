"""Hold loaded skills and expose them to the agent.

The registry produces the startup catalog (tier 1 of progressive disclosure)
and backs the read_skill tool (tier 2). It is deliberately small: skills are
instructions and bundled files, not code that runs in process, so there is no
plugin execution here. Bundled scripts run, when a skill says so, through the
ordinary command tool inside the same sandbox and approval gate as any other
command.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, Field

from ..tools.base import ReadOnlyTool, ToolContext, ToolResult
from .loader import Skill, discover_skills


class SkillRegistry:
    """An ordered, name-addressable collection of loaded skills."""

    def __init__(self, skills: Sequence[Skill] = ()):
        self._skills: dict[str, Skill] = {skill.name: skill for skill in skills}

    @classmethod
    def from_directories(cls, directories: Sequence[Path]) -> "SkillRegistry":
        """Build a registry by discovering skills in the given directories."""
        return cls(discover_skills(directories))

    def __len__(self) -> int:
        return len(self._skills)

    def names(self) -> list[str]:
        return list(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def catalog(self) -> str | None:
        """Return the tier-1 catalog block for the system prompt, or None.

        Only names and descriptions are included, so many skills cost little
        context. The model is told how to load a skill's full instructions.
        """
        if not self._skills:
            return None
        lines = [
            "Available skills (call read_skill with the name to load full "
            "instructions before using one):"
        ]
        for skill in self._skills.values():
            lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)


class ReadSkillArgs(BaseModel):
    """Arguments for the read_skill tool."""

    name: str = Field(description="The name of the skill to load.")


class ReadSkillTool(ReadOnlyTool):
    """Load a skill's full instructions on demand (progressive disclosure)."""

    name = "read_skill"
    description = (
        "Load the full instructions for a named skill listed in the system "
        "prompt. Call this when a skill is relevant before following it. The "
        "skill may reference bundled files under its folder, which you can open "
        "with read_file or run with run_command."
    )
    args_model = ReadSkillArgs

    def __init__(self, registry: SkillRegistry):
        self._registry = registry

    def _run(self, args: ReadSkillArgs, context: ToolContext) -> ToolResult:
        skill = self._registry.get(args.name)
        if skill is None:
            available = ", ".join(self._registry.names()) or "(none)"
            return ToolResult(
                content=f"No skill named '{args.name}'. Available: {available}.",
                ok=False,
                display="not found",
            )
        context.emit(f"loading skill: {skill.name}")
        header = (
            f"Skill: {skill.name}\n"
            f"Folder: {skill.root}\n"
            f"Description: {skill.description}\n\n"
        )
        return ToolResult(
            content=header + skill.body_text(),
            ok=True,
            display=skill.name,
        )
