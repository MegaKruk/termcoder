"""Agent-Skills-style SKILL.md loading with progressive disclosure."""

from .loader import Skill, discover_skills, load_skill
from .registry import ReadSkillTool, SkillRegistry

__all__ = [
    "Skill",
    "load_skill",
    "discover_skills",
    "SkillRegistry",
    "ReadSkillTool",
]
