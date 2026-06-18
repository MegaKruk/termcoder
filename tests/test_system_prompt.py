"""Tests for system prompt composition.

These assert the behavioral guidance that the prompt is responsible for: the
single-approval instruction that prevents the model from asking for permission
in prose before a tool call, the platform-specific shell guidance, and the
inclusion of memory section names so the model edits the right heading.
"""

from __future__ import annotations

from pathlib import Path

from termcoder.agent.system_prompt import build_system_prompt
from termcoder.memory.loader import ProjectMemory, parse_sections


def _prompt(**kwargs) -> str:
    return build_system_prompt(
        Path("/workspace"), ["read_file", "run_command"], "Linux", **kwargs
    )


def test_prompt_tells_model_not_to_double_confirm():
    prompt = _prompt()
    lowered = prompt.lower()
    assert "approval prompt" in lowered
    # The instruction that addresses the redundant verbal confirmation.
    assert "call the tool directly" in lowered
    assert "twice" in lowered


def test_prompt_includes_posix_shell_guidance():
    prompt = build_system_prompt(
        Path("/w"), ["read_file"], "Linux", shell_family="posix"
    )
    assert "POSIX shell" in prompt


def test_prompt_includes_powershell_guidance():
    prompt = build_system_prompt(
        Path("C:/w"), ["read_file"], "Windows", shell_family="powershell"
    )
    assert "PowerShell" in prompt
    assert "Get-ChildItem" in prompt


def test_prompt_lists_memory_sections(tmp_path):
    text = "# Conventions\nUse spaces.\n\n# Architecture\nLoop in loop.py.\n"
    memory = ProjectMemory(
        path=Path("TERMCODER.md"), text=text, sections=parse_sections(text)
    )
    prompt = _prompt(memory=memory)
    assert "organized into sections" in prompt
    assert "Conventions" in prompt
    assert "Architecture" in prompt


def test_prompt_without_memory_has_no_section_note():
    prompt = _prompt()
    assert "organized into sections" not in prompt
