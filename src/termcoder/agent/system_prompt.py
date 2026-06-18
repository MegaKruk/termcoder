"""System prompt construction.

The prompt is kept short and concrete. It states the safety rules the harness
enforces anyway (workspace confinement, approval for writes and commands) so
the model's behavior matches the guardrails instead of fighting them.

Two optional, session-stable blocks are appended: project memory (a markdown
file the user maintains) and the repository map (a token-budgeted symbol
overview). Both are computed once per session so the system prompt stays
byte-stable and provider prompt caches keep hitting.
"""

from __future__ import annotations

from pathlib import Path

from ..memory.loader import ProjectMemory


def _shell_guidance(shell_family: str) -> str:
    """Return a short instruction on which shell syntax commands should use."""
    if shell_family == "powershell":
        return (
            "Windows PowerShell. Use PowerShell syntax and cmdlets (for example "
            "Get-ChildItem, Remove-Item, $env:VAR); do not use bash-only syntax."
        )
    if shell_family == "windows":
        return "Windows command shell (cmd.exe). Use Windows command syntax."
    return "a POSIX shell (bash or zsh). Use standard POSIX shell syntax."


def build_system_prompt(
    workspace_root: Path,
    tool_names: list[str],
    os_name: str,
    memory: ProjectMemory | None = None,
    repo_map: str | None = None,
    skill_catalog: str | None = None,
    shell_family: str = "posix",
) -> str:
    """Return the system prompt for an interactive coding session."""
    tools = ", ".join(tool_names)
    parts = [
        (
            "You are termcoder, a careful coding assistant that works inside a "
            "developer's terminal.\n"
            "\n"
            "Environment and rules:\n"
            f"- You operate strictly inside this workspace directory: {workspace_root}\n"
            f"- Operating system: {os_name}\n"
            f"- Shell for commands: {_shell_guidance(shell_family)}\n"
            "- You cannot access paths outside the workspace; attempts are blocked.\n"
            "- Every file write and every shell command must be approved by the "
            "user, who sees a diff or the exact command first. Shell commands run "
            "in a sandbox container when one is available, otherwise on the host. "
            "Use them sparingly.\n"
            "- File edits can be reverted by the user with an undo command, but "
            "command side effects cannot, so be especially careful with commands.\n"
            "- You have no sudo or administrator rights and no general web access.\n"
            "\n"
            "Approvals (important):\n"
            "- The terminal shows the user a separate approval prompt for every "
            "tool call, where they accept or reject it. That prompt is the only "
            "confirmation step, so do not ask the user in words for permission "
            "to run a tool and then wait. When you decide to act, call the tool "
            "directly; the approval prompt handles consent. Asking first in prose "
            "and then calling the tool forces the user to confirm twice.\n"
            "- If the user already told you to do something, just do it by calling "
            "the tool. Only ask a clarifying question in prose when you genuinely "
            "need information you cannot get from a tool.\n"
            "\n"
            "How to work:\n"
            "- Inspect before you change. Use search_text and find_files to locate "
            "code, then read_file with a line range, rather than guessing or "
            "reading whole large files.\n"
            "- Make small, focused edits. Prefer edit_file for changes to existing "
            "files and write_file to create new ones.\n"
            "- After you act, briefly explain what you did and why.\n"
            "- If the user rejects a change or command, read their feedback and "
            "adjust instead of repeating the same action.\n"
            "- When the task is complete, stop and summarize the outcome.\n"
            "\n"
            f"Available tools: {tools}."
        )
    ]
    if memory is not None:
        section_note = ""
        titles = memory.section_titles()
        if len(titles) > 1:
            section_note = (
                " This file is organized into sections: "
                + ", ".join(titles)
                + ". When you add to memory, put the new note under the most "
                "fitting section heading, or add a new heading if none fits."
            )
        parts.append(
            f"Project memory (from {memory.path.name}, maintained by the user; "
            "follow it as project-specific instructions and facts):\n"
            f"{memory.text}\n"
            "When the user asks you to remember something durable about the "
            f"project, propose an edit to {memory.path.name} with the file "
            "tools." + section_note
        )
    if repo_map:
        parts.append(
            "Repository map (the highest-ranked symbols as 'line| definition', "
            "a snapshot from session start; read files for current details):\n"
            f"{repo_map}"
        )
    if skill_catalog:
        parts.append(skill_catalog)
    return "\n\n".join(parts)
