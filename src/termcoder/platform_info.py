"""Detect the operating system and shell termcoder is running under.

The agent runs the same on Linux, macOS and Windows, but two things differ by
platform: the commands the model should suggest (POSIX shell versus PowerShell)
and how the host command runner launches and stops processes (POSIX process
groups do not exist on Windows). Centralizing detection here keeps that
knowledge in one place instead of scattering ``sys.platform`` checks.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from pathlib import PurePath


@dataclass(frozen=True)
class PlatformInfo:
    """A description of the host OS and shell for prompts and execution."""

    os_name: str  # human label, for example "Linux" or "Windows"
    is_windows: bool
    shell: str  # best guess at the interactive shell, for example "bash"

    def describe(self) -> str:
        """Return a one-line description for the system prompt."""
        return f"{self.os_name} (shell: {self.shell})"

    def shell_family(self) -> str:
        """Return 'powershell' on Windows PowerShell, else 'posix'."""
        if self.is_windows and "powershell" in self.shell.lower():
            return "powershell"
        if self.is_windows:
            return "windows"
        return "posix"


def _detect_shell(is_windows: bool) -> str:
    """Best-effort detection of the interactive shell from the environment."""
    if is_windows:
        # PowerShell sets PSModulePath; this distinguishes it from cmd.exe.
        if os.environ.get("PSModulePath"):
            return "powershell"
        return "cmd"
    shell_path = os.environ.get("SHELL")
    if shell_path:
        return PurePath(shell_path).name or "sh"
    return "sh"


def detect_platform() -> PlatformInfo:
    """Detect the current platform and shell."""
    is_windows = sys.platform.startswith("win")
    os_name = platform.system() or ("Windows" if is_windows else "Unknown")
    return PlatformInfo(
        os_name=os_name,
        is_windows=is_windows,
        shell=_detect_shell(is_windows),
    )
