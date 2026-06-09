"""Run a shell command on the host, gated by approval.

Phase 1 has no sandbox, so commands run directly on the host with the
workspace as the working directory. Because that is inherently risky, this is a
mutating tool that always requires explicit approval and shows the command
first with a clear warning. A later phase moves execution into a rootless
container; the tool interface stays the same.
"""

from __future__ import annotations

import subprocess

from pydantic import BaseModel, Field

from .base import MutatingTool, ToolContext, ToolPreview, ToolResult

MAX_OUTPUT_CHARS = 20_000


class RunCommandArgs(BaseModel):
    """Arguments for the run_command tool."""

    command: str = Field(
        description="Shell command to run on the host within the workspace."
    )
    timeout_seconds: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Maximum number of seconds to allow the command to run.",
    )


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n(output truncated)"


def _format_output(return_code: int, stdout: str, stderr: str) -> str:
    parts = [f"Exit code: {return_code}"]
    if stdout.strip():
        parts.append("Standard output:\n" + stdout.rstrip())
    if stderr.strip():
        parts.append("Standard error:\n" + stderr.rstrip())
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    return _truncate("\n\n".join(parts))


class RunCommandTool(MutatingTool):
    """Run a shell command on the host machine within the workspace."""

    name = "run_command"
    description = (
        "Run a shell command on the host machine, using the workspace as the "
        "working directory. There is no sandbox in this phase, so this requires "
        "explicit user approval each time. Use it only when necessary, for "
        "example to run tests or a build. It does not have sudo or admin rights."
    )
    args_model = RunCommandArgs

    def _preview(self, args: RunCommandArgs, context: ToolContext) -> ToolPreview:
        return ToolPreview(
            summary="Run a host shell command (no sandbox)",
            detail=args.command,
            detail_kind="command",
            destructive=True,
        )

    def _apply(
        self, args: RunCommandArgs, context: ToolContext, preview: ToolPreview
    ) -> ToolResult:
        try:
            completed = subprocess.run(
                args.command,
                shell=True,
                cwd=str(context.workspace.root),
                capture_output=True,
                text=True,
                timeout=args.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                content=f"Command timed out after {args.timeout_seconds} seconds.",
                ok=False,
                display="timeout",
            )
        content = _format_output(
            completed.returncode, completed.stdout or "", completed.stderr or ""
        )
        return ToolResult(
            content=content,
            ok=completed.returncode == 0,
            display=f"exit {completed.returncode}",
        )
