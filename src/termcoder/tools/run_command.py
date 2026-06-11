"""Run a shell command, gated by approval, through a pluggable runner.

The tool itself does not know whether the command runs on the host or in a
sandbox container. It delegates to a :class:`CommandRunner`, which is chosen
from configuration. The approval prompt shows the command plus a note from the
runner describing where it will run, so the user always knows the risk before
approving.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..sandbox.runner import CommandResult, CommandRunner, HostCommandRunner
from .base import MutatingTool, ToolContext, ToolPreview, ToolResult

MAX_OUTPUT_CHARS = 20_000


class RunCommandArgs(BaseModel):
    """Arguments for the run_command tool."""

    command: str = Field(
        description="Shell command to run within the workspace."
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


def _format_result(result: CommandResult) -> str:
    parts = [f"Exit code: {result.returncode}"]
    if result.timed_out:
        parts.append("The command timed out and was stopped.")
    if result.oom_killed:
        parts.append("The command was killed, most likely for exceeding the memory limit.")
    if result.stdout.strip():
        parts.append("Standard output:\n" + result.stdout.rstrip())
    if result.stderr.strip():
        parts.append("Standard error:\n" + result.stderr.rstrip())
    if not result.stdout.strip() and not result.stderr.strip() and not result.timed_out:
        parts.append("(no output)")
    return _truncate("\n\n".join(parts))


class RunCommandTool(MutatingTool):
    """Run a shell command within the workspace, on the host or in a sandbox."""

    name = "run_command"
    description = (
        "Run a shell command with the workspace as the working directory. "
        "Requires explicit user approval each time. Use it when necessary, for "
        "example to run tests or a build. It has no sudo or admin rights. "
        "Depending on configuration it runs in a sandbox container or on the "
        "host."
    )
    args_model = RunCommandArgs

    def __init__(self, runner: CommandRunner | None = None, workspace_root: Path | None = None):
        self._runner = runner or HostCommandRunner(workspace_root or Path.cwd())

    def _preview(self, args: RunCommandArgs, context: ToolContext) -> ToolPreview:
        location = "in a sandbox" if self._runner.is_sandboxed else "on the host"
        return ToolPreview(
            summary=f"Run a shell command ({location})",
            detail=args.command,
            detail_kind="command",
            destructive=True,
            note=self._runner.describe(),
        )

    def _apply(
        self, args: RunCommandArgs, context: ToolContext, preview: ToolPreview
    ) -> ToolResult:
        result = self._runner.run(args.command, args.timeout_seconds)
        if result.error:
            return ToolResult(
                content=f"Could not run the command: {result.error}",
                ok=False,
                display="error",
            )
        display = "timeout" if result.timed_out else f"{result.backend} exit {result.returncode}"
        return ToolResult(content=_format_result(result), ok=result.ok, display=display)
