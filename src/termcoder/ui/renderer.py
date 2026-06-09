"""Terminal rendering with Rich.

The renderer owns all styled output: streamed assistant text, tool activity,
diffs and command previews for approval, and status lines. Streaming uses
markup-free printing so tokens that contain characters like '[' are shown
literally and never parsed as markup.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

from ..approval.types import ApprovalRequest
from ..tools.base import ToolResult

_ARGS_PREVIEW_LIMIT = 200


def _shorten(text: str, limit: int = _ARGS_PREVIEW_LIMIT) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3] + "..."


class Renderer:
    """Render assistant output, tool activity and approval prompts."""

    def __init__(self, console: Console | None = None):
        self._console = console or Console()
        self._label_pending = False
        self._streamed = False

    @property
    def console(self) -> Console:
        return self._console

    def banner(self, workspace: Path, model_name: str) -> None:
        """Show the startup banner."""
        self._console.rule("termcoder")
        self._console.print(f"workspace: {workspace}", style="dim")
        self._console.print(f"model: {model_name}", style="dim")
        self._console.print(
            "Type a message, or /help for commands. Ctrl-D to exit.", style="dim"
        )

    def info(self, text: str) -> None:
        self._console.print(text, style="cyan")

    def warning(self, text: str) -> None:
        self._console.print(text, style="yellow")

    def error(self, text: str) -> None:
        self._console.print(text, style="bold red")

    def plain(self, text: str) -> None:
        self._console.print(text)

    def tool_progress(self, text: str) -> None:
        self._console.print(f"  {text}", style="dim")

    def begin_assistant(self) -> None:
        """Prepare for assistant output. The label is printed lazily."""
        self._label_pending = True
        self._streamed = False

    def stream_assistant(self, text: str) -> None:
        """Print a streamed token from the assistant."""
        if self._label_pending:
            self._console.print("assistant>", style="bold green")
            self._label_pending = False
        self._streamed = True
        self._console.print(text, end="", markup=False, highlight=False, soft_wrap=True)

    def end_assistant(self) -> None:
        """Finish an assistant turn, adding a newline only if text was printed."""
        if self._streamed:
            self._console.print()
        self._label_pending = False
        self._streamed = False

    def tool_started(self, name: str, raw_args: str) -> None:
        self._console.print(
            f"  [tool] {name} {_shorten(raw_args)}", style="dim", markup=False
        )

    def tool_finished(self, name: str, result: ToolResult) -> None:
        summary = result.display or ("ok" if result.ok else "failed")
        style = "green" if result.ok else "red"
        self._console.print(f"  [tool] {name}: {summary}", style=style, markup=False)

    def render_approval(self, request: ApprovalRequest) -> None:
        """Show what is about to happen so the user can decide."""
        self._console.print()
        self._console.print(f"Approval needed: {request.summary}", style="bold")
        if request.detail:
            self._render_detail(request)
        if request.destructive:
            self.warning("This action changes your system and cannot be auto-undone.")

    def _render_detail(self, request: ApprovalRequest) -> None:
        if request.detail_kind == "diff":
            self._console.print(
                Syntax(request.detail, "diff", theme="ansi_dark", word_wrap=True)
            )
        elif request.detail_kind == "command":
            self._console.print(
                Syntax(request.detail, "bash", theme="ansi_dark", word_wrap=True)
            )
            self.warning(
                "This command runs on your host with no sandbox in this phase."
            )
        else:
            self._console.print(request.detail, markup=False)
