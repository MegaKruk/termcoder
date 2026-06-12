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
from ..context.compaction import CompactionResult
from ..providers.usage import UsageStats
from ..snapshots.store import UndoResult
from ..tools.base import ToolResult

_ARGS_PREVIEW_LIMIT = 200


def _shorten(text: str, limit: int = _ARGS_PREVIEW_LIMIT) -> str:
    flattened = " ".join(text.split())
    if len(flattened) <= limit:
        return flattened
    return flattened[: limit - 3] + "..."


def _format_tokens(count: int) -> str:
    """Format a token count compactly: 840 stays 840, 12400 becomes 12.4k."""
    if count < 1000:
        return str(count)
    return f"{count / 1000:.1f}k"


def _format_cost(cost: float) -> str:
    """Format a dollar cost with enough precision to be informative."""
    if cost >= 0.01:
        return f"${cost:.2f}"
    return f"${cost:.4f}"


def _format_scope(stats: UsageStats) -> str:
    text = (
        f"{_format_tokens(stats.prompt_tokens)} in + "
        f"{_format_tokens(stats.completion_tokens)} out"
    )
    if stats.cached_tokens:
        text += f" ({_format_tokens(stats.cached_tokens)} cached)"
    return text


class Renderer:
    """Render assistant output, tool activity and approval prompts."""

    def __init__(self, console: Console | None = None):
        self._console = console or Console()
        self._label_pending = False
        self._streamed = False

    @property
    def console(self) -> Console:
        return self._console

    def banner(self, workspace: Path, model_name: str, sandbox: str | None = None) -> None:
        """Show the startup banner."""
        self._console.rule("termcoder")
        self._console.print(f"workspace: {workspace}", style="dim")
        self._console.print(f"model: {model_name}", style="dim")
        if sandbox:
            self._console.print(f"sandbox: {sandbox}", style="dim")
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

    def compacted(self, result: CompactionResult) -> None:
        """Note that the conversation was compacted to save context space."""
        self._console.print(
            f"  [context] compacted {result.summarized_turns} earlier turn(s), "
            f"about {result.before_tokens} -> {result.after_tokens} tokens.",
            style="dim",
            markup=False,
        )

    def usage(self, turn: UsageStats, session: UsageStats) -> None:
        """Show a one-line token and cost readout after a turn."""
        line = (
            f"  [usage] turn: {turn.calls} call(s), {_format_scope(turn)}; "
            f"session: {_format_scope(session)}"
        )
        if session.cost_usd > 0:
            line += f" (~{_format_cost(session.cost_usd)})"
        self._console.print(line, style="dim", markup=False)

    def usage_report(self, session: UsageStats) -> None:
        """Show the full session usage report for the /usage command."""
        if session.calls == 0:
            self.info("No model calls yet this session.")
            return
        self.info(f"Session usage: {session.calls} model call(s)")
        cached = (
            f" ({_format_tokens(session.cached_tokens)} served from cache)"
            if session.cached_tokens
            else ""
        )
        self.plain(f"  input:  {session.prompt_tokens} tokens{cached}")
        self.plain(f"  output: {session.completion_tokens} tokens")
        if session.cost_usd > 0:
            self.plain(f"  estimated cost: {_format_cost(session.cost_usd)}")
        self.tool_progress(
            "Figures are estimates and include compaction summary calls."
        )

    def undone(self, result: UndoResult | None) -> None:
        """Report the outcome of an undo request."""
        if result is None:
            self.info("There is nothing to undo.")
            return
        counts = []
        if result.restored:
            counts.append(f"restored {len(result.restored)}")
        if result.deleted:
            counts.append(f"removed {len(result.deleted)}")
        if result.skipped:
            counts.append(f"skipped {len(result.skipped)}")
        summary = ", ".join(counts) if counts else "no files affected"
        self.info(f"Undid '{result.label}': {summary}.")
        for path in result.restored:
            self.tool_progress(f"restored {path}")
        for path in result.deleted:
            self.tool_progress(f"removed {path}")
        for path in result.skipped:
            self.tool_progress(f"skipped {path} (could not restore)")

    def render_approval(self, request: ApprovalRequest) -> None:
        """Show what is about to happen so the user can decide."""
        self._console.print()
        self._console.print(f"Approval needed: {request.summary}", style="bold")
        if request.detail:
            self._render_detail(request)
        if request.note:
            self.warning(request.note)

    def _render_detail(self, request: ApprovalRequest) -> None:
        if request.detail_kind == "diff":
            self._console.print(
                Syntax(request.detail, "diff", theme="ansi_dark", word_wrap=True)
            )
        elif request.detail_kind == "command":
            self._console.print(
                Syntax(request.detail, "bash", theme="ansi_dark", word_wrap=True)
            )
        else:
            self._console.print(request.detail, markup=False)
