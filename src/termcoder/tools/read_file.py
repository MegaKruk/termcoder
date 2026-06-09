"""Read a text file from the workspace."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .base import ReadOnlyTool, ToolContext, ToolResult

MAX_READ_BYTES = 2_000_000


class ReadFileArgs(BaseModel):
    """Arguments for the read_file tool."""

    path: str = Field(description="File path relative to the workspace root.")
    start_line: int | None = Field(
        default=None, ge=1, description="Optional 1-based first line to include."
    )
    end_line: int | None = Field(
        default=None, ge=1, description="Optional 1-based last line to include."
    )


def _select_lines(
    lines: list[str], start: int | None, end: int | None
) -> tuple[list[str], str]:
    """Apply an optional inclusive 1-based line range and summarize the result."""
    total = len(lines)
    if start is None and end is None:
        return lines, f"{total} lines"
    first = (start or 1) - 1
    last = end if end is not None else total
    first = max(0, first)
    last = min(total, last)
    if first >= last:
        return [], f"0 of {total} lines (empty range)"
    return lines[first:last], f"lines {first + 1}-{last} of {total}"


class ReadFileTool(ReadOnlyTool):
    """Read the contents of a UTF-8 text file inside the workspace."""

    name = "read_file"
    description = (
        "Read the contents of a UTF-8 text file inside the workspace. "
        "Optionally restrict the output to an inclusive 1-based line range. "
        "Returns the raw file text so it can be used as the basis for an edit."
    )
    args_model = ReadFileArgs

    def _run(self, args: ReadFileArgs, context: ToolContext) -> ToolResult:
        path = context.workspace.resolve(args.path)
        if not path.exists():
            return ToolResult(content=f"File not found: {args.path}", ok=False)
        if path.is_dir():
            return ToolResult(
                content=f"Path is a directory, not a file: {args.path}", ok=False
            )
        if path.stat().st_size > MAX_READ_BYTES:
            return ToolResult(
                content=(
                    f"File is too large to read in full ({path.stat().st_size} "
                    "bytes). Use a line range to read part of it."
                ),
                ok=False,
            )
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                content=f"File is not valid UTF-8 text: {args.path}", ok=False
            )
        selected, summary = _select_lines(
            text.splitlines(), args.start_line, args.end_line
        )
        return ToolResult(content="\n".join(selected), ok=True, display=summary)
