"""Replace an exact piece of text in an existing file, with diff approval.

This mirrors the string-replacement edit pattern used by tools such as Aider
and Claude Code. By default the target text must match exactly once so edits
are unambiguous; set replace_all to change every occurrence.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..approval.diffing import make_unified_diff
from .base import MutatingTool, ToolContext, ToolPreview, ToolResult


class EditFileArgs(BaseModel):
    """Arguments for the edit_file tool."""

    path: str = Field(description="File path relative to the workspace root.")
    old_string: str = Field(
        description="Exact text to find. Must match the file content verbatim."
    )
    new_string: str = Field(description="Text that replaces the matched text.")
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring a unique match.",
    )


class EditFileTool(MutatingTool):
    """Replace exact text within an existing file."""

    name = "edit_file"
    description = (
        "Replace an exact piece of text in an existing file. By default the "
        "match must be unique; include enough surrounding context to make it so, "
        "or set replace_all to change every occurrence. A diff is shown for "
        "approval before the file changes."
    )
    args_model = EditFileArgs

    def _preview(self, args: EditFileArgs, context: ToolContext) -> ToolPreview:
        path = context.workspace.resolve(args.path)
        if not (path.exists() and path.is_file()):
            return ToolPreview(
                summary=f"File not found: {context.workspace.relative(path)}",
                skip_approval=True,
                payload={"error": "not_found"},
            )
        old = path.read_text(encoding="utf-8")
        occurrences = old.count(args.old_string)
        if occurrences == 0:
            return ToolPreview(
                summary="Text to replace was not found.",
                skip_approval=True,
                payload={"error": "no_match"},
            )
        if occurrences > 1 and not args.replace_all:
            return ToolPreview(
                summary=f"Text matches {occurrences} times and is not unique.",
                skip_approval=True,
                payload={"error": "ambiguous", "count": occurrences},
            )
        count = -1 if args.replace_all else 1
        new = old.replace(args.old_string, args.new_string, count)
        rel = context.workspace.relative(path)
        diff = make_unified_diff(old, new, rel)
        if not diff:
            return ToolPreview(summary=f"No changes to {rel}", skip_approval=True)
        return ToolPreview(
            summary=f"Edit {rel}",
            detail=diff,
            detail_kind="diff",
            destructive=True,
            payload={"path": str(path), "new": new},
        )

    def _apply(
        self, args: EditFileArgs, context: ToolContext, preview: ToolPreview
    ) -> ToolResult:
        error = preview.payload.get("error")
        if error == "not_found":
            return ToolResult(
                content="The file does not exist. Create it with write_file first.",
                ok=False,
            )
        if error == "no_match":
            return ToolResult(
                content=(
                    "The exact text was not found. Re-read the file and copy the "
                    "target text verbatim, including whitespace."
                ),
                ok=False,
            )
        if error == "ambiguous":
            return ToolResult(
                content=(
                    f"The text matched {preview.payload['count']} times. Add more "
                    "surrounding context to make it unique, or set replace_all."
                ),
                ok=False,
            )
        if preview.skip_approval:
            return ToolResult(
                content="No changes were necessary.", ok=True, display="no change"
            )
        path = Path(preview.payload["path"])
        path.write_text(preview.payload["new"], encoding="utf-8")
        return ToolResult(
            content=f"Edited {context.workspace.relative(path)}.",
            ok=True,
            display="edited",
        )
