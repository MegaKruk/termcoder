"""Create or overwrite a file, with a diff shown for approval."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ..approval.diffing import make_unified_diff
from .base import MutatingTool, ToolContext, ToolPreview, ToolResult


class WriteFileArgs(BaseModel):
    """Arguments for the write_file tool."""

    path: str = Field(description="File path relative to the workspace root.")
    content: str = Field(description="The full new contents of the file.")


class WriteFileTool(MutatingTool):
    """Create a new file or overwrite an existing one with given content."""

    name = "write_file"
    description = (
        "Create a new file or completely overwrite an existing file with the "
        "provided content. A diff is shown for approval before anything is "
        "written. To change part of an existing file, prefer edit_file."
    )
    args_model = WriteFileArgs

    def _preview(self, args: WriteFileArgs, context: ToolContext) -> ToolPreview:
        path = context.workspace.resolve(args.path)
        exists = path.exists() and path.is_file()
        old = path.read_text(encoding="utf-8") if exists else ""
        rel = context.workspace.relative(path)
        diff = make_unified_diff(old, args.content, rel)
        if exists and not diff:
            return ToolPreview(summary=f"No changes to {rel}", skip_approval=True)
        verb = "Overwrite" if exists else "Create"
        return ToolPreview(
            summary=f"{verb} {rel}",
            detail=diff if diff else f"(new empty file {rel})",
            detail_kind="diff",
            destructive=exists,
            payload={"path": str(path)},
        )

    def _apply(
        self, args: WriteFileArgs, context: ToolContext, preview: ToolPreview
    ) -> ToolResult:
        if preview.skip_approval:
            return ToolResult(
                content="No changes were necessary.", ok=True, display="no change"
            )
        path = Path(preview.payload["path"])
        context.snapshots.capture(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args.content, encoding="utf-8")
        return ToolResult(
            content=f"Wrote {context.workspace.relative(path)}.",
            ok=True,
            display="written",
        )
