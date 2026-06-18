"""A semantic code search tool over the local index.

Read-only and optional. When the agent knows roughly what it is looking for but
not the exact identifier, this retrieves relevant code by meaning. The system
prompt still steers the agent to prefer search_text (exact, fast, no model
call); this is the fallback for "where is the code that does X" style questions
on large codebases.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..config import SemanticSearchSettings
from ..tools.base import ReadOnlyTool, ToolContext, ToolResult
from .index import SemanticIndex


class SemanticSearchArgs(BaseModel):
    """Arguments for the semantic_search tool."""

    query: str = Field(
        description="A natural-language description of the code you are looking for."
    )
    max_results: int = Field(
        default=5, ge=1, le=20, description="Maximum number of code spans to return."
    )


class SemanticSearchTool(ReadOnlyTool):
    """Find code by meaning using the local semantic index."""

    name = "semantic_search"
    description = (
        "Search the codebase by meaning rather than exact text, using a local "
        "embedding index. Use this when you do not know the exact name to grep "
        "for; prefer search_text when you do. Returns file paths and line "
        "ranges you can then open with read_file."
    )
    args_model = SemanticSearchArgs

    def __init__(self, index: SemanticIndex, settings: SemanticSearchSettings):
        self._index = index
        self._settings = settings

    def _run(self, args: SemanticSearchArgs, context: ToolContext) -> ToolResult:
        limit = min(args.max_results, self._settings.max_results) or args.max_results
        context.emit("searching the semantic index")
        hits, reason = self._index.search(args.query, limit)
        if reason is not None:
            return ToolResult(content=f"Semantic search unavailable: {reason}.", ok=False, display="unavailable")
        if not hits:
            return ToolResult(
                content=f"No semantic matches for: {args.query}",
                ok=True,
                display="0 results",
            )
        return ToolResult(
            content=self._format(args.query, hits),
            ok=True,
            display=f"{len(hits)} results",
        )

    @staticmethod
    def _format(query: str, hits: list) -> str:
        lines = [f"Semantic matches for: {query}", ""]
        for index, hit in enumerate(hits, start=1):
            lines.append(
                f"{index}. {hit.rel_path}:{hit.start_line}-{hit.end_line} "
                f"(score {hit.score:.2f})"
            )
            snippet = hit.text.strip().splitlines()
            preview = "\n".join(snippet[:8])
            lines.append(preview)
            lines.append("")
        return "\n".join(lines).rstrip()
