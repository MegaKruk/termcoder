"""Tool framework.

Tools are the agent's only way to affect the world. Each tool declares a
Pydantic argument model, from which a provider-agnostic function schema is
generated for the model. Tools are split into two kinds:

* :class:`ReadOnlyTool` runs immediately because it cannot change anything.
* :class:`MutatingTool` first produces a :class:`ToolPreview` (for example a
  diff or a command), asks the approver for consent, and only then applies the
  change. Approval logic lives here once so every mutating tool behaves the
  same way.

This split is the single seam future phases extend: new local tools, sandboxed
execution, and MCP-backed tools all become either read-only or mutating tools.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Callable

from pydantic import BaseModel, ValidationError

from ..approval.types import ApprovalRequest, Approver
from ..errors import ToolError
from ..workspace.paths import WorkspaceGuard


def _noop_emit(_: str) -> None:
    """Default progress sink that discards messages."""


@dataclass
class ToolContext:
    """Shared services handed to every tool invocation.

    Passing services explicitly (instead of using globals) keeps tools easy to
    test and makes their dependencies obvious.
    """

    workspace: WorkspaceGuard
    approver: Approver
    emit: Callable[[str], None] = _noop_emit


@dataclass
class ToolResult:
    """The result of running a tool.

    ``content`` is the text returned to the model. ``display`` is an optional
    short, human-friendly summary for the terminal.
    """

    content: str
    ok: bool = True
    display: str | None = None

    @classmethod
    def rejected(cls, note: str) -> "ToolResult":
        """Build a result describing that the user declined the action."""
        return cls(
            content=f"The user did not approve this action. Reason: {note}",
            ok=False,
            display="rejected",
        )


@dataclass
class ToolPreview:
    """A description of a pending mutating action, shown before it is applied.

    ``payload`` carries any data computed during preview (such as the new file
    text) through to the apply step so work is not repeated. Set
    ``skip_approval`` when there is nothing to approve, for example a no-op edit
    or an argument error that should be reported without prompting the user.
    """

    summary: str
    detail: str | None = None
    detail_kind: str = "text"
    destructive: bool = False
    skip_approval: bool = False
    payload: dict = field(default_factory=dict)


class Tool(ABC):
    """Base class for all tools."""

    name: str
    description: str
    args_model: type[BaseModel]
    is_read_only: bool = True

    def schema(self) -> dict:
        """Return the OpenAI-style function schema for this tool.

        LiteLLM translates this schema to each provider's native tool format,
        so a single definition works across OpenAI, Anthropic and Ollama.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }

    def parse_args(self, raw_arguments: str) -> BaseModel:
        """Parse and validate raw JSON arguments from the model."""
        try:
            data = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ToolError(
                f"Arguments for '{self.name}' were not valid JSON: {exc}"
            ) from exc
        try:
            return self.args_model.model_validate(data)
        except ValidationError as exc:
            raise ToolError(
                f"Invalid arguments for '{self.name}': {exc}"
            ) from exc

    @abstractmethod
    def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """Run the tool and return its result."""


class ReadOnlyTool(Tool):
    """A tool that cannot change anything and so runs without approval."""

    is_read_only = True

    def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        return self._run(args, context)

    @abstractmethod
    def _run(self, args: BaseModel, context: ToolContext) -> ToolResult:
        """Perform the read-only work."""


class MutatingTool(Tool):
    """A tool that changes state and so requires approval before applying."""

    is_read_only = False

    def execute(self, args: BaseModel, context: ToolContext) -> ToolResult:
        preview = self._preview(args, context)
        if preview.skip_approval:
            return self._apply(args, context, preview)
        request = ApprovalRequest(
            tool_name=self.name,
            summary=preview.summary,
            detail=preview.detail,
            detail_kind=preview.detail_kind,
            destructive=preview.destructive,
        )
        outcome = context.approver.request(request)
        if not outcome.approved:
            return ToolResult.rejected(outcome.feedback or "No reason given.")
        return self._apply(args, context, preview)

    @abstractmethod
    def _preview(self, args: BaseModel, context: ToolContext) -> ToolPreview:
        """Describe the pending change without applying it."""

    @abstractmethod
    def _apply(
        self, args: BaseModel, context: ToolContext, preview: ToolPreview
    ) -> ToolResult:
        """Apply the change described by the preview."""


class ToolRegistry:
    """An ordered collection of tools addressable by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool, rejecting duplicate names."""
        if tool.name in self._tools:
            raise ToolError(f"A tool named '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return the tool with the given name."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"Unknown tool: '{name}'.") from exc

    def schemas(self) -> list[dict]:
        """Return function schemas for every registered tool."""
        return [tool.schema() for tool in self._tools.values()]

    def names(self) -> list[str]:
        """Return the names of every registered tool."""
        return list(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)