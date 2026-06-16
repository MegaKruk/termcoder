"""Adapt MCP server tools into termcoder tools behind the approval gate.

Each tool an MCP server advertises becomes one :class:`MCPTool`. Because MCP
tools are model-controlled and may have arbitrary side effects, every call is
treated as a mutating action: the user sees the server, tool name and arguments
and must approve before the call runs. This satisfies the protocol's
human-in-the-loop requirement and matches how termcoder gates its own
command tool.

Tool names are namespaced as ``mcp_<server>_<tool>`` so two servers cannot
collide (tool shadowing is a known MCP attack), while the original tool name is
preserved for the actual call.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel

from ..config import MCPServerConfig
from ..tools.base import MutatingTool, ToolContext, ToolPreview, ToolRegistry, ToolResult
from .client import MCPClient, MCPConnectionError, MCPToolInfo

_MAX_RESULT_CHARS = 20_000


def _sanitize(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


class _PassthroughArgs(BaseModel):
    """Permissive argument model; MCP servers validate their own input."""

    model_config = {"extra": "allow"}


class MCPTool(MutatingTool):
    """A single MCP server tool, callable behind the approval gate."""

    args_model = _PassthroughArgs

    def __init__(self, client: MCPClient, info: MCPToolInfo):
        self._client = client
        self._info = info
        self.name = f"mcp_{_sanitize(info.server)}_{_sanitize(info.name)}"
        origin = f"Provided by MCP server '{info.server}'. "
        self.description = origin + (info.description or "No description provided.")

    def schema(self) -> dict:
        """Use the server-provided JSON schema for arguments directly."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._info.input_schema,
            },
        }

    def parse_args(self, raw_arguments: str) -> BaseModel:
        """Keep the model's arguments as-is for forwarding to the server."""
        data = json.loads(raw_arguments or "{}")
        return _PassthroughArgs.model_validate(data)

    def _preview(self, args: BaseModel, context: ToolContext) -> ToolPreview:
        arguments = args.model_dump()
        detail = json.dumps(arguments, indent=2, ensure_ascii=True) if arguments else "(no arguments)"
        return ToolPreview(
            summary=f"Call MCP tool '{self._info.name}' on server '{self._info.server}'",
            detail=detail,
            detail_kind="text",
            destructive=True,
            note="This runs an external MCP tool, which may have side effects.",
            payload={"arguments": arguments},
        )

    def _apply(
        self, args: BaseModel, context: ToolContext, preview: ToolPreview
    ) -> ToolResult:
        arguments = preview.payload.get("arguments", {})
        context.emit(f"calling MCP tool {self._info.server}:{self._info.name}")
        try:
            output = self._client.call_tool(
                self._info.server, self._info.name, arguments
            )
        except MCPConnectionError as exc:
            return ToolResult(content=str(exc), ok=False, display="error")
        if len(output) > _MAX_RESULT_CHARS:
            output = output[:_MAX_RESULT_CHARS] + "\n(output truncated)"
        return ToolResult(content=output, ok=True, display=f"{self._info.server} ok")


def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClient,
    servers: list[MCPServerConfig],
    on_status,
) -> int:
    """Connect to each server and register its tools, returning the count.

    Connection failures are reported through ``on_status`` and skipped so one
    bad server does not stop the others or the session.
    """
    total = 0
    for server in servers:
        if not server.enabled:
            continue
        try:
            tools = client.connect(server)
        except MCPConnectionError as exc:
            on_status(f"MCP: {exc}")
            continue
        for info in tools:
            tool = MCPTool(client, info)
            try:
                registry.register(tool)
            except Exception as exc:
                on_status(f"MCP: skipped tool {tool.name}: {exc}")
                continue
            total += 1
        on_status(
            f"MCP: connected '{server.name}' ({len(tools)} tool(s))"
        )
    return total
