"""A synchronous client for MCP (Model Context Protocol) servers.

MCP is the de-facto standard for giving agents external tools, resources and
prompts. Its Python SDK is asyncio-based, but termcoder's agent loop is
synchronous, so this module runs a single asyncio event loop in a dedicated
background thread and marshals blocking calls onto it. Each configured server
gets one long-lived session for the life of the program.

Only the stdio transport is implemented here. It is the right default for a
local-first, privacy-preserving tool: servers run as local subprocesses with no
network exposure. The design leaves room for HTTP or SSE transports to be added
as more connection types without changing callers.

Security note: MCP tools are model-controlled and an external supply-chain
surface (tool poisoning, shadowing, prompt injection). termcoder treats every
MCP tool as a mutating action behind the approval gate, and servers should only
be configured from trusted sources.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

from ..config import MCPServerConfig


@dataclass(frozen=True)
class MCPToolInfo:
    """A tool advertised by an MCP server."""

    server: str
    name: str
    description: str
    input_schema: dict


class MCPConnectionError(Exception):
    """Raised when an MCP server cannot be connected or queried."""


class _BackgroundLoop:
    """An asyncio event loop running on its own thread.

    A single shared loop hosts every server session, so all MCP I/O happens on
    one thread and synchronous callers simply wait on futures.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="termcoder-mcp", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: float | None = None) -> Any:
        """Run a coroutine on the loop and block for its result."""
        future: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def stop(self) -> None:
        """Stop the loop and join its thread."""
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


class _ServerSession:
    """Owns one server's MCP session and its async lifecycle."""

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._session = None
        self._exit_stack = None

    async def open(self) -> list[MCPToolInfo]:
        """Start the server, initialize the session and list its tools."""
        from contextlib import AsyncExitStack

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self._config.command,
            args=list(self._config.args),
            env=dict(self._config.env) or None,
            cwd=self._config.cwd or None,
        )
        read, write = await self._exit_stack.enter_async_context(
            stdio_client(params)
        )
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )
        await session.initialize()
        self._session = session
        listed = await session.list_tools()
        return [
            MCPToolInfo(
                server=self._config.name,
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema or {"type": "object", "properties": {}},
            )
            for tool in listed.tools
        ]

    async def call(self, tool_name: str, arguments: dict) -> str:
        """Call a tool and return its content rendered as text."""
        if self._session is None:
            raise MCPConnectionError(f"Server '{self._config.name}' is not connected.")
        result = await self._session.call_tool(tool_name, arguments)
        text = _render_content(result.content)
        if getattr(result, "isError", False):
            return f"The tool reported an error:\n{text}"
        return text

    async def close(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None


def _render_content(content: Any) -> str:
    """Render MCP result content blocks into plain text."""
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
            continue
        block_type = getattr(block, "type", "content")
        parts.append(f"[{block_type} content omitted]")
    return "\n".join(parts) if parts else "(no content)"


class MCPClient:
    """Connects to configured MCP servers and runs their tools synchronously."""

    def __init__(self, call_timeout: float = 60.0):
        self._loop = _BackgroundLoop()
        self._sessions: dict[str, _ServerSession] = {}
        self._call_timeout = call_timeout

    def connect(self, config: MCPServerConfig) -> list[MCPToolInfo]:
        """Start one server and return the tools it advertises.

        Connection failures are raised as MCPConnectionError so the caller can
        report the bad server and continue with the rest.
        """
        session = _ServerSession(config)
        try:
            tools = self._loop.run(session.open(), timeout=config.startup_timeout)
        except Exception as exc:
            raise MCPConnectionError(
                f"Could not start MCP server '{config.name}': {exc}"
            ) from exc
        self._sessions[config.name] = session
        return tools

    def call_tool(self, server: str, tool_name: str, arguments: dict) -> str:
        """Call a tool on a connected server and return its text result."""
        session = self._sessions.get(server)
        if session is None:
            raise MCPConnectionError(f"MCP server '{server}' is not connected.")
        return self._loop.run(
            session.call(tool_name, arguments), timeout=self._call_timeout
        )

    def close(self) -> None:
        """Close all sessions and stop the background loop."""
        for session in self._sessions.values():
            try:
                self._loop.run(session.close(), timeout=5)
            except Exception:
                pass
        self._sessions.clear()
        self._loop.stop()
