"""Tests for the MCP client and tool adapter.

Configuration parsing and the content renderer are tested directly. The full
client is exercised against a real FastMCP echo server launched as a stdio
subprocess, which is the same transport used in production; that test is
skipped if the mcp SDK is not importable so the suite still runs in minimal
environments.
"""

from __future__ import annotations

import sys
import textwrap
from types import SimpleNamespace

import pytest

from termcoder.approval.auto import AutoApprover, RejectingApprover
from termcoder.config import MCPServerConfig, load_config
from termcoder.mcp.client import _render_content
from termcoder.tools.base import ToolContext, ToolRegistry
from termcoder.workspace.paths import WorkspaceGuard


def _mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


requires_mcp = pytest.mark.skipif(not _mcp_available(), reason="mcp SDK not installed")


def _context(tmp_path, approver=None):
    return ToolContext(
        workspace=WorkspaceGuard(tmp_path), approver=approver or AutoApprover()
    )


def test_render_content_joins_text_blocks():
    blocks = [SimpleNamespace(text="line one"), SimpleNamespace(text="line two")]
    assert _render_content(blocks) == "line one\nline two"


def test_render_content_handles_non_text_blocks():
    blocks = [SimpleNamespace(text=None, type="image")]
    assert "image content omitted" in _render_content(blocks)


def test_render_content_handles_empty():
    assert _render_content([]) == "(no content)"


def test_config_parses_mcp_servers(tmp_path):
    config_dir = tmp_path / ".termcoder"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        textwrap.dedent(
            """
            [[mcp_servers]]
            name = "files"
            command = "uvx"
            args = ["mcp-server-files", "--root", "."]
            enabled = true

            [[mcp_servers]]
            name = "disabled-one"
            command = "foo"
            enabled = false
            """
        ),
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert len(config.mcp_servers) == 2
    first = config.mcp_servers[0]
    assert first.name == "files"
    assert first.command == "uvx"
    assert first.args == ("mcp-server-files", "--root", ".")
    assert config.mcp_servers[1].enabled is False


def test_config_rejects_mcp_server_without_command(tmp_path):
    config_dir = tmp_path / ".termcoder"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        "[[mcp_servers]]\nname = \"incomplete\"\n", encoding="utf-8"
    )
    from termcoder.errors import ConfigError

    with pytest.raises(ConfigError):
        load_config(tmp_path)


_ECHO_SERVER = textwrap.dedent(
    """
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("echo-test")

    @mcp.tool()
    def echo(text: str) -> str:
        \"\"\"Return the given text, prefixed.\"\"\"
        return f"echo: {text}"

    if __name__ == "__main__":
        mcp.run()
    """
)


@requires_mcp
def test_mcp_client_round_trip(tmp_path):
    from termcoder.mcp import MCPClient, register_mcp_tools

    server_path = tmp_path / "echo_server.py"
    server_path.write_text(_ECHO_SERVER, encoding="utf-8")

    client = MCPClient()
    registry = ToolRegistry()
    statuses: list[str] = []
    config = MCPServerConfig(
        name="echo", command=sys.executable, args=(str(server_path),)
    )
    try:
        count = register_mcp_tools(registry, client, [config], statuses.append)
        assert count == 1
        assert registry.names() == ["mcp_echo_echo"]

        tool = registry.get("mcp_echo_echo")
        # The server's JSON schema is exposed to the model.
        assert "text" in tool.schema()["function"]["parameters"]["properties"]

        # Approved call returns the server's result.
        args = tool.parse_args('{"text": "hi"}')
        result = tool.execute(args, _context(tmp_path))
        assert result.ok
        assert "echo: hi" in result.content

        # Rejected call never reaches the server.
        rejected = tool.execute(args, _context(tmp_path, RejectingApprover()))
        assert not rejected.ok
    finally:
        client.close()


@requires_mcp
def test_mcp_client_reports_unreachable_server(tmp_path):
    from termcoder.mcp import MCPClient, register_mcp_tools

    client = MCPClient()
    registry = ToolRegistry()
    statuses: list[str] = []
    config = MCPServerConfig(
        name="broken",
        command=sys.executable,
        args=("-c", "import sys; sys.exit(1)"),
        startup_timeout=10.0,
    )
    try:
        count = register_mcp_tools(registry, client, [config], statuses.append)
        assert count == 0
        assert any("broken" in status for status in statuses)
    finally:
        client.close()
