"""Tests for the web search tool.

The tool wraps litellm.search, so these tests substitute a fake search
function to keep them offline and deterministic. They check result formatting,
the untrusted-content labelling, the disabled and error paths, and that the
SearXNG misconfiguration message is actionable.
"""

from __future__ import annotations

from types import SimpleNamespace

from termcoder.approval.auto import AutoApprover
from termcoder.config import WebSearchSettings
from termcoder.tools.base import ToolContext
from termcoder.web import build_web_search_tool
from termcoder.web.search import WebSearchTool
from termcoder.workspace.paths import WorkspaceGuard


def _context(tmp_path):
    return ToolContext(workspace=WorkspaceGuard(tmp_path), approver=AutoApprover())


def _response(results):
    return SimpleNamespace(
        results=[
            SimpleNamespace(title=t, url=u, snippet=s) for t, u, s in results
        ]
    )


def test_build_returns_none_when_disabled():
    assert build_web_search_tool(WebSearchSettings(enabled=False)) is None


def test_build_returns_tool_when_enabled():
    tool = build_web_search_tool(WebSearchSettings(enabled=True))
    assert isinstance(tool, WebSearchTool)


def test_search_formats_results_with_untrusted_label(tmp_path, monkeypatch):
    import litellm

    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return _response(
            [
                ("First Result", "https://example.com/a", "A snippet."),
                ("Second Result", "https://example.com/b", "Another snippet."),
            ]
        )

    monkeypatch.setattr(litellm, "search", fake_search)
    tool = WebSearchTool(WebSearchSettings(enabled=True, api_base="http://localhost:8080"))

    result = tool.execute(tool.args_model(query="python typing"), _context(tmp_path))

    assert result.ok
    assert "First Result" in result.content
    assert "https://example.com/a" in result.content
    assert "untrusted" in result.content.lower()
    assert result.display == "2 results"
    # The configured provider and api_base are forwarded.
    assert captured["search_provider"] == "searxng"
    assert captured["api_base"] == "http://localhost:8080"


def test_search_handles_no_results(tmp_path, monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "search", lambda **k: _response([]))
    tool = WebSearchTool(WebSearchSettings(enabled=True))

    result = tool.execute(tool.args_model(query="nothing here"), _context(tmp_path))

    assert result.ok
    assert result.display == "0 results"


def test_search_explains_missing_searxng(tmp_path, monkeypatch):
    import litellm

    def raise_missing(**kwargs):
        raise ValueError("SEARXNG_API_BASE is not set. Please set it.")

    monkeypatch.setattr(litellm, "search", raise_missing)
    tool = WebSearchTool(WebSearchSettings(enabled=True, provider="searxng"))

    result = tool.execute(tool.args_model(query="anything"), _context(tmp_path))

    assert not result.ok
    assert "no SearXNG instance is configured" in result.content


def test_search_reports_generic_failure(tmp_path, monkeypatch):
    import litellm

    def raise_other(**kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(litellm, "search", raise_other)
    tool = WebSearchTool(WebSearchSettings(enabled=True))

    result = tool.execute(tool.args_model(query="anything"), _context(tmp_path))

    assert not result.ok
    assert "connection refused" in result.content
