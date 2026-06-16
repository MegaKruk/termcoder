"""Web search as an optional, privacy-preserving tool.

Defaults to a self-hosted SearXNG instance so queries do not go to a tracking
search provider, but any provider LiteLLM supports can be selected by config.
"""

from .search import WebSearchTool, build_web_search_tool

__all__ = ["WebSearchTool", "build_web_search_tool"]
