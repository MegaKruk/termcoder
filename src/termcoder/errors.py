"""Exception types used across termcoder.

A single base class makes it easy for callers to catch any expected,
recoverable error from this package while letting genuine bugs surface.
"""

from __future__ import annotations


class TermcoderError(Exception):
    """Base class for all expected, recoverable termcoder errors."""


class ConfigError(TermcoderError):
    """Raised when configuration is missing or invalid."""


class WorkspaceViolationError(TermcoderError):
    """Raised when a path would escape the configured workspace root."""


class ToolError(TermcoderError):
    """Raised when a tool cannot run due to bad arguments or internal failure."""


class ProviderError(TermcoderError):
    """Raised when the language model request fails."""
