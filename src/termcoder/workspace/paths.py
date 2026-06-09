"""Workspace confinement.

Every path the agent touches is resolved and checked against the workspace
root. Parent-directory escapes and symlinks that point outside the workspace
are rejected. This is the host-side guardrail that complements the sandbox
introduced in a later phase.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import WorkspaceViolationError


class WorkspaceGuard:
    """Resolve and validate paths so they always stay inside one directory."""

    def __init__(self, root: Path):
        self._root = Path(root).resolve(strict=True)

    @property
    def root(self) -> Path:
        """The absolute, resolved workspace root."""
        return self._root

    def resolve(self, candidate: str | Path) -> Path:
        """Resolve a path and confirm it stays within the workspace root.

        Relative paths are resolved against the workspace root. Symlinks are
        followed during resolution, so a link pointing outside the workspace is
        rejected. Works for paths that do not exist yet, which is needed when
        creating new files.
        """
        raw = Path(candidate)
        base = raw if raw.is_absolute() else (self._root / raw)
        resolved = base.resolve()
        if not self._is_within(resolved):
            raise WorkspaceViolationError(
                f"Path '{candidate}' resolves outside the workspace root "
                f"'{self._root}' and was blocked."
            )
        return resolved

    def relative(self, path: str | Path) -> str:
        """Return a path expressed relative to the workspace root for display."""
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(self._root))
        except ValueError:
            return str(resolved)

    def _is_within(self, path: Path) -> bool:
        return path == self._root or path.is_relative_to(self._root)
