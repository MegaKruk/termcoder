"""File snapshots and undo.

Before the agent writes to a file, the current state of that file is captured.
Snapshots are grouped per user turn, so a single undo reverts every file the
agent changed in its most recent turn, restoring prior contents and removing
files it newly created.

Snapshots are stored as plain JSON on disk under the workspace config
directory, so undo survives a restart. Note that run_command effects are not
snapshotted, since arbitrary commands can change anything; undo covers the
write_file and edit_file tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..workspace.paths import WorkspaceGuard

# Files larger than this are not snapshotted, to keep the store small. Their
# changes will not be undoable, which undo reports.
MAX_SNAPSHOT_BYTES = 5_000_000


@dataclass
class UndoResult:
    """A summary of what an undo operation changed."""

    label: str
    restored: list[str]
    deleted: list[str]
    skipped: list[str]


class NullSnapshotStore:
    """A no-op store used when undo is disabled or in non-interactive contexts."""

    def start_turn(self, label: str) -> None:
        pass

    def capture(self, path: Path) -> None:
        pass

    def has_undo(self) -> bool:
        return False

    def undo_last(self) -> UndoResult | None:
        return None


class SnapshotStore:
    """Records file states per turn and restores them on undo."""

    def __init__(
        self,
        root: Path,
        workspace: WorkspaceGuard | None = None,
        max_bytes: int = MAX_SNAPSHOT_BYTES,
    ):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._workspace = workspace
        self._max_bytes = max_bytes
        self._current: dict | None = None

    def start_turn(self, label: str) -> None:
        """Begin a new snapshot group for the turn about to run."""
        self._current = {"index": None, "label": label or "change", "entries": [], "paths": set()}

    def capture(self, path: Path) -> None:
        """Record the current state of a file before it is changed.

        Only the first capture of a given path within a turn is kept, since that
        is the state to roll back to.
        """
        path = Path(path)
        if self._current is None:
            self.start_turn("change")
        key = str(path)
        if key in self._current["paths"]:
            return

        existed = path.is_file()
        content: str | None = None
        restorable = True
        if existed:
            try:
                if path.stat().st_size > self._max_bytes:
                    restorable = False
                else:
                    content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                restorable = False

        self._current["paths"].add(key)
        self._current["entries"].append(
            {"path": key, "existed": existed, "content": content, "restorable": restorable}
        )
        if self._current["index"] is None:
            self._current["index"] = self._next_index()
        self._flush_current()

    def has_undo(self) -> bool:
        """True when there is at least one group that can be undone."""
        return any(self._root.glob("group-*.json"))

    def undo_last(self) -> UndoResult | None:
        """Restore the most recent snapshot group and remove it from the store."""
        files = sorted(self._root.glob("group-*.json"))
        if not files:
            return None
        latest = files[-1]
        data = json.loads(latest.read_text(encoding="utf-8"))

        restored: list[str] = []
        deleted: list[str] = []
        skipped: list[str] = []
        for entry in reversed(data["entries"]):
            target = Path(entry["path"])
            label = self._display(target)
            if not entry["existed"]:
                if self._safe_delete(target):
                    deleted.append(label)
                else:
                    skipped.append(label)
            elif entry.get("restorable") and entry.get("content") is not None:
                if self._safe_write(target, entry["content"]):
                    restored.append(label)
                else:
                    skipped.append(label)
            else:
                skipped.append(label)

        latest.unlink()
        if self._current and self._current.get("index") == data.get("index"):
            self._current = None
        return UndoResult(
            label=data.get("label", "change"),
            restored=restored,
            deleted=deleted,
            skipped=skipped,
        )

    def _display(self, path: Path) -> str:
        if self._workspace is not None:
            return self._workspace.relative(path)
        return str(path)

    @staticmethod
    def _safe_delete(path: Path) -> bool:
        try:
            if path.exists():
                path.unlink()
            return True
        except OSError:
            return False

    @staticmethod
    def _safe_write(path: Path, content: str) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return True
        except OSError:
            return False

    def _next_index(self) -> int:
        files = sorted(self._root.glob("group-*.json"))
        if not files:
            return 1
        try:
            return int(files[-1].stem.split("-")[1]) + 1
        except (IndexError, ValueError):
            return len(files) + 1

    def _flush_current(self) -> None:
        current = self._current
        path = self._root / f"group-{current['index']:06d}.json"
        payload = {
            "index": current["index"],
            "label": current["label"],
            "entries": current["entries"],
        }
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
