"""File snapshots and undo."""

from .store import MAX_SNAPSHOT_BYTES, NullSnapshotStore, SnapshotStore, UndoResult

__all__ = ["SnapshotStore", "NullSnapshotStore", "UndoResult", "MAX_SNAPSHOT_BYTES"]
