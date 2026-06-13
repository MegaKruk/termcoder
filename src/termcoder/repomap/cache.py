"""Plain JSON cache for extracted tags, keyed by file mtime and size.

Parsing every source file on each startup would make the repository map slow
on real repositories, so tags are cached and a file is only re-parsed when its
modification time or size changes. JSON keeps the cache inspectable, in line
with the project's plain-text state principle. A corrupt or version-mismatched
cache is silently discarded and rebuilt.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tags import Tag

_VERSION = 1


class TagCache:
    """Loads, queries and persists per-file tag lists."""

    def __init__(self, path: Path):
        self._path = path
        self._files: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict) and data.get("version") == _VERSION:
            files = data.get("files")
            if isinstance(files, dict):
                self._files = files

    def get(self, rel_path: str, mtime: float, size: int) -> list[Tag] | None:
        """Return cached tags when the file is unchanged, else None."""
        entry = self._files.get(rel_path)
        if not entry or entry.get("mtime") != mtime or entry.get("size") != size:
            return None
        tags: list[Tag] = []
        for item in entry.get("tags", []):
            try:
                name, kind, line, text = item
            except (TypeError, ValueError):
                return None
            tags.append(
                Tag(rel_path=rel_path, name=name, kind=kind, line=line, text=text)
            )
        return tags

    def put(self, rel_path: str, mtime: float, size: int, tags: list[Tag]) -> None:
        """Store freshly extracted tags for a file."""
        self._files[rel_path] = {
            "mtime": mtime,
            "size": size,
            "tags": [[tag.name, tag.kind, tag.line, tag.text] for tag in tags],
        }

    def prune(self, keep: set[str]) -> None:
        """Drop cache entries for files that no longer exist."""
        self._files = {rel: entry for rel, entry in self._files.items() if rel in keep}

    def save(self) -> None:
        """Persist the cache; failures are ignored since it is only a cache."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": _VERSION, "files": self._files}
            self._path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
