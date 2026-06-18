"""Split source files into chunks suitable for embedding.

The design doc points at AST/cAST chunking for the most syntactically faithful
splits. To keep dependencies light and behavior predictable, this uses a
simpler line-window chunker: fixed-size windows with overlap, walked over the
same file set the repository map already understands. It is language-agnostic,
never splits mid-file in a way that loses content, and is good enough to make
semantic retrieval useful. The chunker is isolated behind one function so an
AST-based splitter can replace it later without touching the index or tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..repomap.tags import LANGUAGES
from ..workspace.ignore import IGNORED_DIRS

# Window sizing in lines. Overlap keeps a definition that straddles a boundary
# discoverable from both chunks.
_CHUNK_LINES = 60
_CHUNK_OVERLAP = 15
_MAX_FILE_BYTES = 512_000
_MAX_FILES = 5000


@dataclass(frozen=True)
class CodeChunk:
    """One embeddable span of a source file."""

    rel_path: str
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str

    def identifier(self) -> str:
        """A stable id for this chunk within the workspace."""
        return f"{self.rel_path}:{self.start_line}-{self.end_line}"


def discover_source_files(root: Path) -> list[tuple[str, Path]]:
    """Find supported source files as sorted (relative path, path) pairs.

    Mirrors the repository map's discovery so the two features agree on which
    files are code and which directories are noise.
    """
    import os

    found: list[tuple[str, Path]] = []
    for current, dirs, names in os.walk(root):
        dirs[:] = sorted(
            d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")
        )
        for name in sorted(names):
            path = Path(current) / name
            if path.suffix.lower() not in LANGUAGES:
                continue
            try:
                if path.stat().st_size > _MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append((path.relative_to(root).as_posix(), path))
            if len(found) >= _MAX_FILES:
                return found
    return found


def chunk_text(rel_path: str, text: str) -> list[CodeChunk]:
    """Split one file's text into overlapping line-window chunks."""
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[CodeChunk] = []
    step = max(1, _CHUNK_LINES - _CHUNK_OVERLAP)
    start = 0
    while start < len(lines):
        end = min(start + _CHUNK_LINES, len(lines))
        body = "\n".join(lines[start:end]).strip()
        if body:
            chunks.append(
                CodeChunk(
                    rel_path=rel_path,
                    start_line=start + 1,
                    end_line=end,
                    text=body,
                )
            )
        if end == len(lines):
            break
        start += step
    return chunks
