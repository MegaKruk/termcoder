"""A local semantic index of workspace code, backed by LanceDB.

Chunks of source are embedded with a configurable model (local Ollama by
default, so code never leaves the machine) and stored in an embedded LanceDB
table under the workspace cache directory. Retrieval embeds the query and runs
a vector search. Everything here degrades gracefully: if LanceDB is not
installed or the embedding model is unreachable, the index reports the reason
instead of raising, because semantic search is an optional enhancement on top
of agentic grep, never a hard requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import SemanticSearchSettings
from .chunker import CodeChunk, chunk_text, discover_source_files

_TABLE_NAME = "code_chunks"
_EMBED_BATCH = 64


def _table_names(db: object) -> list[str]:
    """Return table names across LanceDB versions.

    Older versions return a plain list from ``list_tables``; newer versions
    return a paginated object exposing the names under ``tables``.
    """
    listing = db.list_tables()
    names = getattr(listing, "tables", listing)
    return list(names)


@dataclass(frozen=True)
class SearchHit:
    """One semantic search result."""

    rel_path: str
    start_line: int
    end_line: int
    text: str
    score: float


@dataclass(frozen=True)
class IndexStatus:
    """The outcome of building or updating the index."""

    ok: bool
    chunk_count: int = 0
    file_count: int = 0
    reason: str | None = None


def lancedb_available() -> bool:
    """True when LanceDB and pyarrow can be imported."""
    try:
        import lancedb  # noqa: F401
        import pyarrow  # noqa: F401
    except Exception:
        return False
    return True


class SemanticIndex:
    """Builds and queries the workspace semantic index."""

    def __init__(self, root: Path, db_path: Path, settings: SemanticSearchSettings):
        self._root = Path(root)
        self._db_path = Path(db_path)
        self._settings = settings

    def build(self) -> IndexStatus:
        """Embed all workspace chunks into the index, replacing prior contents.

        The index is rebuilt wholesale rather than incrementally; for the
        workspace sizes this targets the cost is acceptable and it keeps the
        stored vectors consistent with the current embedding model.
        """
        if not lancedb_available():
            return IndexStatus(
                ok=False,
                reason=(
                    "LanceDB is not installed; install termcoder with the "
                    "'semantic' extra to enable semantic search"
                ),
            )
        files = discover_source_files(self._root)
        if not files:
            return IndexStatus(ok=False, reason="no supported source files to index")

        chunks: list[CodeChunk] = []
        for rel, path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            chunks.extend(chunk_text(rel, text))
        if not chunks:
            return IndexStatus(ok=False, reason="no content to index")

        try:
            vectors = self._embed([chunk.text for chunk in chunks])
        except _EmbeddingError as exc:
            return IndexStatus(ok=False, reason=str(exc))

        try:
            self._write_table(chunks, vectors)
        except Exception as exc:
            return IndexStatus(ok=False, reason=f"could not write the index: {exc}")

        return IndexStatus(
            ok=True, chunk_count=len(chunks), file_count=len(files)
        )

    def search(self, query: str, limit: int) -> tuple[list[SearchHit], str | None]:
        """Return the top matching chunks for a query, with an optional reason.

        On any failure the hit list is empty and the reason explains why, so
        the caller can surface an actionable message to the model.
        """
        if not lancedb_available():
            return [], "LanceDB is not installed"
        try:
            import lancedb
        except Exception as exc:
            return [], f"semantic search unavailable: {exc}"
        if not self._db_path.exists():
            return [], "the semantic index has not been built yet"

        try:
            db = lancedb.connect(str(self._db_path))
            if _TABLE_NAME not in _table_names(db):
                return [], "the semantic index is empty"
            table = db.open_table(_TABLE_NAME)
        except Exception as exc:
            return [], f"could not open the index: {exc}"

        try:
            query_vector = self._embed([query])[0]
        except _EmbeddingError as exc:
            return [], str(exc)

        try:
            rows = table.search(query_vector).limit(limit).to_list()
        except Exception as exc:
            return [], f"semantic search failed: {exc}"

        hits: list[SearchHit] = []
        for row in rows:
            distance = float(row.get("_distance", 0.0))
            hits.append(
                SearchHit(
                    rel_path=row["rel_path"],
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    text=row["text"],
                    score=1.0 / (1.0 + distance),
                )
            )
        return hits, None

    def _write_table(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> None:
        import lancedb

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        records = [
            {
                "id": chunk.identifier(),
                "rel_path": chunk.rel_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "text": chunk.text,
                "vector": vector,
            }
            for chunk, vector in zip(chunks, vectors)
        ]
        db = lancedb.connect(str(self._db_path))
        if _TABLE_NAME in _table_names(db):
            db.drop_table(_TABLE_NAME)
        db.create_table(_TABLE_NAME, data=records)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with the configured model, in batches."""
        try:
            import litellm
        except ImportError as exc:
            raise _EmbeddingError("litellm is not installed") from exc

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = texts[start : start + _EMBED_BATCH]
            try:
                response = litellm.embedding(
                    model=self._settings.model,
                    input=batch,
                    api_base=self._settings.api_base or None,
                )
            except Exception as exc:
                raise _EmbeddingError(self._embedding_hint(exc)) from exc
            for item in response.data:
                vectors.append(_embedding_vector(item))
        return vectors

    def _embedding_hint(self, exc: Exception) -> str:
        model = self._settings.model
        if model.startswith("ollama"):
            return (
                f"could not get embeddings from '{model}'. Make sure Ollama is "
                "running and the embedding model is pulled (for example "
                "'ollama pull nomic-embed-text'). Details: " + str(exc)
            )
        return f"could not get embeddings from '{model}': {exc}"


class _EmbeddingError(Exception):
    """Raised when embeddings cannot be produced."""


def _embedding_vector(item: object) -> list[float]:
    """Extract the embedding list from a litellm embedding item."""
    if isinstance(item, dict):
        return list(item["embedding"])
    return list(getattr(item, "embedding"))
