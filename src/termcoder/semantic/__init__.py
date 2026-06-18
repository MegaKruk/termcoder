"""Optional local semantic code search (LanceDB plus local embeddings)."""

from .chunker import CodeChunk, chunk_text, discover_source_files
from .index import IndexStatus, SearchHit, SemanticIndex, lancedb_available
from .tool import SemanticSearchTool

__all__ = [
    "CodeChunk",
    "chunk_text",
    "discover_source_files",
    "SemanticIndex",
    "IndexStatus",
    "SearchHit",
    "lancedb_available",
    "SemanticSearchTool",
]
