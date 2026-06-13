"""Symbol tag extraction with tree-sitter.

A tag records where a symbol is defined or referenced. Definitions carry the
source line text so the repository map can show signatures without re-reading
files. Languages are supported by dropping a ``<query>-tags.scm`` file into
the ``queries`` directory and mapping file extensions to it below; adding a
language never requires code changes elsewhere.

The query API differs across py-tree-sitter versions, so a small shim prefers
the modern Query/QueryCursor interface and falls back to the older
``language.query`` form.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_QUERY_DIR = Path(__file__).parent / "queries"

# Maximum length of the stored definition line, to keep the map and the tag
# cache compact.
_LINE_TEXT_LIMIT = 120

# Maps a file extension to (tree-sitter language name, query file name).
LANGUAGES: dict[str, tuple[str, str]] = {
    ".py": ("python", "python-tags.scm"),
    ".js": ("javascript", "javascript-tags.scm"),
    ".mjs": ("javascript", "javascript-tags.scm"),
    ".cjs": ("javascript", "javascript-tags.scm"),
    ".jsx": ("javascript", "javascript-tags.scm"),
    ".ts": ("typescript", "typescript-tags.scm"),
    ".tsx": ("tsx", "typescript-tags.scm"),
    ".go": ("go", "go-tags.scm"),
    ".rs": ("rust", "rust-tags.scm"),
    ".java": ("java", "java-tags.scm"),
    ".c": ("c", "c-tags.scm"),
    ".h": ("cpp", "cpp-tags.scm"),
    ".cpp": ("cpp", "cpp-tags.scm"),
    ".cc": ("cpp", "cpp-tags.scm"),
    ".cxx": ("cpp", "cpp-tags.scm"),
    ".hpp": ("cpp", "cpp-tags.scm"),
    ".hh": ("cpp", "cpp-tags.scm"),
}


@dataclass(frozen=True)
class Tag:
    """One symbol definition or reference in a source file."""

    rel_path: str
    name: str
    kind: str  # "def" or "ref"
    line: int  # 1-based line number
    text: str = ""  # the stripped source line, definitions only


def detect_language(path: Path) -> str | None:
    """Return the tree-sitter language for a path, or None if unsupported."""
    entry = LANGUAGES.get(path.suffix.lower())
    return entry[0] if entry else None


def tree_sitter_available() -> bool:
    """True when the tree-sitter runtime and grammar pack can be imported."""
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_language_pack  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache(maxsize=None)
def _query_source(query_file: str) -> str:
    return (_QUERY_DIR / query_file).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _language_and_query(language: str, query_file: str):
    """Load and cache the parser, language object and compiled query."""
    from tree_sitter_language_pack import get_language, get_parser

    lang = get_language(language)
    parser = get_parser(language)
    source = _query_source(query_file)
    try:
        from tree_sitter import Query

        query = Query(lang, source)
    except ImportError:
        query = lang.query(source)
    return parser, query


def _run_captures(query, root) -> list[tuple[object, str]]:
    """Run a query and normalize captures to (node, capture_name) pairs."""
    try:
        from tree_sitter import QueryCursor

        captures = QueryCursor(query).captures(root)
    except ImportError:
        captures = query.captures(root)
    if isinstance(captures, dict):
        return [(node, name) for name, nodes in captures.items() for node in nodes]
    return [(node, name) for node, name in captures]


def extract_tags(source: bytes, path: Path, rel_path: str) -> list[Tag]:
    """Extract definition and reference tags from one file's source.

    Returns an empty list for unsupported languages or unparseable content;
    extraction must never break the map build.
    """
    entry = LANGUAGES.get(path.suffix.lower())
    if entry is None:
        return []
    language, query_file = entry
    try:
        parser, query = _language_and_query(language, query_file)
        tree = parser.parse(source)
        captures = _run_captures(query, tree.root_node)
    except Exception:
        return []

    lines = source.decode("utf-8", errors="replace").splitlines()
    tags: list[Tag] = []
    for node, capture in captures:
        if capture.startswith("name.definition."):
            kind = "def"
        elif capture.startswith("name.reference."):
            kind = "ref"
        else:
            continue
        name = (node.text or b"").decode("utf-8", errors="replace")
        if not name:
            continue
        line = node.start_point[0] + 1
        text = ""
        if kind == "def" and 0 < line <= len(lines):
            text = lines[line - 1].strip()[:_LINE_TEXT_LIMIT]
        tags.append(Tag(rel_path=rel_path, name=name, kind=kind, line=line, text=text))
    return tags
