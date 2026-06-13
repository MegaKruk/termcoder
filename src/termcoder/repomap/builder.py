"""Build a token-budgeted map of the most important symbols in a repository.

The approach follows Aider's repo map: extract definition and reference tags
per file with tree-sitter, build a directed graph in which a file that
references a symbol points at the files defining it, rank files with PageRank,
distribute each file's rank over its definitions in proportion to how often
they are referenced, then render as many of the top definitions as fit the
token budget (found by binary search).

Token-economy decision: the map is built once and treated as a stable block of
the system prompt for the whole session, so the prompt prefix stays cacheable.
It is refreshed only on demand. Identifier weighting mirrors Aider's
multipliers in spirit: private names (leading underscore) are downweighted and
long, well-formed names are boosted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..context.tokens import TokenCounter
from ..workspace.ignore import IGNORED_DIRS
from .cache import TagCache
from .tags import LANGUAGES, Tag, extract_tags, tree_sitter_available

_MAX_FILE_BYTES = 512_000
_MAX_FILES = 5000
_PRIVATE_MULTIPLIER = 0.1
_WELL_NAMED_MULTIPLIER = 10.0
_RANK_SMOOTHING = 0.01


@dataclass
class RepoMapResult:
    """The outcome of one map build, including why it may be empty."""

    text: str | None
    file_count: int = 0
    tag_count: int = 0
    tokens: int = 0
    reason: str | None = None
    scanned_files: int = 0


@dataclass
class _Definition:
    tag: Tag
    rank: float = 0.0
    in_weight: float = field(default=0.0, repr=False)


def _ident_weight(name: str, count: int) -> float:
    """Weight an identifier's references, like Aider's multipliers."""
    weight = float(count)
    if name.startswith("_"):
        return weight * _PRIVATE_MULTIPLIER
    mixed_case = name != name.lower() and name != name.upper()
    if len(name) >= 8 and ("_" in name or mixed_case):
        weight *= _WELL_NAMED_MULTIPLIER
    return weight


def _pagerank(
    nodes: list[str],
    edges: dict[tuple[str, str], float],
    damping: float = 0.85,
    iterations: int = 100,
    tolerance: float = 1.0e-6,
) -> dict[str, float]:
    """Weighted PageRank by power iteration, deterministic and dependency-free.

    networkx 3.x delegates pagerank to scipy, which would pull numpy into a
    terminal tool for one small algorithm; thirty lines of power iteration
    keep the dependency tree lean. Dangling mass (nodes with no outgoing
    edges) is redistributed uniformly, matching the standard formulation.
    """
    count = len(nodes)
    if count == 0:
        return {}
    out_weight = {node: 0.0 for node in nodes}
    outgoing: dict[str, list[tuple[str, float]]] = {node: [] for node in nodes}
    for (source, target), weight in edges.items():
        out_weight[source] += weight
        outgoing[source].append((target, weight))

    rank = {node: 1.0 / count for node in nodes}
    base = (1.0 - damping) / count
    for _ in range(iterations):
        next_rank = {node: base for node in nodes}
        dangling = 0.0
        for node in nodes:
            value = rank[node]
            total_out = out_weight[node]
            if total_out > 0.0:
                share = damping * value / total_out
                for target, weight in outgoing[node]:
                    next_rank[target] += share * weight
            else:
                dangling += value
        if dangling:
            spread = damping * dangling / count
            for node in nodes:
                next_rank[node] += spread
        delta = sum(abs(next_rank[node] - rank[node]) for node in nodes)
        rank = next_rank
        if delta < tolerance:
            break
    return rank


class RepoMapBuilder:
    """Builds the repository map for one workspace."""

    def __init__(
        self,
        root: Path,
        cache_path: Path,
        budget_tokens: int,
        token_counter: TokenCounter | None = None,
    ):
        self._root = Path(root)
        self._cache_path = cache_path
        self._budget = max(0, budget_tokens)
        self._tokens = token_counter or TokenCounter()

    def build(self) -> RepoMapResult:
        """Scan the workspace and produce a budget-fitted map."""
        if self._budget <= 0:
            return RepoMapResult(text=None, reason="map token budget is zero")
        if not tree_sitter_available():
            return RepoMapResult(
                text=None,
                reason=(
                    "tree-sitter is not installed; install termcoder's "
                    "dependencies to enable the repository map"
                ),
            )
        files = self._discover()
        if not files:
            return RepoMapResult(
                text=None, reason="no supported source files in the workspace"
            )
        tags = self._extract_all(files)
        definitions = self._rank(tags)
        if not definitions:
            return RepoMapResult(
                text=None,
                reason="no definitions found in the supported source files",
                scanned_files=len(files),
            )
        text, used, tokens = self._fit(definitions)
        if text is None:
            return RepoMapResult(
                text=None,
                reason="nothing fits the map token budget; raise repomap.tokens",
                scanned_files=len(files),
            )
        shown_files = len({definition.tag.rel_path for definition in used})
        return RepoMapResult(
            text=text,
            file_count=shown_files,
            tag_count=len(used),
            tokens=tokens,
            scanned_files=len(files),
        )

    def _discover(self) -> list[tuple[str, Path]]:
        """Find supported source files, as sorted (relative path, path) pairs."""
        import os

        found: list[tuple[str, Path]] = []
        for current, dirs, names in os.walk(self._root):
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
                found.append((path.relative_to(self._root).as_posix(), path))
                if len(found) >= _MAX_FILES:
                    return found
        return found

    def _extract_all(self, files: list[tuple[str, Path]]) -> list[Tag]:
        """Extract tags for every file, using the mtime cache where possible."""
        cache = TagCache(self._cache_path)
        tags: list[Tag] = []
        for rel, path in files:
            try:
                stat = path.stat()
                cached = cache.get(rel, stat.st_mtime, stat.st_size)
                if cached is None:
                    cached = extract_tags(path.read_bytes(), path, rel)
                    cache.put(rel, stat.st_mtime, stat.st_size, cached)
            except OSError:
                continue
            tags.extend(cached)
        cache.prune({rel for rel, _ in files})
        cache.save()
        return tags

    def _rank(self, tags: list[Tag]) -> list[_Definition]:
        """Rank definitions by PageRank over the file reference graph.

        Files are nodes; a file referencing a symbol points at every file that
        defines it (self-references are skipped so the map favors API surface
        over internal churn). Each file's rank is then distributed over its
        definitions in proportion to how heavily each one is referenced, with
        light smoothing so unreferenced definitions still get a small share.
        """
        definitions: dict[tuple[str, str], _Definition] = {}
        definers: dict[str, set[str]] = {}
        references: dict[tuple[str, str], int] = {}
        nodes: list[str] = []
        seen_nodes: set[str] = set()
        edges: dict[tuple[str, str], float] = {}

        for tag in tags:
            if tag.rel_path not in seen_nodes:
                seen_nodes.add(tag.rel_path)
                nodes.append(tag.rel_path)
            if tag.kind == "def":
                key = (tag.rel_path, tag.name)
                if key not in definitions:
                    definitions[key] = _Definition(tag=tag)
                definers.setdefault(tag.name, set()).add(tag.rel_path)
            else:
                references[(tag.rel_path, tag.name)] = (
                    references.get((tag.rel_path, tag.name), 0) + 1
                )

        for (ref_file, name), count in references.items():
            weight = _ident_weight(name, count)
            for def_file in sorted(definers.get(name, ())):
                if def_file == ref_file:
                    continue
                definitions[(def_file, name)].in_weight += weight
                edge = (ref_file, def_file)
                edges[edge] = edges.get(edge, 0.0) + weight

        if not nodes:
            return []
        ranks = _pagerank(nodes, edges)

        per_file: dict[str, list[_Definition]] = {}
        for definition in definitions.values():
            per_file.setdefault(definition.tag.rel_path, []).append(definition)
        for rel, defs in per_file.items():
            file_rank = ranks.get(rel, 0.0)
            total = sum(item.in_weight for item in defs)
            smoothing = _RANK_SMOOTHING
            denominator = total + smoothing * len(defs)
            for item in defs:
                item.rank = file_rank * (item.in_weight + smoothing) / denominator

        return sorted(
            definitions.values(),
            key=lambda item: (-item.rank, item.tag.rel_path, item.tag.line),
        )

    def _fit(
        self, definitions: list[_Definition]
    ) -> tuple[str | None, list[_Definition], int]:
        """Binary search the largest prefix of definitions within budget."""
        low, high = 0, len(definitions)
        best: tuple[str, list[_Definition], int] | None = None
        while low < high:
            middle = (low + high + 1) // 2
            subset = definitions[:middle]
            text = self._render(subset)
            tokens = self._tokens.count_text(text)
            if tokens <= self._budget:
                best = (text, subset, tokens)
                low = middle
            else:
                high = middle - 1
        if best is None:
            return None, [], 0
        return best

    @staticmethod
    def _render(definitions: list[_Definition]) -> str:
        """Render definitions grouped by file, in rank order."""
        order: list[str] = []
        grouped: dict[str, list[_Definition]] = {}
        for definition in definitions:
            rel = definition.tag.rel_path
            if rel not in grouped:
                grouped[rel] = []
                order.append(rel)
            grouped[rel].append(definition)
        lines: list[str] = []
        for rel in order:
            lines.append(f"{rel}:")
            for definition in sorted(grouped[rel], key=lambda item: item.tag.line):
                tag = definition.tag
                lines.append(f"  {tag.line}| {tag.text or tag.name}")
        return "\n".join(lines)
