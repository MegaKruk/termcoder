"""Tests for optional semantic search.

The chunker and graceful-degradation paths are tested directly. The full index
round trip runs against a real LanceDB table but a fake embedding function, so
the test needs no Ollama and no network; it is skipped when LanceDB is absent.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from termcoder.config import SemanticSearchSettings
from termcoder.semantic import SemanticIndex, chunk_text, lancedb_available
from termcoder.semantic.chunker import discover_source_files

requires_lancedb = pytest.mark.skipif(
    not lancedb_available(), reason="LanceDB is not installed"
)


def test_chunk_text_produces_line_windows():
    text = "\n".join(f"line {n}" for n in range(1, 201))
    chunks = chunk_text("big.py", text)

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert all(chunk.rel_path == "big.py" for chunk in chunks)
    # Windows overlap, so the second chunk starts before the first one ends.
    assert chunks[1].start_line <= chunks[0].end_line


def test_chunk_text_empty_file():
    assert chunk_text("empty.py", "") == []


def test_chunk_identifier_is_stable():
    chunks = chunk_text("a.py", "def f():\n    pass\n")
    assert chunks[0].identifier() == "a.py:1-2"


def test_discover_source_files_skips_noise(tmp_path):
    (tmp_path / "keep.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hello\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.py").write_text("y = 2\n", encoding="utf-8")

    found = discover_source_files(tmp_path)
    rels = [rel for rel, _ in found]

    assert "keep.py" in rels
    assert "notes.txt" not in rels
    assert all(".git" not in rel for rel in rels)


def test_search_without_index_reports_reason(tmp_path):
    settings = SemanticSearchSettings(enabled=True)
    index = SemanticIndex(tmp_path, tmp_path / "cache" / "semantic", settings)
    if not lancedb_available():
        hits, reason = index.search("anything", 5)
        assert hits == []
        assert reason is not None
        return
    hits, reason = index.search("anything", 5)
    assert hits == []
    assert "not been built" in (reason or "")


def _fake_embedding(model, input, api_base=None):
    data = []
    for text in input:
        digest = hashlib.sha256(text.encode()).digest()
        data.append(SimpleNamespace(embedding=[b / 255.0 for b in digest[:16]]))
    return SimpleNamespace(data=data)


@requires_lancedb
def test_index_build_and_search_round_trip(tmp_path):
    (tmp_path / "auth.py").write_text(
        "def login(user, password):\n    return verify(user, password)\n",
        encoding="utf-8",
    )
    (tmp_path / "math_utils.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8"
    )
    settings = SemanticSearchSettings(enabled=True)
    index = SemanticIndex(tmp_path, tmp_path / "cache" / "semantic", settings)

    with patch("litellm.embedding", side_effect=_fake_embedding):
        status = index.build()
        assert status.ok
        assert status.chunk_count == 2
        assert status.file_count == 2

        hits, reason = index.search("login", limit=2)
        assert reason is None
        assert len(hits) == 2
        assert {hit.rel_path for hit in hits} == {"auth.py", "math_utils.py"}


@requires_lancedb
def test_index_build_reports_embedding_failure(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    settings = SemanticSearchSettings(enabled=True, model="ollama/nomic-embed-text")
    index = SemanticIndex(tmp_path, tmp_path / "cache" / "semantic", settings)

    def boom(model, input, api_base=None):
        raise RuntimeError("connection refused")

    with patch("litellm.embedding", side_effect=boom):
        status = index.build()

    assert not status.ok
    assert "Ollama" in (status.reason or "")
