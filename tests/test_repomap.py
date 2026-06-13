"""Tests for the repository map.

A small synthetic project provides a known reference structure: one module
defines well-named helpers that two other modules call, so it must rank first.
The tests stay offline; tree-sitter grammars ship in the installed wheel so no
network is involved.
"""

from __future__ import annotations

import json

import pytest

from termcoder.repomap import builder as builder_module
from termcoder.repomap.builder import RepoMapBuilder
from termcoder.repomap.cache import TagCache
from termcoder.repomap.tags import Tag, extract_tags, tree_sitter_available

pytestmark = pytest.mark.skipif(
    not tree_sitter_available(), reason="tree-sitter is not installed"
)


def _write_project(root):
    (root / "core.py").write_text(
        "def shared_helper_function(value):\n"
        "    return value * 2\n"
        "\n"
        "def another_named_helper(value):\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    (root / "alpha.py").write_text(
        "from core import shared_helper_function\n"
        "\n"
        "def run_alpha():\n"
        "    return shared_helper_function(1)\n",
        encoding="utf-8",
    )
    (root / "beta.py").write_text(
        "from core import shared_helper_function, another_named_helper\n"
        "\n"
        "def run_beta():\n"
        "    return shared_helper_function(2) + another_named_helper(3)\n",
        encoding="utf-8",
    )
    (root / "notes.txt").write_text("not source code\n", encoding="utf-8")


def _build(root, tmp_path, budget=512):
    cache_path = tmp_path / "cache" / "repomap.json"
    return RepoMapBuilder(root, cache_path, budget_tokens=budget).build()


def test_map_ranks_the_most_referenced_file_first(tmp_path):
    _write_project(tmp_path)
    result = _build(tmp_path, tmp_path)

    assert result.text is not None
    assert "shared_helper_function" in result.text
    assert result.tag_count > 0
    assert result.scanned_files == 3
    # core.py is referenced by both other modules, so it leads the map.
    assert result.text.splitlines()[0] == "core.py:"


def test_map_respects_the_token_budget(tmp_path):
    _write_project(tmp_path)
    large = _build(tmp_path, tmp_path, budget=512)
    small = _build(tmp_path, tmp_path, budget=20)

    assert large.text is not None and small.text is not None
    assert small.tokens <= 20
    assert small.tag_count < large.tag_count


def test_zero_budget_yields_a_reason(tmp_path):
    _write_project(tmp_path)
    result = _build(tmp_path, tmp_path, budget=0)
    assert result.text is None
    assert "budget" in (result.reason or "")


def test_no_source_files_yields_a_reason(tmp_path):
    (tmp_path / "readme.txt").write_text("hello\n", encoding="utf-8")
    result = _build(tmp_path, tmp_path)
    assert result.text is None
    assert "no supported source files" in (result.reason or "")


def test_missing_tree_sitter_degrades_gracefully(tmp_path, monkeypatch):
    _write_project(tmp_path)
    monkeypatch.setattr(builder_module, "tree_sitter_available", lambda: False)
    result = _build(tmp_path, tmp_path)
    assert result.text is None
    assert "tree-sitter" in (result.reason or "")


def test_map_builds_are_deterministic(tmp_path):
    _write_project(tmp_path)
    first = _build(tmp_path, tmp_path / "a")
    second = _build(tmp_path, tmp_path / "b")
    assert first.text == second.text


def test_cache_avoids_reparsing_unchanged_files(tmp_path, monkeypatch):
    _write_project(tmp_path)
    calls = {"count": 0}
    real_extract = builder_module.extract_tags

    def counting_extract(source, path, rel):
        calls["count"] += 1
        return real_extract(source, path, rel)

    monkeypatch.setattr(builder_module, "extract_tags", counting_extract)

    _build(tmp_path, tmp_path)
    assert calls["count"] == 3

    _build(tmp_path, tmp_path)
    assert calls["count"] == 3  # all served from the cache

    # Changing one file re-extracts only that file.
    (tmp_path / "alpha.py").write_text(
        "def run_alpha():\n    return 1\n", encoding="utf-8"
    )
    _build(tmp_path, tmp_path)
    assert calls["count"] == 4


def test_cache_prunes_deleted_files(tmp_path):
    _write_project(tmp_path)
    cache_path = tmp_path / "cache" / "repomap.json"
    RepoMapBuilder(tmp_path, cache_path, budget_tokens=512).build()
    (tmp_path / "beta.py").unlink()
    RepoMapBuilder(tmp_path, cache_path, budget_tokens=512).build()

    data = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "beta.py" not in data["files"]
    assert "core.py" in data["files"]


def test_tag_cache_invalidates_on_mtime_change(tmp_path):
    cache = TagCache(tmp_path / "cache.json")
    tag = Tag(rel_path="a.py", name="thing", kind="def", line=1, text="def thing():")
    cache.put("a.py", 100.0, 10, [tag])
    cache.save()

    reloaded = TagCache(tmp_path / "cache.json")
    assert reloaded.get("a.py", 100.0, 10) == [tag]
    assert reloaded.get("a.py", 200.0, 10) is None
    assert reloaded.get("a.py", 100.0, 11) is None


def test_extract_tags_handles_unsupported_and_broken_input(tmp_path):
    path = tmp_path / "data.xyz"
    assert extract_tags(b"whatever", path, "data.xyz") == []
    broken = tmp_path / "broken.py"
    assert isinstance(extract_tags(b"\xff\xfe\x00", broken, "broken.py"), list)
