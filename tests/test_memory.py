"""Tests for project markdown memory loading."""

from __future__ import annotations

from termcoder.memory import load_project_memory

_FILES = ("TERMCODER.md", "AGENTS.md")


def test_prefers_the_first_listed_file(tmp_path):
    (tmp_path / "TERMCODER.md").write_text("# Conventions\nUse spaces.", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("# Other\nIgnored.", encoding="utf-8")

    memory = load_project_memory(tmp_path, _FILES)

    assert memory is not None
    assert memory.path.name == "TERMCODER.md"
    assert "Use spaces." in memory.text
    assert memory.truncated is False


def test_falls_back_to_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Run pytest before commits.", encoding="utf-8")

    memory = load_project_memory(tmp_path, _FILES)

    assert memory is not None
    assert memory.path.name == "AGENTS.md"
    assert "pytest" in memory.text


def test_missing_files_return_none(tmp_path):
    assert load_project_memory(tmp_path, _FILES) is None


def test_empty_file_returns_none(tmp_path):
    (tmp_path / "TERMCODER.md").write_text("   \n  ", encoding="utf-8")
    assert load_project_memory(tmp_path, _FILES) is None


def test_oversized_memory_is_truncated(tmp_path):
    (tmp_path / "TERMCODER.md").write_text("x" * 20_000, encoding="utf-8")

    memory = load_project_memory(tmp_path, _FILES)

    assert memory is not None
    assert memory.truncated is True
    assert len(memory.text) < 20_000
    assert "truncated" in memory.text
