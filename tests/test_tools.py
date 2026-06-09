"""Tests for the built-in tools.

These use the auto-approver so mutating tools apply without prompting, and run
entirely on the local filesystem with no network or model access.
"""

from __future__ import annotations

import pytest

from termcoder.approval.auto import AutoApprover, RejectingApprover
from termcoder.tools.base import ToolContext
from termcoder.tools.edit_file import EditFileArgs, EditFileTool
from termcoder.tools.list_directory import ListDirectoryArgs, ListDirectoryTool
from termcoder.tools.read_file import ReadFileArgs, ReadFileTool
from termcoder.tools.run_command import RunCommandArgs, RunCommandTool
from termcoder.tools.search_text import SearchTextArgs, SearchTextTool
from termcoder.tools.write_file import WriteFileArgs, WriteFileTool
from termcoder.workspace.paths import WorkspaceGuard


def make_context(tmp_path, approver=None):
    return ToolContext(
        workspace=WorkspaceGuard(tmp_path),
        approver=approver or AutoApprover(),
    )


def test_read_file_returns_contents(tmp_path):
    (tmp_path / "hello.txt").write_text("line one\nline two\n", encoding="utf-8")
    tool = ReadFileTool()
    result = tool.execute(ReadFileArgs(path="hello.txt"), make_context(tmp_path))
    assert result.ok
    assert "line one" in result.content
    assert "line two" in result.content


def test_read_file_line_range(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\nd\n", encoding="utf-8")
    tool = ReadFileTool()
    result = tool.execute(
        ReadFileArgs(path="f.txt", start_line=2, end_line=3), make_context(tmp_path)
    )
    assert result.content == "b\nc"


def test_read_missing_file_is_not_ok(tmp_path):
    tool = ReadFileTool()
    result = tool.execute(ReadFileArgs(path="nope.txt"), make_context(tmp_path))
    assert not result.ok


def test_list_directory_lists_entries(tmp_path):
    (tmp_path / "a.py").write_text("x", encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    tool = ListDirectoryTool()
    result = tool.execute(ListDirectoryArgs(path="."), make_context(tmp_path))
    assert "a.py" in result.content
    assert "pkg/" in result.content


def test_search_text_finds_matches(tmp_path, monkeypatch):
    # Force the built-in Python scanner so the test does not depend on ripgrep.
    import termcoder.tools.search_text as search_module

    monkeypatch.setattr(search_module.shutil, "which", lambda name: None)
    (tmp_path / "code.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    tool = SearchTextTool()
    result = tool.execute(SearchTextArgs(pattern="alpha"), make_context(tmp_path))
    assert result.ok
    assert "code.py" in result.content
    assert "alpha" in result.content


def test_write_file_creates_file(tmp_path):
    tool = WriteFileTool()
    result = tool.execute(
        WriteFileArgs(path="out/new.txt", content="hello\n"), make_context(tmp_path)
    )
    assert result.ok
    assert (tmp_path / "out" / "new.txt").read_text(encoding="utf-8") == "hello\n"


def test_write_file_rejected_does_not_write(tmp_path):
    tool = WriteFileTool()
    context = make_context(tmp_path, approver=RejectingApprover("no thanks"))
    result = tool.execute(WriteFileArgs(path="x.txt", content="data"), context)
    assert not result.ok
    assert not (tmp_path / "x.txt").exists()


def test_edit_file_replaces_text(tmp_path):
    (tmp_path / "f.py").write_text("value = 1\n", encoding="utf-8")
    tool = EditFileTool()
    result = tool.execute(
        EditFileArgs(path="f.py", old_string="value = 1", new_string="value = 2"),
        make_context(tmp_path),
    )
    assert result.ok
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "value = 2\n"


def test_edit_file_reports_no_match(tmp_path):
    (tmp_path / "f.py").write_text("value = 1\n", encoding="utf-8")
    tool = EditFileTool()
    result = tool.execute(
        EditFileArgs(path="f.py", old_string="missing", new_string="x"),
        make_context(tmp_path),
    )
    assert not result.ok


def test_edit_file_ambiguous_match_is_blocked(tmp_path):
    (tmp_path / "f.py").write_text("x\nx\n", encoding="utf-8")
    tool = EditFileTool()
    result = tool.execute(
        EditFileArgs(path="f.py", old_string="x", new_string="y"),
        make_context(tmp_path),
    )
    assert not result.ok
    # The file must be unchanged when the match is ambiguous.
    assert (tmp_path / "f.py").read_text(encoding="utf-8") == "x\nx\n"


def test_run_command_captures_output(tmp_path):
    tool = RunCommandTool()
    result = tool.execute(
        RunCommandArgs(command="echo termcoder_ok"), make_context(tmp_path)
    )
    assert result.ok
    assert "termcoder_ok" in result.content
