"""Tests for unified diff generation."""

from __future__ import annotations

from termcoder.approval.diffing import make_unified_diff


def test_identical_content_yields_empty_diff():
    assert make_unified_diff("same\n", "same\n", "a.txt") == ""


def test_diff_marks_added_and_removed_lines():
    diff = make_unified_diff("one\ntwo\n", "one\nTWO\n", "a.txt")
    assert "-two" in diff
    assert "+TWO" in diff
    assert "a/a.txt" in diff
    assert "b/a.txt" in diff


def test_diff_handles_missing_trailing_newline():
    diff = make_unified_diff("a", "b", "f.txt")
    assert diff.endswith("\n")
