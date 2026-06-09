"""Tests for workspace path confinement."""

from __future__ import annotations

import os

import pytest

from termcoder.errors import WorkspaceViolationError
from termcoder.workspace.paths import WorkspaceGuard


def test_resolves_path_inside_workspace(tmp_path):
    guard = WorkspaceGuard(tmp_path)
    target = guard.resolve("sub/file.txt")
    assert str(target).startswith(str(tmp_path.resolve()))


def test_allows_not_yet_existing_file(tmp_path):
    guard = WorkspaceGuard(tmp_path)
    # Resolving a path for a file that does not exist yet must succeed,
    # because that is how new files get created.
    target = guard.resolve("brand/new/file.py")
    assert target.name == "file.py"


def test_rejects_parent_directory_escape(tmp_path):
    guard = WorkspaceGuard(tmp_path)
    with pytest.raises(WorkspaceViolationError):
        guard.resolve("../outside.txt")


def test_rejects_absolute_path_outside(tmp_path):
    guard = WorkspaceGuard(tmp_path)
    with pytest.raises(WorkspaceViolationError):
        guard.resolve("/etc/passwd")


def test_rejects_symlink_escape(tmp_path):
    outside_dir = tmp_path.parent / "outside_target"
    outside_dir.mkdir()
    (outside_dir / "secret.txt").write_text("secret", encoding="utf-8")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    link = workspace / "escape"
    try:
        os.symlink(outside_dir, link)
    except (OSError, NotImplementedError):
        pytest.skip("Symlinks are not supported on this platform.")

    guard = WorkspaceGuard(workspace)
    with pytest.raises(WorkspaceViolationError):
        guard.resolve("escape/secret.txt")
