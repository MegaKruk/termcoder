"""Tests for file snapshots and undo."""

from __future__ import annotations

from termcoder.snapshots.store import NullSnapshotStore, SnapshotStore


def test_capture_then_undo_restores_content(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("original", encoding="utf-8")
    store = SnapshotStore(tmp_path / "snaps")

    store.start_turn("edit a")
    store.capture(target)
    target.write_text("changed", encoding="utf-8")

    assert store.has_undo()
    result = store.undo_last()

    assert target.read_text(encoding="utf-8") == "original"
    assert result is not None
    assert str(target) in result.restored
    assert not store.has_undo()


def test_undo_removes_newly_created_file(tmp_path):
    target = tmp_path / "new.txt"
    store = SnapshotStore(tmp_path / "snaps")

    store.start_turn("create new")
    store.capture(target)
    target.write_text("created", encoding="utf-8")

    result = store.undo_last()

    assert not target.exists()
    assert str(target) in result.deleted


def test_undo_reverts_multiple_files_in_one_turn(tmp_path):
    existing = tmp_path / "a.txt"
    existing.write_text("a0", encoding="utf-8")
    created = tmp_path / "b.txt"
    store = SnapshotStore(tmp_path / "snaps")

    store.start_turn("turn")
    store.capture(existing)
    existing.write_text("a1", encoding="utf-8")
    store.capture(created)
    created.write_text("b1", encoding="utf-8")

    store.undo_last()

    assert existing.read_text(encoding="utf-8") == "a0"
    assert not created.exists()


def test_first_capture_per_file_wins(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("v0", encoding="utf-8")
    store = SnapshotStore(tmp_path / "snaps")

    store.start_turn("turn")
    store.capture(target)
    target.write_text("v1", encoding="utf-8")
    store.capture(target)
    target.write_text("v2", encoding="utf-8")

    store.undo_last()
    assert target.read_text(encoding="utf-8") == "v0"


def test_undo_persists_across_instances(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("orig", encoding="utf-8")
    root = tmp_path / "snaps"

    first = SnapshotStore(root)
    first.start_turn("turn")
    first.capture(target)
    target.write_text("new", encoding="utf-8")

    second = SnapshotStore(root)
    assert second.has_undo()
    second.undo_last()
    assert target.read_text(encoding="utf-8") == "orig"


def test_only_latest_group_is_undone(tmp_path):
    first_file = tmp_path / "first.txt"
    first_file.write_text("first-0", encoding="utf-8")
    second_file = tmp_path / "second.txt"
    second_file.write_text("second-0", encoding="utf-8")
    store = SnapshotStore(tmp_path / "snaps")

    store.start_turn("turn one")
    store.capture(first_file)
    first_file.write_text("first-1", encoding="utf-8")

    store.start_turn("turn two")
    store.capture(second_file)
    second_file.write_text("second-1", encoding="utf-8")

    store.undo_last()

    assert second_file.read_text(encoding="utf-8") == "second-0"
    assert first_file.read_text(encoding="utf-8") == "first-1"
    assert store.has_undo()


def test_null_store_is_a_noop(tmp_path):
    store = NullSnapshotStore()
    store.start_turn("x")
    store.capture(tmp_path / "a.txt")
    assert store.has_undo() is False
    assert store.undo_last() is None
