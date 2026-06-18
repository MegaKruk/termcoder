"""Tests for environment loading and platform detection."""

from __future__ import annotations

import os

from termcoder.config_env import load_env_file
from termcoder.platform_info import PlatformInfo, detect_platform


def test_load_env_sets_unset_variables(tmp_path, monkeypatch):
    monkeypatch.delenv("TC_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("TC_TEST_KEY=secret-value\n", encoding="utf-8")

    loaded = load_env_file(tmp_path)

    assert loaded == ["TC_TEST_KEY"]
    assert os.environ["TC_TEST_KEY"] == "secret-value"


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    monkeypatch.setenv("TC_TEST_KEY", "already-here")
    (tmp_path / ".env").write_text("TC_TEST_KEY=from-file\n", encoding="utf-8")

    loaded = load_env_file(tmp_path)

    assert loaded == []
    assert os.environ["TC_TEST_KEY"] == "already-here"


def test_load_env_parses_quotes_comments_and_export(tmp_path, monkeypatch):
    for key in ("TC_A", "TC_B", "TC_C"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / ".env").write_text(
        "# a comment line\n"
        'export TC_A="quoted value"\n'
        "TC_B=plain  # inline comment\n"
        "not a valid line without equals\n"
        "TC_C='single quoted'\n",
        encoding="utf-8",
    )

    load_env_file(tmp_path)

    assert os.environ["TC_A"] == "quoted value"
    assert os.environ["TC_B"] == "plain"
    assert os.environ["TC_C"] == "single quoted"


def test_load_env_missing_file_returns_empty(tmp_path):
    assert load_env_file(tmp_path) == []


def test_detect_platform_returns_consistent_info():
    info = detect_platform()
    assert isinstance(info, PlatformInfo)
    assert info.os_name
    assert info.shell_family() in {"posix", "windows", "powershell"}


def test_shell_family_classifies_windows_powershell():
    info = PlatformInfo(os_name="Windows", is_windows=True, shell="powershell")
    assert info.shell_family() == "powershell"
    assert "shell: powershell" in info.describe()


def test_shell_family_classifies_posix():
    info = PlatformInfo(os_name="Linux", is_windows=False, shell="bash")
    assert info.shell_family() == "posix"


def test_shell_family_classifies_windows_cmd():
    info = PlatformInfo(os_name="Windows", is_windows=True, shell="cmd")
    assert info.shell_family() == "windows"
