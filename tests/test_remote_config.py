"""Tests for the remote and thinking configuration settings.

These check the defaults, that the ``[remote]`` table and ``show_thinking``
flag are parsed from TOML, and that copying a config with a different active
model preserves the new fields.
"""

from __future__ import annotations

from pathlib import Path

from termcoder.config import AppConfig, RemoteSettings, load_config


def _write_config(root: Path, body: str) -> None:
    """Create a .termcoder/config.toml under root with the given body."""
    config_dir = root / ".termcoder"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(body)


def test_remote_settings_defaults():
    settings = RemoteSettings()
    assert settings.enabled is False
    assert settings.host == "0.0.0.0"
    assert settings.port == 8642
    assert settings.token == ""


def test_appconfig_has_remote_and_thinking_defaults(tmp_path):
    config = AppConfig(
        workspace=tmp_path,
        config_dir=tmp_path / ".termcoder",
        active_model="ollama",
        models={},
    )
    assert config.show_thinking is False
    assert config.remote.enabled is False
    assert config.remote.port == 8642


def test_load_config_parses_remote_and_thinking(tmp_path):
    _write_config(
        tmp_path,
        'show_thinking = true\n'
        "[remote]\n"
        "enabled = true\n"
        'host = "127.0.0.1"\n'
        "port = 9000\n"
        'token = "fixed-secret"\n',
    )
    config = load_config(tmp_path)
    assert config.show_thinking is True
    assert config.remote.enabled is True
    assert config.remote.host == "127.0.0.1"
    assert config.remote.port == 9000
    assert config.remote.token == "fixed-secret"


def test_load_config_remote_defaults_when_section_absent(tmp_path):
    _write_config(tmp_path, 'active_model = "ollama"\n')
    config = load_config(tmp_path)
    assert config.remote.enabled is False
    assert config.remote.port == 8642
    assert config.show_thinking is False


def test_with_active_model_preserves_remote_and_thinking(tmp_path):
    _write_config(
        tmp_path,
        "show_thinking = true\n[remote]\nenabled = true\nport = 9100\n",
    )
    config = load_config(tmp_path)
    other = next(iter(config.models))
    copied = config.with_active_model(other)
    assert copied.active_model == other
    assert copied.show_thinking is True
    assert copied.remote.enabled is True
    assert copied.remote.port == 9100
