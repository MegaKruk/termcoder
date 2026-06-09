"""Tests for model configuration and the default tool registry."""

from __future__ import annotations

from pathlib import Path

from termcoder.config import AppConfig, ModelConfig, default_models
from termcoder.tools import build_default_registry


def _config(tmp_path, allow_run_command=True):
    return AppConfig(
        workspace=tmp_path,
        config_dir=tmp_path / ".termcoder",
        active_model="ollama",
        models=default_models(),
        allow_run_command=allow_run_command,
    )


def test_completion_kwargs_include_core_fields():
    model = ModelConfig(name="ollama", model="ollama_chat/llama3.1", api_base="http://x")
    kwargs = model.to_completion_kwargs()
    assert kwargs["model"] == "ollama_chat/llama3.1"
    assert "temperature" in kwargs
    assert kwargs["api_base"] == "http://x"
    # No api_key_env was set, so no key should be present.
    assert "api_key" not in kwargs


def test_completion_kwargs_include_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "secret-value")
    model = ModelConfig(name="cloud", model="gpt-4o", api_key_env="MY_TEST_KEY")
    kwargs = model.to_completion_kwargs()
    assert kwargs["api_key"] == "secret-value"


def test_default_registry_contains_expected_tools(tmp_path):
    registry = build_default_registry(_config(tmp_path))
    names = set(registry.names())
    assert {"read_file", "list_directory", "search_text", "write_file", "edit_file"} <= names
    assert "run_command" in names


def test_run_command_can_be_disabled(tmp_path):
    registry = build_default_registry(_config(tmp_path, allow_run_command=False))
    assert "run_command" not in registry.names()


def test_schemas_are_function_tools(tmp_path):
    registry = build_default_registry(_config(tmp_path))
    schemas = registry.schemas()
    assert all(item["type"] == "function" for item in schemas)
    assert all("name" in item["function"] for item in schemas)
