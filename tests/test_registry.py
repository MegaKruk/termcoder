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


def test_load_config_parses_audit_keys(tmp_path):
    from termcoder.config import load_config

    config_dir = tmp_path / ".termcoder"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        """
show_usage = false

[sandbox]
read_only = true

[context]
summary_model = "ollama"

[models.gpt]
model = "gpt-4o"
cache_prompts = false
""",
        encoding="utf-8",
    )

    config = load_config(tmp_path)

    assert config.show_usage is False
    assert config.sandbox.read_only is True
    assert config.context.summary_model == "ollama"
    assert config.models["gpt"].cache_prompts is False
    # Untouched models keep the default.
    assert config.models["ollama"].cache_prompts is True


def test_web_search_tool_added_when_enabled(tmp_path):
    from dataclasses import replace
    from termcoder.config import WebSearchSettings

    config = _config(tmp_path)
    config = replace(config, web_search=WebSearchSettings(enabled=True))
    registry = build_default_registry(config)
    assert "web_search" in registry.names()


def test_web_search_tool_absent_by_default(tmp_path):
    registry = build_default_registry(_config(tmp_path))
    assert "web_search" not in registry.names()


def test_read_skill_tool_added_with_skills(tmp_path):
    from termcoder.skills import SkillRegistry

    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\n\nBody.", encoding="utf-8"
    )
    skills = SkillRegistry.from_directories([tmp_path / "skills"])
    registry = build_default_registry(_config(tmp_path), skills=skills)
    assert "read_skill" in registry.names()


def test_read_skill_tool_absent_without_skills(tmp_path):
    from termcoder.skills import SkillRegistry

    registry = build_default_registry(_config(tmp_path), skills=SkillRegistry())
    assert "read_skill" not in registry.names()
