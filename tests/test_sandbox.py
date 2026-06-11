"""Tests for command runners and runner selection.

The container runner cannot actually run a container in this test environment,
so its behavior is verified by inspecting the argument vector it builds. The
host runner is exercised directly with a harmless echo.
"""

from __future__ import annotations

import shutil

import pytest

from termcoder.config import SandboxSettings
from termcoder.errors import ConfigError
from termcoder.sandbox.runner import (
    CommandResult,
    ContainerCommandRunner,
    HostCommandRunner,
    build_command_runner,
)


def test_host_runner_runs_echo(tmp_path):
    runner = HostCommandRunner(tmp_path)
    result = runner.run("echo termcoder_ok", timeout=10)
    assert result.ok
    assert "termcoder_ok" in result.stdout
    assert result.backend == "host"


def test_command_result_ok_logic():
    assert CommandResult(0, "", "", "host").ok
    assert not CommandResult(1, "", "", "host").ok
    assert not CommandResult(0, "", "", "host", timed_out=True).ok
    assert not CommandResult(0, "", "", "podman", error="boom").ok


def test_container_argv_has_hardening(tmp_path):
    runner = ContainerCommandRunner("podman", SandboxSettings(), tmp_path)
    argv = runner._build_argv("pytest -q", "termcoder-abcd")

    assert argv[:2] == ["podman", "run"]
    assert "--rm" in argv
    assert "--userns=keep-id" in argv
    assert "--cap-drop" in argv
    assert "ALL" in argv
    assert "no-new-privileges" in argv

    network_index = argv.index("--network")
    assert argv[network_index + 1] == "none"

    assert "--volume" in argv
    assert f"{tmp_path}:/workspace" in argv
    workdir_index = argv.index("--workdir")
    assert argv[workdir_index + 1] == "/workspace"

    assert argv[-4:] == ["python:3.14-slim", "sh", "-c", "pytest -q"]


def test_container_argv_respects_limits(tmp_path):
    settings = SandboxSettings(memory="512m", cpus=1.5, pids_limit=64)
    runner = ContainerCommandRunner("podman", settings, tmp_path)
    argv = runner._build_argv("ls", "name")

    assert argv[argv.index("--memory") + 1] == "512m"
    assert argv[argv.index("--cpus") + 1] == "1.5"
    assert argv[argv.index("--pids-limit") + 1] == "64"


def test_container_argv_network_enabled(tmp_path):
    runner = ContainerCommandRunner("podman", SandboxSettings(network=True), tmp_path)
    argv = runner._build_argv("ls", "name")
    assert "--network" not in argv


def test_docker_argv_omits_userns(tmp_path):
    runner = ContainerCommandRunner("docker", SandboxSettings(), tmp_path)
    argv = runner._build_argv("ls", "name")
    assert "--userns=keep-id" not in argv
    assert runner.backend == "docker"


def test_build_runner_host(tmp_path):
    runner = build_command_runner(SandboxSettings(backend="host"), tmp_path)
    assert isinstance(runner, HostCommandRunner)
    assert not runner.is_sandboxed


def test_build_runner_auto_prefers_podman(tmp_path, monkeypatch):
    monkeypatch.setattr(
        shutil, "which", lambda name: "/usr/bin/podman" if name == "podman" else None
    )
    runner = build_command_runner(SandboxSettings(backend="auto"), tmp_path)
    assert isinstance(runner, ContainerCommandRunner)
    assert runner.backend == "podman"
    assert runner.is_sandboxed


def test_build_runner_auto_falls_back_to_host(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    runner = build_command_runner(SandboxSettings(backend="auto"), tmp_path)
    assert isinstance(runner, HostCommandRunner)


def test_build_runner_explicit_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ConfigError):
        build_command_runner(SandboxSettings(backend="podman"), tmp_path)
