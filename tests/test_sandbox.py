"""Tests for command runners and runner selection.

The container runner cannot actually run a container in this test environment,
so its behavior is verified by inspecting the argument vector it builds. The
host runner is exercised directly with a harmless echo.
"""

from __future__ import annotations

import shutil
import sys
import time

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


def test_host_runner_timeout_kills_process_group(tmp_path):
    runner = HostCommandRunner(tmp_path)
    # Write the sleep as a script file and run it, so the command has no nested
    # quotes to parse. Inline 'python -c "..."' is quoted differently by sh,
    # cmd.exe and PowerShell; a bare 'python script.py' parses the same in all
    # of them, so this exercises the timeout-kill path portably.
    script = tmp_path / "_sleep.py"
    script.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    sleep_command = f'"{sys.executable}" "{script}"'
    started = time.monotonic()
    result = runner.run(sleep_command, timeout=1)
    elapsed = time.monotonic() - started

    assert result.timed_out
    assert result.returncode == 124
    assert not result.ok
    assert elapsed < 10


def test_container_argv_read_only_rootfs(tmp_path):
    runner = ContainerCommandRunner(
        "podman", SandboxSettings(read_only=True), tmp_path
    )
    argv = runner._build_argv("ls", "name")
    assert "--read-only" in argv
    assert "--tmpfs" in argv
    assert argv[argv.index("--tmpfs") + 1] == "/tmp"
    assert "read-only rootfs" in runner.describe()


def test_container_argv_default_is_writable_rootfs(tmp_path):
    runner = ContainerCommandRunner("podman", SandboxSettings(), tmp_path)
    argv = runner._build_argv("ls", "name")
    assert "--read-only" not in argv


def test_popen_invocation_posix_uses_shell(monkeypatch):
    from termcoder.sandbox import runner

    monkeypatch.setattr(runner.os, "name", "posix")
    args, use_shell = runner._popen_invocation("ls -la")
    assert args == "ls -la"
    assert use_shell is True


def test_popen_invocation_windows_powershell_wraps_command(monkeypatch):
    from termcoder.sandbox import runner

    monkeypatch.setattr(runner.os, "name", "nt")
    monkeypatch.setenv("PSModulePath", "C:\\dummy")
    monkeypatch.setattr(
        runner.shutil, "which", lambda n: "powershell" if n == "powershell" else None
    )
    # A bare command runs as-is through powershell.
    args, use_shell = runner._popen_invocation("Get-ChildItem")
    assert use_shell is False
    assert args[0] == "powershell"
    assert "-Command" in args
    assert args[-1] == "Get-ChildItem"
    # A command starting with a quoted path gets the call operator so it runs.
    args, _ = runner._popen_invocation('"C:\\tools\\python.exe" script.py')
    assert args[-1].startswith("& ")
    # A PowerShell language construct is left untouched (no call operator).
    args, _ = runner._popen_invocation("$env:FOO = 'bar'")
    assert args[-1] == "$env:FOO = 'bar'"


def test_popen_invocation_windows_cmd_falls_back_to_shell(monkeypatch):
    from termcoder.sandbox import runner

    monkeypatch.setattr(runner.os, "name", "nt")
    monkeypatch.delenv("PSModulePath", raising=False)
    args, use_shell = runner._popen_invocation("dir")
    assert args == "dir"
    assert use_shell is True
