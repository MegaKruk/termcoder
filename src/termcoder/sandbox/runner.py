"""Command execution backends.

Phase 1 ran shell commands directly on the host. Phase 2 introduces a sandbox:
the agent process still runs on the host, but commands it wants to run are
executed inside an ephemeral, rootless container (Podman by default, Docker as
a fallback). The container drops all capabilities, disables new privileges, has
no network by default, and is resource limited.

The :class:`CommandRunner` protocol keeps the run_command tool independent of
which backend is in use. A later phase can add a microVM backend by
implementing the same protocol, with no change to the tool.
"""

from __future__ import annotations

import os
import secrets
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import SandboxSettings
from ..errors import ConfigError

# Backstop added to the container wall-clock timeout for the host-side process,
# so the host call does not give up before the container's own timeout fires.
_HOST_TIMEOUT_SLACK_SECONDS = 15


def _process_creation_kwargs() -> dict:
    """Return Popen kwargs that isolate the child so a timeout can kill its tree.

    On POSIX, a new session makes the child its own process group leader, so
    the whole group can be signalled. On Windows, a new process group is the
    closest standard-library equivalent.
    """
    if os.name == "posix":
        return {"start_new_session": True}
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return {"creationflags": creationflags}



@dataclass
class CommandResult:
    """The outcome of running a single command."""

    returncode: int
    stdout: str
    stderr: str
    backend: str
    timed_out: bool = False
    oom_killed: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the command ran to completion successfully."""
        return self.returncode == 0 and not self.timed_out and self.error is None


class CommandRunner(Protocol):
    """Anything that can run a shell command and report the result."""

    @property
    def backend(self) -> str:
        """Short backend identifier, for example 'host' or 'podman'."""
        ...

    @property
    def is_sandboxed(self) -> bool:
        """True when commands run in an isolated environment."""
        ...

    def describe(self) -> str:
        """A short sentence shown to the user before a command is approved."""
        ...

    def run(self, command: str, timeout: int) -> CommandResult:
        """Run a command and return its result."""
        ...


class HostCommandRunner:
    """Run commands directly on the host, with the workspace as the directory.

    This has no isolation, so the run_command tool always requires approval and
    warns the user. It is the fallback when no container engine is available.
    """

    backend = "host"
    is_sandboxed = False

    def __init__(self, workspace_root: Path):
        self._root = Path(workspace_root)

    def describe(self) -> str:
        return "This command runs directly on your host with no sandbox."

    def run(self, command: str, timeout: int) -> CommandResult:
        """Run the command, killing the whole process tree on timeout.

        ``subprocess.run(timeout=...)`` only kills the immediate child; any
        background children it spawned would survive. On POSIX the child starts
        a new session so the process group can be signalled; on Windows a job
        is approximated by terminating the child, which is the best the standard
        library offers without extra dependencies.
        """
        creation = _process_creation_kwargs()
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(self._root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **creation,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_tree(process)
            return CommandResult(
                returncode=124,
                stdout="",
                stderr="",
                backend=self.backend,
                timed_out=True,
            )
        return CommandResult(
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            backend=self.backend,
        )

    @staticmethod
    def _kill_tree(process: subprocess.Popen) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                # Windows: terminate the child; CREATE_NEW_PROCESS_GROUP limits
                # stray children but there is no portable group kill.
                process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
        try:
            process.communicate(timeout=_HOST_TIMEOUT_SLACK_SECONDS)
        except (subprocess.TimeoutExpired, ValueError):
            pass


class ContainerCommandRunner:
    """Run commands inside an ephemeral, rootless container.

    The workspace is bind-mounted read-write so edits and build artifacts are
    visible on the host. For Podman, ``--userns=keep-id`` maps the container
    user to the host user, so files created in the container are owned by you.
    """

    is_sandboxed = True

    def __init__(self, engine: str, settings: SandboxSettings, workspace_root: Path):
        self._engine = engine
        self._settings = settings
        self._root = Path(workspace_root)

    @property
    def backend(self) -> str:
        return self._engine

    def describe(self) -> str:
        network = "network enabled" if self._settings.network else "no network"
        rootfs = ", read-only rootfs" if self._settings.read_only else ""
        return (
            f"This command runs in a rootless {self._engine} container "
            f"(image {self._settings.image}, {network}, workspace mounted "
            f"read-write{rootfs})."
        )

    def run(self, command: str, timeout: int) -> CommandResult:
        name = f"termcoder-{secrets.token_hex(4)}"
        argv = self._build_argv(command, name)
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout + _HOST_TIMEOUT_SLACK_SECONDS,
            )
        except FileNotFoundError:
            return CommandResult(
                returncode=127,
                stdout="",
                stderr="",
                backend=self._engine,
                error=f"'{self._engine}' was not found on PATH.",
            )
        except subprocess.TimeoutExpired:
            self._force_remove(name)
            return CommandResult(
                returncode=124,
                stdout="",
                stderr="",
                backend=self._engine,
                timed_out=True,
            )
        # A 137 exit code is SIGKILL, which for a memory-limited container is
        # most often the out-of-memory killer.
        oom = completed.returncode == 137
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            backend=self._engine,
            oom_killed=oom,
        )

    def _build_argv(self, command: str, name: str) -> list[str]:
        """Assemble the container run command.

        The user command is passed as a single argument to the container's
        shell, so it is never interpreted by a host shell.
        """
        argv = [self._engine, "run", "--rm", "--name", name]
        if self._engine == "podman":
            argv.append("--userns=keep-id")
        argv += ["--cap-drop", "ALL", "--security-opt", "no-new-privileges"]
        if not self._settings.network:
            argv += ["--network", "none"]
        if self._settings.memory:
            argv += ["--memory", self._settings.memory]
        if self._settings.cpus:
            argv += ["--cpus", str(self._settings.cpus)]
        if self._settings.pids_limit:
            argv += ["--pids-limit", str(self._settings.pids_limit)]
        if self._settings.read_only:
            argv += ["--read-only", "--tmpfs", "/tmp"]
        argv += ["--volume", f"{self._root}:/workspace", "--workdir", "/workspace"]
        argv += [self._settings.image, "sh", "-c", command]
        return argv

    def _force_remove(self, name: str) -> None:
        for action in (["kill", name], ["rm", "-f", name]):
            try:
                subprocess.run(
                    [self._engine, *action],
                    capture_output=True,
                    text=True,
                    timeout=_HOST_TIMEOUT_SLACK_SECONDS,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass


def build_command_runner(
    settings: SandboxSettings, workspace_root: Path
) -> CommandRunner:
    """Pick a command runner based on settings and what is installed.

    With the default 'auto' backend, a container engine is used when available
    and the host runner is the last resort. An explicit 'podman' or 'docker'
    backend that is not installed is a configuration error.
    """
    backend = (settings.backend or "auto").lower()
    if backend == "host":
        return HostCommandRunner(workspace_root)
    if backend in ("podman", "docker"):
        if shutil.which(backend) is None:
            raise ConfigError(
                f"Sandbox backend '{backend}' was requested but '{backend}' is "
                "not installed. Install it, or set sandbox.backend to 'auto' or "
                "'host'."
            )
        return ContainerCommandRunner(backend, settings, workspace_root)
    for engine in ("podman", "docker"):
        if shutil.which(engine) is not None:
            return ContainerCommandRunner(engine, settings, workspace_root)
    return HostCommandRunner(workspace_root)
