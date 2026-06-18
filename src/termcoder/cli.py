"""Command-line interface.

``termcoder`` with no arguments starts an interactive chat in the current
directory. Subcommands cover listing models and sessions.
"""

from __future__ import annotations

from pathlib import Path

import typer

from .config import load_config
from .errors import TermcoderError
from .providers.setup import configure_litellm

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="termcoder: a privacy-first, terminal-native agentic coding assistant.",
)

_WORKSPACE_OPTION = typer.Option(
    None, "--workspace", "-w", help="Workspace directory the agent may access."
)
_MODEL_OPTION = typer.Option(
    None, "--model", "-m", help="Model name from the registry to use."
)


def _workspace(path: Path | None) -> Path:
    return (path or Path.cwd()).resolve()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Start interactive chat when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        chat(workspace=None, model=None)


@app.command()
def chat(
    workspace: Path | None = _WORKSPACE_OPTION,
    model: str | None = _MODEL_OPTION,
) -> None:
    """Start an interactive coding session."""
    from .config_env import load_env_file
    from .ui.repl import run_repl

    workspace_path = _workspace(workspace)
    loaded = load_env_file(workspace_path)
    if loaded:
        typer.echo(f"Loaded {len(loaded)} secret(s) from .env: {', '.join(loaded)}")
    configure_litellm()
    try:
        config = load_config(workspace_path, model_override=model)
    except TermcoderError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    run_repl(config)


@app.command()
def models(workspace: Path | None = _WORKSPACE_OPTION) -> None:
    """List the configured models for a workspace."""
    config = load_config(_workspace(workspace))
    for name, model_config in config.models.items():
        marker = "*" if name == config.active_model else " "
        target = model_config.model
        typer.echo(f"{marker} {name:12s} {target}")


@app.command()
def sessions(workspace: Path | None = _WORKSPACE_OPTION) -> None:
    """List chat sessions stored for a workspace."""
    from .sessions.store import SessionStore

    config = load_config(_workspace(workspace))
    store = SessionStore(config.sessions_dir)
    found = store.list()
    if not found:
        typer.echo("No sessions yet.")
        return
    for meta in found:
        typer.echo(f"{meta.id}  {meta.model:10s}  {meta.title}")
