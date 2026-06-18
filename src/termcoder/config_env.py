"""Load secrets from a workspace ``.env`` file into the environment.

API keys are referenced by name from configuration and read from the
environment, never stored in config files. Exporting them by hand in every new
shell is tedious, so termcoder loads a ``.env`` file from the workspace at
startup when one is present. Values already set in the environment win, so an
explicit ``export`` still overrides the file and nothing is silently shadowed.

The parser is intentionally tiny: ``.env`` is a flat list of ``KEY=value``
lines. Blank lines and ``#`` comments are ignored, surrounding quotes are
stripped, and an optional leading ``export`` is tolerated so a file that
doubles as a shell script still works. Anything malformed is skipped rather
than raising, because a secrets file should never stop the tool from starting.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILENAME = ".env"


def load_env_file(workspace: Path, filename: str = ENV_FILENAME) -> list[str]:
    """Load ``workspace/.env`` into ``os.environ`` without overriding.

    Returns the list of variable names that were newly set, so the caller can
    report what was loaded without ever exposing the values.
    """
    path = Path(workspace) / filename
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    loaded: list[str] = []
    for raw_line in text.splitlines():
        parsed = _parse_line(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded


def _parse_line(raw_line: str) -> tuple[str, str] | None:
    """Parse one ``.env`` line into a (key, value) pair, or None to skip it."""
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, separator, value = line.partition("=")
    if not separator:
        return None
    key = key.strip()
    if not key or not key.replace("_", "").isalnum():
        return None
    return key, _clean_value(value)


def _clean_value(value: str) -> str:
    cleaned = value.strip()
    # Drop an inline comment only when the value is not quoted.
    if cleaned[:1] not in "\"'" and " #" in cleaned:
        cleaned = cleaned.split(" #", 1)[0].strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1]
    return cleaned
