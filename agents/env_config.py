"""Small, dependency-free environment loader for command-line entry points."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path


_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_OPENALEX_ALIASES = ("OPENALEX_KEY", "OPENALEX_APIKEY")


@dataclass(frozen=True)
class EnvironmentLoadResult:
    """Non-secret metadata describing an environment load."""

    path: Path | None
    loaded_names: tuple[str, ...]
    openalex_alias: str | None = None


def _parse_value(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        lexer = shlex.shlex(value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        parts = list(lexer)
        if len(parts) != 1:
            raise ValueError("quoted value must contain exactly one shell token")
        return parts[0]
    # In dotenv syntax, an inline comment begins after whitespace. A literal
    # ``#`` inside a key or URL is preserved.
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def _read_environment_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected NAME=value")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError(f"{path}:{line_number}: invalid environment name")
        values[name] = _parse_value(raw_value)
    return values


def load_project_environment(project_root: Path) -> EnvironmentLoadResult:
    """Load a project ``.env`` without overriding exported process values.

    ``AIBUILDAI_ENV_FILE`` may select another file. Its relative paths are
    resolved from the project root. Only variable names—not values—are returned
    so callers can safely report configuration state.
    """

    root = Path(project_root).resolve()
    configured_path = os.getenv("AIBUILDAI_ENV_FILE", "").strip()
    if configured_path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Configured environment file does not exist: {path}")
    else:
        path = root / ".env"
        if not path.is_file():
            path = None

    loaded_names: list[str] = []
    if path is not None:
        for name, value in _read_environment_file(path).items():
            if name not in os.environ:
                os.environ[name] = value
                loaded_names.append(name)

    alias_used: str | None = None
    if not os.getenv("OPENALEX_API_KEY", "").strip():
        for alias in _OPENALEX_ALIASES:
            alias_value = os.getenv(alias, "").strip()
            if alias_value:
                os.environ["OPENALEX_API_KEY"] = alias_value
                alias_used = alias
                break

    return EnvironmentLoadResult(
        path=path,
        loaded_names=tuple(sorted(loaded_names)),
        openalex_alias=alias_used,
    )
