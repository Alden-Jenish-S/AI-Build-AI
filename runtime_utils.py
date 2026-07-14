"""Shared validation and subprocess-safety helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping, Optional


_STORAGE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_ENV_PARTS = (
    "API_KEY",
    "ACCESS_KEY",
    "AUTH",
    "BEARER",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION_TOKEN",
)
_SENSITIVE_ENV_EXACT = {
    "AWS_PROFILE",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "KAGGLE_CONFIG_DIR",
    "NETRC",
}


def validate_storage_identifier(value: object, field_name: str) -> str:
    """Validate an L1 category or L2 artifact identifier."""
    if not isinstance(value, str) or not _STORAGE_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{field_name} must match {_STORAGE_IDENTIFIER.pattern!r}; got {value!r}"
        )
    return value


def validate_path_component(value: object, field_name: str) -> str:
    """Validate a user-controlled directory name without allowing traversal."""
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or not _PATH_COMPONENT.fullmatch(value)
    ):
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return value


def resolve_within(base_dir: Path, *parts: str) -> Path:
    """Resolve a descendant path and reject attempts to escape ``base_dir``."""
    base = Path(base_dir).resolve()
    candidate = base.joinpath(*parts).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"Path escapes allowed directory {base}: {candidate}")
    return candidate


def sanitized_subprocess_env(
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Copy an environment while removing credentials from untrusted children."""
    source = os.environ if base_env is None else base_env
    clean: dict[str, str] = {}
    for name, value in source.items():
        upper_name = name.upper()
        if upper_name in _SENSITIVE_ENV_EXACT:
            continue
        if any(part in upper_name for part in _SENSITIVE_ENV_PARTS):
            continue
        clean[name] = value
    return clean


def redact_local_paths(text: str, *additional_paths: Path) -> str:
    """Remove machine-specific absolute paths from persisted diagnostics."""
    redacted = text
    replacements = {Path.home(): "<HOME>"}
    replacements.update({Path(path).resolve(): "<WORKDIR>" for path in additional_paths})
    for path, label in sorted(
        replacements.items(), key=lambda item: len(str(item[0])), reverse=True
    ):
        redacted = redacted.replace(str(path), label)
    redacted = re.sub(
        r"/(?:private/)?var/folders/[^\s'\"]+|/tmp/[^\s'\"]+",
        "<TEMP_PATH>",
        redacted,
    )
    return redacted
