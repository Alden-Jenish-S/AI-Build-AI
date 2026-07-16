"""Shared validation and subprocess-safety helpers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Mapping, Optional


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
_ACCELERATORS = {"cpu", "cuda", "mps"}


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


def detect_available_accelerators(python_executable: str) -> set[str]:
    """Detect accelerators usable by the run without making GPU availability mandatory."""
    available = {"cpu"}
    try:
        probe = subprocess.run(
            [
                python_executable,
                "-c",
                (
                    "import torch; "
                    "print('cuda' if torch.cuda.is_available() else "
                    "('mps' if hasattr(torch.backends, 'mps') and "
                    "torch.backends.mps.is_available() else 'none'))"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            env=sanitized_subprocess_env(),
        )
        detected = probe.stdout.strip().lower()
        if probe.returncode == 0 and detected in {"cuda", "mps"}:
            available.add(detected)
    except Exception:
        pass

    # CatBoost/LightGBM can use CUDA even when PyTorch is absent. nvidia-smi is
    # therefore a useful secondary hardware probe, unless CUDA was explicitly
    # hidden from this process.
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        "cuda" not in available
        and cuda_visible not in {"", "-1"}
        and shutil.which("nvidia-smi")
    ):
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                env=sanitized_subprocess_env(),
            )
            if probe.returncode == 0 and probe.stdout.strip():
                available.add("cuda")
        except Exception:
            pass
    return available


def select_preferred_accelerator(
    available: Iterable[str], preference: str = "auto"
) -> str:
    """Select CUDA, then MPS, then CPU while respecting an optional preference."""
    normalized = {str(item).lower() for item in available} & _ACCELERATORS
    normalized.add("cpu")
    requested = str(preference or "auto").lower()
    if requested not in {"auto", "gpu", "cpu", "cuda", "mps"}:
        raise ValueError(
            "preferred_accelerator must be one of auto, gpu, cpu, cuda, or mps"
        )
    if requested == "cpu":
        return "cpu"
    if requested in {"cuda", "mps"} and requested in normalized:
        return requested
    for candidate in ("cuda", "mps", "cpu"):
        if candidate in normalized:
            return candidate
    return "cpu"


def accelerator_subprocess_env(
    accelerator: str,
    base_env: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    """Return a sanitized child environment carrying the selected accelerator contract."""
    selected = str(accelerator).lower()
    if selected not in _ACCELERATORS:
        raise ValueError(f"Unsupported selected accelerator: {accelerator!r}")
    clean = sanitized_subprocess_env(base_env)
    clean["AIBUILDAI_ACCELERATOR"] = selected
    clean["AIBUILDAI_ACTUAL_ACCELERATOR"] = selected
    clean["AIBUILDAI_PREFER_GPU"] = "1" if selected in {"cuda", "mps"} else "0"
    if selected == "cuda":
        clean.setdefault("AIBUILDAI_CUDA_DEVICES", "0")
        clean.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
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
