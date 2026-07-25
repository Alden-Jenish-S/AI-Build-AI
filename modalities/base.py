"""Common protocol implemented by modality adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle


@runtime_checkable
class ModalityAdapter(Protocol):
    """Minimum task-discovery surface required by ``TaskAnalyzer``."""

    name: str

    def discover(self, task_dir: Path) -> TaskSpec:
        """Resolve task-owned configuration and files into a canonical spec."""
        ...

    def profile(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> Mapping[str, object]:
        """Return a compact JSON-serializable dataset profile."""
        ...

    def build_bundle(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> DatasetBundle:
        """Build a lazy sample/entity index without decoding the corpus."""
        ...

    def render_report(
        self, task_dir: Path, task_spec: TaskSpec | None = None
    ) -> str:
        """Return a human-readable report suitable for agent prompts."""
        ...
