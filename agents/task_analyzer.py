"""Registry-driven task discovery and profiling."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from core.contracts import TaskSpec, normalize_modality
from core.modality_registry import ModalityRegistry
from core.runtime_contracts import DatasetBundle
from modalities import build_default_registry
from modalities.base import ModalityAdapter


@dataclass(frozen=True)
class TaskAnalysis:
    """Canonical task contract plus machine and human-readable profiles."""

    task_spec: TaskSpec
    profile: Mapping[str, object]
    report: str
    bundle: DatasetBundle | None = None


def _read_task_config(task_dir: Path) -> dict[str, object]:
    path = Path(task_dir) / "task_config.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("task_config.json must contain a JSON object")
    return loaded


class TaskAnalyzer:
    """Resolve a task through the adapter registered for its modality."""

    def __init__(
        self, registry: ModalityRegistry | None = None
    ) -> None:
        self.registry = registry or build_default_registry()

    def _adapter_for(self, task_dir: Path) -> ModalityAdapter:
        config = _read_task_config(task_dir)
        modality = normalize_modality(config.get("modality", "tabular"))
        adapter = self.registry.get(modality)
        if not isinstance(adapter, ModalityAdapter):
            raise TypeError(
                f"registered {modality!r} adapter does not implement "
                "the ModalityAdapter contract"
            )
        return adapter

    def resolve(self, task_dir: Path) -> TaskSpec:
        """Resolve only the canonical task contract without profiling data."""
        task_dir = Path(task_dir)
        return self._adapter_for(task_dir).discover(task_dir)

    @staticmethod
    def _validate_output_boundary(
        task_dir: Path, output_dir: Path
    ) -> None:
        task_root = Path(task_dir).resolve()
        output_root = Path(output_dir).resolve()
        if output_root == task_root or task_root in output_root.parents:
            raise ValueError(
                "task analysis output must be outside the read-only task "
                f"directory: {output_dir}"
            )

    def analyze(
        self,
        task_dir: Path,
        *,
        output_dir: Path | None = None,
        include_index: bool = False,
    ) -> TaskAnalysis:
        """Discover, profile, and optionally persist a canonical task."""
        task_dir = Path(task_dir)
        adapter = self._adapter_for(task_dir)
        task_spec = adapter.discover(task_dir)
        if task_spec.modality != adapter.name:
            raise ValueError(
                f"adapter {adapter.name!r} returned modality "
                f"{task_spec.modality!r}"
            )
        profile = dict(adapter.profile(task_dir, task_spec))
        report = adapter.render_report(task_dir, task_spec)
        bundle = (
            adapter.build_bundle(task_dir, task_spec)
            if include_index
            else None
        )
        analysis = TaskAnalysis(
            task_spec=task_spec,
            profile=profile,
            report=report,
            bundle=bundle,
        )
        if output_dir is not None:
            output_dir = Path(output_dir)
            self._validate_output_boundary(task_dir, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolved_task_spec.json").write_text(
                json.dumps(task_spec.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "dataset_profile.json").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "dataset_analysis_report.txt").write_text(
                report, encoding="utf-8"
            )
            if bundle is not None:
                (output_dir / "dataset_index_manifest.json").write_text(
                    json.dumps(
                        bundle.to_index_dict(),
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                with open(
                    output_dir / "dataset_index.jsonl",
                    "w",
                    encoding="utf-8",
                ) as stream:
                    for record in (
                        *bundle.train_records,
                        *bundle.test_records,
                    ):
                        stream.write(
                            json.dumps(record.to_dict(), default=str) + "\n"
                        )
        return analysis
