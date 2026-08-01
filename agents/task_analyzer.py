"""Registry-driven task discovery and profiling."""

from __future__ import annotations

import json
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from core.contracts import TaskSpec, normalize_modality
from core.modality_registry import ModalityRegistry
from core.runtime_contracts import DatasetBundle
from evaluation.metrics import (
    infer_metric_direction,
    infer_metric_from_description,
)
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
        if not config:
            multimodal = self.registry.get("multimodal")
            if (
                hasattr(multimodal, "can_auto_discover")
                and multimodal.can_auto_discover(task_dir)
            ):
                adapter = multimodal
                modality = "multimodal"
            else:
                modality = "tabular"
                adapter = self.registry.get(modality)
        else:
            modality = normalize_modality(config.get("modality", "tabular"))
            adapter = self.registry.get(modality)
        if not isinstance(adapter, ModalityAdapter):
            raise TypeError(
                f"registered {modality!r} adapter does not implement "
                "the ModalityAdapter contract"
            )
        return adapter

    @staticmethod
    def _explicit_metric_config(task_dir: Path) -> bool:
        config = _read_task_config(task_dir)
        return any(
            config.get(key) is not None
            for key in ("metrics", "metric_name", "primary_metric")
        )

    @staticmethod
    def _explicit_sample_id_config(task_dir: Path) -> bool:
        config = _read_task_config(task_dir)
        return any(
            config.get(key) is not None
            for key in ("sample_id_field", "id_column")
        )

    @staticmethod
    def _task_description(task_dir: Path) -> str:
        path = Path(task_dir) / "task_description.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    @staticmethod
    def _safe_task_path(task_dir: Path, source: str) -> Path | None:
        root = Path(task_dir).resolve()
        raw = Path(str(source))
        candidates = (
            (raw,)
            if raw.is_absolute()
            else (root / raw, root / "input" / raw)
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved != root and root not in resolved.parents:
                continue
            if resolved.is_file():
                return resolved
        return None

    @classmethod
    def _submission_class_names(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> tuple[str, ...]:
        if (
            task_spec.output.type != "class_probabilities"
            or task_spec.output.class_names
        ):
            return task_spec.output.class_names
        candidates = [
            item.source
            for item in task_spec.inputs.values()
            if item.role == "sample_submission"
            or item.name == "sample_submission"
        ]
        candidates.extend(
            ("sample_submission.csv", "input/sample_submission.csv")
        )
        for source in candidates:
            path = cls._safe_task_path(task_dir, source)
            if path is None:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            prediction_columns = tuple(
                str(column) for column in columns[1:] if str(column)
            )
            # A single probability column is normally the positive class, not
            # the complete binary class vocabulary.
            if len(prediction_columns) > 1:
                return prediction_columns
        return ()

    @classmethod
    def _submission_columns(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> tuple[str, ...]:
        candidates = [
            item.source
            for item in task_spec.inputs.values()
            if item.role == "sample_submission"
            or item.name == "sample_submission"
        ]
        candidates.extend(
            ("sample_submission.csv", "input/sample_submission.csv")
        )
        for source in candidates:
            path = cls._safe_task_path(task_dir, source)
            if path is None:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            if len(columns) >= 2:
                return tuple(str(column) for column in columns)
        return ()

    @classmethod
    def _submission_id_field(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> str:
        """Infer the final-output ID only when it exists in declared test data."""
        if cls._explicit_sample_id_config(task_dir):
            return task_spec.sample_id_field
        template_sources = [
            item.source
            for item in task_spec.inputs.values()
            if item.role == "sample_submission"
            or item.name == "sample_submission"
        ]
        template_sources.extend(
            ("sample_submission.csv", "input/sample_submission.csv")
        )
        template_id = None
        for source in template_sources:
            path = cls._safe_task_path(task_dir, source)
            if path is None:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            if columns:
                template_id = str(columns[0])
                break
        if not template_id:
            return task_spec.sample_id_field
        for item in task_spec.inputs.values():
            if item.role != "test" and item.name != "test":
                continue
            path = cls._safe_task_path(task_dir, item.source)
            if path is None or path.suffix.lower() not in {".csv", ".tsv"}:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            if template_id in columns:
                return template_id
        return task_spec.sample_id_field

    @classmethod
    def _enrich_discovered_task(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> TaskSpec:
        mapping = task_spec.to_dict()
        changed = False
        if not cls._explicit_metric_config(task_dir):
            inferred_metric = infer_metric_from_description(
                cls._task_description(task_dir)
            )
            if inferred_metric and inferred_metric != task_spec.primary_metric:
                mapping["metrics"] = [
                    {
                        "name": inferred_metric,
                        "direction": infer_metric_direction(inferred_metric),
                    }
                ]
                mapping["primary_metric"] = inferred_metric
                changed = True
        inferred_classes = cls._submission_class_names(task_dir, task_spec)
        if inferred_classes and inferred_classes != task_spec.output.class_names:
            output = dict(mapping["output"])
            output["class_names"] = list(inferred_classes)
            mapping["output"] = output
            changed = True
        submission_columns = cls._submission_columns(task_dir, task_spec)
        if submission_columns:
            output = dict(mapping["output"])
            options = dict(output.get("options", {}))
            submission_contract = {
                "submission_id_column": submission_columns[0],
                "submission_prediction_columns": list(
                    submission_columns[1:]
                ),
            }
            if any(
                options.get(key) != value
                for key, value in submission_contract.items()
            ):
                options.update(submission_contract)
                output["options"] = options
                mapping["output"] = output
                changed = True
        inferred_id_field = cls._submission_id_field(task_dir, task_spec)
        if inferred_id_field != task_spec.sample_id_field:
            mapping["sample_id_field"] = inferred_id_field
            changed = True
        return (
            TaskSpec.from_mapping(task_spec.task_id, mapping)
            if changed
            else task_spec
        )

    def _discover(
        self, task_dir: Path
    ) -> tuple[ModalityAdapter, TaskSpec]:
        adapter = self._adapter_for(task_dir)
        task_spec = self._enrich_discovered_task(
            task_dir, adapter.discover(task_dir)
        )
        return adapter, task_spec

    def resolve(self, task_dir: Path) -> TaskSpec:
        """Resolve only the canonical task contract without profiling data."""
        task_dir = Path(task_dir)
        return self._discover(task_dir)[1]

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
        adapter, task_spec = self._discover(task_dir)
        if task_spec.modality != adapter.name:
            raise ValueError(
                f"adapter {adapter.name!r} returned modality "
                f"{task_spec.modality!r}"
            )
        profile = dict(adapter.profile(task_dir, task_spec))
        report = adapter.render_report(task_dir, task_spec)
        # Tabular data already has an efficient columnar source (CSV/TSV).
        # Expanding every row into JSONL duplicates the complete dataset and
        # can add hundreds of megabytes before the first experiment starts.
        # Structured modalities still need the record index for path/component
        # joins, so retain it only for those adapters.
        direct_tabular = include_index and task_spec.modality == "tabular"
        bundle = (
            adapter.build_bundle(task_dir, task_spec)
            if include_index and not direct_tabular
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
            elif direct_tabular:
                source_metadata = []
                for input_spec in task_spec.inputs.values():
                    source_path = self._safe_task_path(
                        task_dir, input_spec.source
                    )
                    if source_path is None:
                        continue
                    stat = source_path.stat()
                    source_metadata.append(
                        {
                            "name": input_spec.name,
                            "role": input_spec.role,
                            "source": input_spec.source,
                            "size_bytes": int(stat.st_size),
                            "mtime_ns": int(stat.st_mtime_ns),
                        }
                    )
                fingerprint_payload = {
                    "task": task_spec.to_dict(),
                    "sources": source_metadata,
                }
                fingerprint = hashlib.sha256(
                    json.dumps(
                        fingerprint_payload,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()
                (output_dir / "dataset_index_manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "storage": "direct_tabular",
                            "dataset_fingerprint": fingerprint,
                            "sources": source_metadata,
                            "row_index_materialized": False,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        return analysis
