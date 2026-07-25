"""Manifest-backed image/audio/video adapter foundation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from core.contracts import InputSpec, TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord
from .common import (
    canonical_input_reference,
    read_manifest,
    read_task_config,
    records_to_bundle,
    render_profile_report,
    resolve_task_source,
    row_split,
)


class ManifestMediaAdapter:
    """Common lazy path indexing for one media modality."""

    name = ""
    extensions: frozenset[str] = frozenset()
    path_field_candidates: tuple[str, ...] = ("path",)

    def discover(self, task_dir: Path) -> TaskSpec:
        config = read_task_config(task_dir)
        if not config:
            raise ValueError(
                f"{self.name} tasks require task_config.json schema_version 2"
            )
        configured = str(config.get("modality", "")).strip().lower()
        if configured != self.name:
            raise ValueError(
                f"{self.__class__.__name__} cannot resolve modality "
                f"{configured!r}"
            )
        return TaskSpec.from_mapping(Path(task_dir).name, config)

    def _input_spec(self, task: TaskSpec) -> InputSpec:
        candidates = [
            item
            for item in task.inputs.values()
            if item.modality == self.name
        ]
        if not candidates:
            raise ValueError(
                f"{self.name} task does not define a {self.name} input"
            )
        return candidates[0]

    def _path_field(
        self, frame: pd.DataFrame, input_spec: InputSpec
    ) -> str:
        configured = input_spec.options.get("path_field")
        candidates = (
            (str(configured),) if configured else self.path_field_candidates
        )
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        raise ValueError(
            f"{self.name} manifest must contain one of {list(candidates)}"
        )

    def _manifest_for(
        self, task_dir: Path, task: TaskSpec, input_spec: InputSpec
    ) -> tuple[pd.DataFrame | None, Path]:
        source = resolve_task_source(task_dir, input_spec.source)
        manifest_option = input_spec.options.get("manifest")
        target_source = task.target.source if task.target else None
        manifest_path = None
        if manifest_option:
            manifest_path = resolve_task_source(
                task_dir, str(manifest_option)
            )
        elif target_source:
            candidate = resolve_task_source(task_dir, target_source)
            if candidate.is_file():
                manifest_path = candidate
        elif source.is_file():
            manifest_path = source
        if manifest_path is not None:
            return read_manifest(manifest_path), source
        if not source.is_dir():
            raise ValueError(
                f"{self.name} source must be a manifest or directory"
            )
        return None, source

    def _directory_records(
        self, task_dir: Path, task: TaskSpec, source: Path
    ) -> Iterable[SampleRecord]:
        paths = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        )
        if not paths:
            raise ValueError(
                f"no supported {self.name} files found below {source}"
            )
        for index, path in enumerate(paths):
            target = (
                path.parent.name
                if task.problem_type
                in {"classification", "multilabel_classification"}
                else None
            )
            if task.target is not None and target is None:
                raise ValueError(
                    "directory-only regression requires a target manifest"
                )
            sample_id = path.relative_to(source).as_posix()
            yield SampleRecord(
                sample_id=sample_id or str(index),
                inputs={
                    self._input_spec(task).name:
                    canonical_input_reference(task_dir, path)
                },
                target=target,
                entity_id=sample_id,
                split="train",
            )

    def build_bundle(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> DatasetBundle:
        input_spec = self._input_spec(task_spec)
        frame, source = self._manifest_for(
            task_dir, task_spec, input_spec
        )
        if frame is None:
            return records_to_bundle(
                task_spec,
                self._directory_records(task_dir, task_spec, source),
                metadata={"index_source": "directory"},
            )
        path_field = self._path_field(frame, input_spec)
        target_field = (
            task_spec.target.field if task_spec.target is not None else None
        )
        sample_id_field = task_spec.sample_id_field
        split_field = str(
            input_spec.options.get("split_field", "split")
        )
        records = []
        for index, series in frame.iterrows():
            row = series.to_dict()
            raw_path = row.get(path_field)
            if raw_path is None or pd.isna(raw_path):
                raise ValueError(
                    f"{self.name} manifest row {index} has no {path_field}"
                )
            path = resolve_task_source(
                task_dir,
                str(raw_path),
                relative_to=source if source.is_dir() else source.parent,
            )
            if path.suffix.lower() not in self.extensions:
                raise ValueError(
                    f"unsupported {self.name} file extension: {path}"
                )
            sample_id_value = row.get(sample_id_field)
            sample_id = (
                str(sample_id_value)
                if sample_id_value is not None
                and not pd.isna(sample_id_value)
                else str(index)
            )
            target = (
                row.get(target_field) if target_field is not None else None
            )
            split = row_split(
                row,
                split_field=split_field,
                target_field=target_field,
            )
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    inputs={
                        input_spec.name: canonical_input_reference(
                            task_dir, path
                        )
                    },
                    target=None if split == "test" else target,
                    entity_id=(
                        str(row.get(task_spec.entity_id_field))
                        if task_spec.entity_id_field
                        and row.get(task_spec.entity_id_field) is not None
                        else sample_id
                    ),
                    group_id=(
                        str(row.get(task_spec.group_id_field))
                        if task_spec.group_id_field
                        and row.get(task_spec.group_id_field) is not None
                        else None
                    ),
                    timestamp=(
                        row.get(task_spec.time_field)
                        if task_spec.time_field
                        else None
                    ),
                    split=split,
                )
            )
        return records_to_bundle(
            task_spec,
            records,
            metadata={
                "index_source": (
                    canonical_input_reference(task_dir, source)
                    if source.is_file()
                    else str(source.name)
                ),
                "path_field": path_field,
            },
        )

    def render_report(
        self, task_dir: Path, task_spec: TaskSpec | None = None
    ) -> str:
        task = task_spec or self.discover(task_dir)
        return render_profile_report(task, self.profile(task_dir, task))
