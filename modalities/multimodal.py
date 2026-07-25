"""Entity-aligned multimodal task adapter."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import pandas as pd

from core.contracts import InputSpec, TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord
from .common import (
    canonical_input_reference,
    json_safe,
    read_manifest,
    read_task_config,
    records_to_bundle,
    render_profile_report,
    resolve_task_source,
    row_split,
)


_PATH_FIELDS = {
    "image": ("image_path", "image", "path", "file_path", "filename"),
    "audio": ("audio_path", "recording_path", "path", "file_path", "filename"),
    "video": ("video_path", "clip_path", "path", "file_path", "filename"),
}


class MultimodalAdapter:
    name = "multimodal"

    def discover(self, task_dir: Path) -> TaskSpec:
        config = read_task_config(task_dir)
        if not config:
            raise ValueError(
                "multimodal tasks require task_config.json schema_version 2"
            )
        return TaskSpec.from_mapping(Path(task_dir).name, config)

    @staticmethod
    def _frame(
        task_dir: Path, input_spec: InputSpec
    ) -> tuple[pd.DataFrame, Path]:
        source = resolve_task_source(task_dir, input_spec.source)
        manifest = input_spec.options.get("manifest")
        if manifest:
            manifest_path = resolve_task_source(task_dir, str(manifest))
        elif source.is_file():
            manifest_path = source
        else:
            raise ValueError(
                f"multimodal input {input_spec.name!r} with directory source "
                "must define a manifest"
            )
        return read_manifest(manifest_path), source

    @staticmethod
    def _field(
        frame: pd.DataFrame,
        input_spec: InputSpec,
        option: str,
        candidates: tuple[str, ...],
    ) -> str:
        configured = input_spec.options.get(option)
        names = (str(configured),) if configured else candidates
        for name in names:
            if name in frame.columns:
                return name
        raise ValueError(
            f"input {input_spec.name!r} requires one of {list(names)}"
        )

    def _input_values(
        self,
        task_dir: Path,
        task_spec: TaskSpec,
        input_spec: InputSpec,
    ) -> tuple[dict[str, object], dict[str, str], dict[str, Mapping]]:
        frame, source = self._frame(task_dir, input_spec)
        id_field = (
            task_spec.entity_id_field or task_spec.sample_id_field
        )
        if id_field not in frame.columns:
            raise ValueError(
                f"multimodal manifest for {input_spec.name!r} is missing "
                f"entity field {id_field!r}"
            )
        values: dict[str, object] = {}
        splits: dict[str, str] = {}
        rows: dict[str, Mapping] = {}
        for _, series in frame.iterrows():
            row = series.to_dict()
            entity_id = str(row[id_field])
            if entity_id in values:
                raise ValueError(
                    f"duplicate entity {entity_id!r} for input "
                    f"{input_spec.name!r}"
                )
            if input_spec.modality in _PATH_FIELDS:
                path_field = self._field(
                    frame,
                    input_spec,
                    "path_field",
                    _PATH_FIELDS[input_spec.modality],
                )
                raw_path = row.get(path_field)
                value = None
                if raw_path is not None and not pd.isna(raw_path):
                    path = resolve_task_source(
                        task_dir,
                        str(raw_path),
                        relative_to=(
                            source if source.is_dir() else source.parent
                        ),
                    )
                    value = canonical_input_reference(task_dir, path)
            elif input_spec.modality == "text":
                text_field = self._field(
                    frame,
                    input_spec,
                    "text_field",
                    ("text", "content", "caption", input_spec.name),
                )
                value = json_safe(row.get(text_field))
            elif input_spec.modality == "tabular":
                configured = input_spec.options.get("feature_fields")
                if configured is None:
                    excluded = {
                        id_field,
                        task_spec.target.field if task_spec.target else None,
                        task_spec.group_id_field,
                        task_spec.time_field,
                        "split",
                    }
                    fields = [
                        column
                        for column in frame.columns
                        if column not in excluded
                    ]
                elif isinstance(configured, list):
                    fields = [str(item) for item in configured]
                else:
                    raise ValueError(
                        "tabular feature_fields must be a list"
                    )
                missing = [
                    field for field in fields if field not in frame.columns
                ]
                if missing:
                    raise ValueError(
                        f"tabular input fields are missing: {missing}"
                    )
                value = {
                    field: json_safe(row.get(field)) for field in fields
                }
            else:
                raise ValueError(
                    f"unsupported multimodal component "
                    f"{input_spec.modality!r}"
                )
            values[entity_id] = value
            splits[entity_id] = row_split(
                row,
                split_field=str(
                    input_spec.options.get("split_field", "split")
                ),
                target_field=(
                    task_spec.target.field if task_spec.target else None
                ),
            )
            rows[entity_id] = row
        return values, splits, rows

    def build_bundle(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> DatasetBundle:
        component_values: dict[str, dict[str, object]] = {}
        component_splits: dict[str, dict[str, str]] = {}
        source_rows: dict[str, Mapping] = {}
        entity_ids: set[str] = set()
        for name, input_spec in task_spec.inputs.items():
            values, splits, rows = self._input_values(
                task_dir, task_spec, input_spec
            )
            component_values[name] = values
            component_splits[name] = splits
            entity_ids.update(values)
            source_rows.update(rows)

        target_values: dict[str, object] = {}
        if task_spec.target is not None:
            target_source = resolve_task_source(
                task_dir, task_spec.target.source or ""
            )
            target_frame = read_manifest(target_source)
            id_field = (
                task_spec.entity_id_field or task_spec.sample_id_field
            )
            if (
                id_field not in target_frame.columns
                or task_spec.target.field not in target_frame.columns
            ):
                raise ValueError(
                    "target manifest is missing entity or target fields"
                )
            target_values = {
                str(row[id_field]): row[task_spec.target.field]
                for _, row in target_frame.iterrows()
                if not pd.isna(row[task_spec.target.field])
            }
            entity_ids.update(
                str(item) for item in target_frame[id_field].tolist()
            )

        records = []
        coverage = Counter()
        for entity_id in sorted(entity_ids):
            inputs = {}
            split_votes = []
            missing_required = []
            for name, input_spec in task_spec.inputs.items():
                value = component_values[name].get(entity_id)
                if value is None and input_spec.required:
                    missing_required.append(name)
                inputs[name] = value
                if value is not None:
                    coverage[name] += 1
                if entity_id in component_splits[name]:
                    split_votes.append(component_splits[name][entity_id])
            if missing_required:
                raise ValueError(
                    f"entity {entity_id!r} is missing required modalities "
                    f"{missing_required}"
                )
            split = (
                "test"
                if "test" in split_votes or entity_id not in target_values
                else "train"
            )
            source_row = source_rows.get(entity_id, {})
            records.append(
                SampleRecord(
                    sample_id=entity_id,
                    entity_id=entity_id,
                    inputs=inputs,
                    target=(
                        target_values.get(entity_id)
                        if split == "train"
                        else None
                    ),
                    group_id=(
                        str(source_row.get(task_spec.group_id_field))
                        if task_spec.group_id_field
                        and source_row.get(task_spec.group_id_field)
                        is not None
                        else entity_id
                    ),
                    timestamp=(
                        source_row.get(task_spec.time_field)
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
                "entity_count": len(entity_ids),
                "component_coverage": dict(coverage),
                "missing_modality_policy": "explicit_none_and_mask",
            },
        )

    def profile(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> dict[str, object]:
        bundle = self.build_bundle(task_dir, task_spec)
        missing = {
            name: sum(
                record.inputs.get(name) is None
                for record in (*bundle.train_records, *bundle.test_records)
            )
            for name in task_spec.inputs
        }
        return {
            **bundle.to_index_dict(),
            "component_modalities": list(
                task_spec.component_modalities
            ),
            "missing_components": missing,
            "target_distribution": dict(
                Counter(
                    str(record.target)
                    for record in bundle.train_records
                )
            ),
            "split_unit": "entity_id",
        }

    def render_report(
        self, task_dir: Path, task_spec: TaskSpec | None = None
    ) -> str:
        task = task_spec or self.discover(task_dir)
        return render_profile_report(task, self.profile(task_dir, task))
