"""Entity-aligned multimodal task adapter."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from core.contracts import InputSpec, TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord
from runtime_utils import task_data_files
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
from .tabular import TabularAdapter, discover_dataset_layout
from .paired_directory import (
    build_paired_directory_bundle,
    discover_paired_directory_layout,
    paired_layout_config,
)


_PATH_FIELDS = {
    "image": ("image_path", "image", "path", "file_path", "filename"),
    "audio": ("audio_path", "recording_path", "path", "file_path", "filename"),
    "video": ("video_path", "clip_path", "path", "file_path", "filename"),
}
_MEDIA_EXTENSIONS = {
    "audio": frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}),
    "image": frozenset(
        {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    ),
    "video": frozenset(
        {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
    ),
}


def _join_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return Path(str(value).strip()).stem


def _task_relative(task_dir: Path, path: Path) -> str:
    source_root = (
        Path(task_dir) / "input"
        if (Path(task_dir) / "input").is_dir()
        else Path(task_dir)
    )
    return path.relative_to(source_root).as_posix()


class MultimodalAdapter:
    name = "multimodal"

    @staticmethod
    def _legacy_join_info(task_dir: Path) -> dict[str, object] | None:
        """Detect a legacy table plus ID-addressable media corpus."""
        task_dir = Path(task_dir)
        if read_task_config(task_dir):
            return None
        layout = discover_dataset_layout(task_dir)
        train_name = layout.get("roles", {}).get("train")
        if not train_name:
            return None
        source_root = (
            task_dir / "input"
            if (task_dir / "input").is_dir()
            else task_dir
        )
        train_path = source_root / str(train_name)
        test_name = layout.get("roles", {}).get("test")
        test_path = source_root / str(test_name) if test_name else None
        if not train_path.is_file():
            return None
        try:
            train = pd.read_csv(
                train_path,
                sep="\t" if train_path.suffix.lower() == ".tsv" else ",",
            )
            test = (
                pd.read_csv(
                    test_path,
                    sep=(
                        "\t"
                        if test_path and test_path.suffix.lower() == ".tsv"
                        else ","
                    ),
                )
                if test_path is not None and test_path.is_file()
                else pd.DataFrame()
            )
        except Exception:
            return None
        if train.empty:
            return None

        discovered_files = task_data_files(task_dir)
        media_by_modality: dict[str, list[Path]] = {}
        for modality, extensions in _MEDIA_EXTENSIONS.items():
            files = [
                path
                for path in discovered_files
                if path.suffix.lower() in extensions
            ]
            if files:
                media_by_modality[modality] = files
        if not media_by_modality:
            return None
        media_keys = {
            modality: {_join_key(path.stem) for path in files}
            for modality, files in media_by_modality.items()
        }
        shared_columns = [
            column
            for column in train.columns
            if test.empty or column in test.columns
        ]
        preferred = (
            "id",
            "sample_id",
            "image_id",
            "audio_id",
            "video_id",
            "file_id",
            "filename",
        )
        ordered_columns = [
            *[column for column in preferred if column in shared_columns],
            *[column for column in shared_columns if column not in preferred],
        ]
        populations = pd.concat(
            [
                train[ordered_columns],
                test[ordered_columns] if not test.empty else pd.DataFrame(),
            ],
            ignore_index=True,
        )
        best: tuple[float, str, dict[str, list[Path]]] | None = None
        for column in ordered_columns:
            values = {_join_key(value) for value in populations[column]}
            values.discard("")
            if not values:
                continue
            matched_modalities = {
                modality: files
                for modality, files in media_by_modality.items()
                if len(values & media_keys[modality]) / len(values) >= 0.50
            }
            if not matched_modalities:
                continue
            coverage = max(
                len(values & media_keys[modality]) / len(values)
                for modality in matched_modalities
            )
            candidate = (coverage, str(column), matched_modalities)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        return {
            "layout": layout,
            "train_path": train_path,
            "test_path": test_path,
            "train": train,
            "test": test,
            "join_field": best[1],
            "media": best[2],
        }

    @classmethod
    def can_auto_discover(cls, task_dir: Path) -> bool:
        """Return whether a config-less mixed task has a safe entity join."""
        paired = discover_paired_directory_layout(task_dir)
        if paired is not None and paired.modality == "multimodal":
            return True
        info = cls._legacy_join_info(task_dir)
        if info is None:
            return False
        # A table/media join is not enough to prove that a supervised target
        # exists.  Targetless metadata tables previously became invalid schema
        # v2 classification contracts.
        try:
            return TabularAdapter().discover(task_dir).target is not None
        except (FileNotFoundError, TypeError, ValueError):
            return False

    def discover(self, task_dir: Path) -> TaskSpec:
        config = read_task_config(task_dir)
        if not config:
            paired = discover_paired_directory_layout(task_dir)
            if paired is not None and paired.modality == "multimodal":
                config = paired_layout_config(task_dir, paired)
                return TaskSpec.from_mapping(Path(task_dir).name, config)
            info = self._legacy_join_info(task_dir)
            if info is None:
                raise ValueError(
                    "multimodal tasks require task_config.json schema_version "
                    "2 unless tabular rows can be joined safely to media by ID"
                )
            tabular_task = TabularAdapter().discover(task_dir)
            join_field = str(info["join_field"])
            train_path = Path(info["train_path"])
            test_path = info.get("test_path")
            train = info["train"]
            target_field = (
                tabular_task.target.field if tabular_task.target else None
            )
            inputs: dict[str, object] = {}
            component_modalities = []
            all_rows = len(train) + len(info["test"])
            for modality, files in info["media"].items():
                root = Path(files[0]).parent
                while any(root not in path.parents and root != path for path in files):
                    root = root.parent
                covered = {
                    _join_key(path.stem) for path in files
                }
                row_keys = {
                    _join_key(value)
                    for value in pd.concat(
                        [train[join_field], info["test"].get(join_field, pd.Series(dtype=object))]
                    )
                }
                inputs[modality] = {
                    "modality": modality,
                    "source": _task_relative(task_dir, root),
                    "format": "directory",
                    "required": len(covered & row_keys) == all_rows,
                    "auto_join": True,
                    "join_field": join_field,
                }
                component_modalities.append(modality)
            feature_fields = [
                str(column)
                for column in train.columns
                if column not in {join_field, target_field}
            ]
            inputs["metadata"] = {
                "modality": "tabular",
                "source": _task_relative(task_dir, train_path),
                "format": train_path.suffix.lower().lstrip("."),
                "feature_fields": feature_fields,
                "test_source": (
                    _task_relative(task_dir, Path(test_path))
                    if test_path is not None
                    else None
                ),
                "auto_join": True,
                "join_field": join_field,
            }
            component_modalities.append("tabular")
            config = {
                "schema_version": 2,
                "modality": "multimodal",
                "component_modalities": component_modalities,
                "problem_type": tabular_task.problem_type,
                "inputs": inputs,
                "sample_id_field": join_field,
                "entity_id_field": join_field,
                "target": (
                    {
                        "source": _task_relative(task_dir, train_path),
                        "field": target_field,
                    }
                    if target_field is not None
                    else None
                ),
                "output": tabular_task.output.to_dict(),
                "metrics": [
                    metric.to_dict() for metric in tabular_task.metrics
                ],
                "primary_metric": tabular_task.primary_metric,
            }
        return TaskSpec.from_mapping(Path(task_dir).name, config)

    def _build_legacy_join_bundle(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> DatasetBundle:
        info = self._legacy_join_info(task_dir)
        if info is None:
            raise ValueError("legacy multimodal join is no longer resolvable")
        join_field = str(info["join_field"])
        target_field = task_spec.target.field if task_spec.target else None
        media_maps = {
            modality: {
                _join_key(path.stem): path for path in files
            }
            for modality, files in info["media"].items()
        }
        metadata_spec = task_spec.inputs["metadata"]
        feature_fields = [
            str(item)
            for item in metadata_spec.options.get("feature_fields", [])
        ]
        records = []
        for split, frame in (("train", info["train"]), ("test", info["test"])):
            for _, series in frame.iterrows():
                row = series.to_dict()
                key = _join_key(row.get(join_field))
                inputs: dict[str, object] = {
                    "metadata": {
                        field: json_safe(row.get(field))
                        for field in feature_fields
                    }
                }
                for name, input_spec in task_spec.inputs.items():
                    if input_spec.modality not in media_maps:
                        continue
                    path = media_maps[input_spec.modality].get(key)
                    if path is None and input_spec.required:
                        raise ValueError(
                            f"entity {key!r} is missing required "
                            f"{input_spec.modality} input"
                        )
                    inputs[name] = (
                        canonical_input_reference(task_dir, path)
                        if path is not None
                        else None
                    )
                records.append(
                    SampleRecord(
                        sample_id=key,
                        entity_id=key,
                        group_id=key,
                        inputs=inputs,
                        target=(
                            json_safe(row.get(target_field))
                            if split == "train" and target_field
                            else None
                        ),
                        split=split,
                    )
                )
        return records_to_bundle(
            task_spec,
            records,
            metadata={
                "index_source": "auto_joined_legacy_tables_and_media",
                "join_field": join_field,
                "component_coverage": {
                    modality: len(paths)
                    for modality, paths in media_maps.items()
                },
                "missing_modality_policy": "explicit_none_and_mask",
            },
        )

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
        if any(
            bool(input_spec.options.get("auto_paired_target"))
            for input_spec in task_spec.inputs.values()
        ):
            return build_paired_directory_bundle(task_dir, task_spec)
        if any(
            bool(input_spec.options.get("auto_join"))
            for input_spec in task_spec.inputs.values()
        ):
            return self._build_legacy_join_bundle(task_dir, task_spec)
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
        profile = {
            **bundle.to_index_dict(),
            "component_modalities": list(
                task_spec.component_modalities
            ),
            "missing_components": missing,
            "split_unit": "entity_id",
        }
        if task_spec.target and str(task_spec.target.type or "").endswith("_path"):
            profile["structured_target_references"] = len(bundle.train_records)
            profile["target_storage"] = task_spec.target.type
        else:
            profile["target_distribution"] = dict(
                Counter(str(record.target) for record in bundle.train_records)
            )
        return profile

    def render_report(
        self, task_dir: Path, task_spec: TaskSpec | None = None
    ) -> str:
        task = task_spec or self.discover(task_dir)
        return render_profile_report(task, self.profile(task_dir, task))
