"""Task-native adapter for evidence-resolved, previously unseen data formats."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord
from .base import ModalityAdapter
from .common import canonical_input_reference, records_to_bundle, task_input_root


class GenericAdapter(ModalityAdapter):
    """Materialize a contract from declared sources without category dispatch.

    This adapter is intentionally conservative. It handles row-oriented files
    and stem-aligned file collections using the exact sources/roles in the
    resolved contract. If those facts are insufficient, it raises before search
    instead of inventing a split or target.
    """

    name = "task_native"
    handles_arbitrary_task_identifiers = True

    def __init__(self, mapping: Mapping[str, object]) -> None:
        self.mapping = dict(mapping)
        resolution = self.mapping.get("_resolution", {})
        self.resolution = (
            dict(resolution) if isinstance(resolution, Mapping) else {}
        )

    def discover(self, task_dir: Path) -> TaskSpec:
        return TaskSpec.from_mapping(Path(task_dir).name, self.mapping)

    @staticmethod
    def _source(task_dir: Path, source: str) -> Path:
        root = task_input_root(task_dir).resolve()
        raw = Path(str(source))
        candidates = (raw,) if raw.is_absolute() else (root / raw, Path(task_dir) / raw)
        for candidate in candidates:
            resolved = candidate.resolve()
            task_root = Path(task_dir).resolve()
            if resolved != task_root and task_root not in resolved.parents:
                continue
            if resolved.exists():
                return resolved
        raise FileNotFoundError(f"declared task source does not exist: {source!r}")

    @staticmethod
    def _read_frame(path: Path, declared_format: str) -> pd.DataFrame:
        normalized = str(declared_format or path.suffix.lstrip(".")).lower()
        if normalized in {"csv", "tsv"}:
            return pd.read_csv(path, sep="\t" if normalized == "tsv" else ",")
        if normalized in {"jsonl", "ndjson"}:
            return pd.read_json(path, lines=True)
        if normalized == "json":
            return pd.read_json(path)
        if normalized == "parquet":
            return pd.read_parquet(path)
        raise ValueError(
            f"declared row-oriented format {declared_format!r} needs a registered reader"
        )

    @staticmethod
    def _role_specs(task_spec: TaskSpec, role: str):
        return [spec for spec in task_spec.inputs.values() if spec.role == role]

    def _table_records(
        self,
        task_dir: Path,
        task_spec: TaskSpec,
        input_spec,
        *,
        split: str,
    ) -> list[SampleRecord]:
        source = self._source(task_dir, input_spec.source)
        frame = self._read_frame(source, input_spec.format)
        target_field = None
        if split == "train" and task_spec.target is not None:
            target_source = (
                self._source(task_dir, task_spec.target.source)
                if task_spec.target.source
                else source
            )
            if target_source == source:
                target_field = task_spec.target.field
        id_field = (
            input_spec.id_field
            or (
                task_spec.sample_id_field
                if task_spec.sample_id_field in frame.columns
                else None
            )
        )
        configured_features = input_spec.options.get("feature_fields")
        features = (
            [str(value) for value in configured_features]
            if isinstance(configured_features, list)
            else [
                str(column)
                for column in frame.columns
                if str(column) not in {target_field, id_field}
            ]
        )
        records = []
        for position, row in frame.iterrows():
            sample_id = str(row[id_field]) if id_field else f"{split}_{position}"
            inputs = {
                input_spec.name: {
                    field: row[field]
                    for field in features
                    if field in frame.columns
                }
            }
            target = (
                row[target_field]
                if target_field is not None and target_field in frame.columns
                else None
            )
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    inputs=inputs,
                    target=target,
                    split=split,
                )
            )
        return records

    @staticmethod
    def _merge_component_records(
        component_records: list[list[SampleRecord]], *, split: str
    ) -> list[SampleRecord]:
        """Join declared components only when their observed IDs align."""
        if not component_records:
            return []
        indexed = [
            {record.sample_id: record for record in records}
            for records in component_records
        ]
        expected = set(indexed[0])
        for position, records in enumerate(indexed[1:], start=2):
            if set(records) != expected:
                raise ValueError(
                    "declared task-native input components do not have the "
                    f"same observed sample IDs (component {position}); refusing "
                    "to invent a positional or fuzzy join"
                )
        merged: list[SampleRecord] = []
        for sample_id in sorted(expected):
            inputs: dict[str, object] = {}
            targets = []
            for records in indexed:
                record = records[sample_id]
                overlap = set(inputs) & set(record.inputs)
                if overlap:
                    raise ValueError(
                        "task-native input names are not unique: "
                        + ", ".join(sorted(overlap))
                    )
                inputs.update(record.inputs)
                if record.target is not None:
                    targets.append(record.target)
            target = targets[0] if targets else None
            if len(targets) > 1 and any(
                repr(value) != repr(target) for value in targets[1:]
            ):
                raise ValueError(
                    f"conflicting declared targets for sample {sample_id!r}"
                )
            merged.append(
                SampleRecord(
                    sample_id=sample_id,
                    inputs=inputs,
                    target=target,
                    split=split,
                )
            )
        return merged

    def _directory_records(
        self,
        task_dir: Path,
        task_spec: TaskSpec,
        input_spec,
        *,
        split: str,
    ) -> list[SampleRecord]:
        source = self._source(task_dir, input_spec.source)
        files = [path for path in sorted(source.rglob("*")) if path.is_file()]
        if not files:
            raise ValueError(f"declared directory source is empty: {input_spec.source!r}")
        targets: dict[str, Path] = {}
        if split == "train" and task_spec.target is not None and task_spec.target.source:
            target_root = self._source(task_dir, task_spec.target.source)
            if target_root.is_dir():
                for path in sorted(target_root.rglob("*")):
                    if path.is_file():
                        key = path.relative_to(target_root).with_suffix("").as_posix()
                        targets[key] = path
        records = []
        for path in files:
            key = path.relative_to(source).with_suffix("").as_posix()
            target = targets.get(key)
            if split == "train" and task_spec.target is not None and targets and target is None:
                raise ValueError(f"no aligned target was found for {key!r}")
            records.append(
                SampleRecord(
                    sample_id=key,
                    inputs={
                        input_spec.name: canonical_input_reference(task_dir, path)
                    },
                    target=(
                        canonical_input_reference(task_dir, target)
                        if target is not None
                        else None
                    ),
                    split=split,
                )
            )
        return records

    def _records_for(self, task_dir: Path, task_spec: TaskSpec, spec, split: str):
        source = self._source(task_dir, spec.source)
        if source.is_dir():
            return self._directory_records(
                task_dir, task_spec, spec, split=split
            )
        try:
            return self._table_records(task_dir, task_spec, spec, split=split)
        except ValueError as error:
            if "needs a registered reader" not in str(error):
                raise
        target = None
        if split == "train" and task_spec.target is not None:
            target_source = (
                self._source(task_dir, task_spec.target.source)
                if task_spec.target.source
                else None
            )
            if target_source is not None and target_source != source:
                target = canonical_input_reference(task_dir, target_source)
        return [
            SampleRecord(
                sample_id=f"{split}_{source.stem}",
                inputs={
                    spec.name: canonical_input_reference(task_dir, source)
                },
                target=target,
                split=split,
            )
        ]

    def _attach_separate_targets(
        self,
        task_dir: Path,
        task_spec: TaskSpec,
        records: list[SampleRecord],
        train_specs,
    ) -> list[SampleRecord]:
        target_spec = task_spec.target
        if target_spec is None or not target_spec.source:
            return records
        target_source = self._source(task_dir, target_spec.source)
        input_sources = {
            self._source(task_dir, spec.source) for spec in train_specs
        }
        if target_source in input_sources or target_source.is_dir():
            return records
        declared_format = str(
            target_spec.options.get(
                "format", target_source.suffix.lower().lstrip(".")
            )
        )
        try:
            frame = self._read_frame(target_source, declared_format)
        except ValueError as error:
            if len(records) == 1:
                target_value = canonical_input_reference(task_dir, target_source)
                record = records[0]
                return [
                    SampleRecord(
                        sample_id=record.sample_id,
                        inputs=record.inputs,
                        target=target_value,
                        entity_id=record.entity_id,
                        group_id=record.group_id,
                        timestamp=record.timestamp,
                        split=record.split,
                    )
                ]
            raise ValueError(
                "a separate target file for multiple samples needs a declared "
                "row reader and ID alignment"
            ) from error
        target_field = target_spec.field
        if target_field is None or target_field not in frame.columns:
            raise ValueError(
                "declared separate target table does not contain its target field"
            )
        id_field = str(target_spec.options.get("id_field") or "").strip()
        if not id_field and task_spec.sample_id_field in frame.columns:
            id_field = task_spec.sample_id_field
        values: dict[str, object]
        if id_field:
            if id_field not in frame.columns:
                raise ValueError(
                    f"declared target ID field {id_field!r} is absent"
                )
            values = {
                str(row[id_field]): row[target_field]
                for _, row in frame.iterrows()
            }
        elif (
            str(target_spec.options.get("alignment") or "").lower()
            == "position"
            and len(frame) == len(records)
        ):
            values = {
                record.sample_id: frame.iloc[position][target_field]
                for position, record in enumerate(records)
            }
        else:
            raise ValueError(
                "separate target rows need an observed ID field or explicitly "
                "verified positional alignment"
            )
        missing = [record.sample_id for record in records if record.sample_id not in values]
        if missing:
            raise ValueError(
                "separate target table is missing declared training IDs: "
                f"{missing[:5]}"
            )
        return [
            SampleRecord(
                sample_id=record.sample_id,
                inputs=record.inputs,
                target=values[record.sample_id],
                entity_id=record.entity_id,
                group_id=record.group_id,
                timestamp=record.timestamp,
                split=record.split,
            )
            for record in records
        ]

    def build_bundle(self, task_dir: Path, task_spec: TaskSpec) -> DatasetBundle:
        train_specs = self._role_specs(task_spec, "train")
        test_specs = self._role_specs(task_spec, "test")
        if not train_specs:
            raise ValueError(
                "task-native contracts must explicitly identify at least one "
                "source with role='train'"
            )
        records = self._merge_component_records(
            [
                self._records_for(task_dir, task_spec, spec, "train")
                for spec in train_specs
            ],
            split="train",
        )
        records = self._attach_separate_targets(
            task_dir, task_spec, records, train_specs
        )
        if test_specs:
            records.extend(
                self._merge_component_records(
                    [
                        self._records_for(task_dir, task_spec, spec, "test")
                        for spec in test_specs
                    ],
                    split="test",
                )
            )
        return records_to_bundle(
            task_spec,
            records,
            metadata={"index_source": "verified_task_native_contract"},
        )

    def profile(self, task_dir: Path, task_spec: TaskSpec) -> Mapping[str, object]:
        bundle = self.build_bundle(task_dir, task_spec)
        return {
            "schema_version": 1,
            "task_id": task_spec.task_id,
            "index_source": "verified_task_native_contract",
            "train_count": len(bundle.train_records),
            "test_count": len(bundle.test_records),
            "declared_inputs": {
                name: spec.to_dict() for name, spec in task_spec.inputs.items()
            },
            "agent_resolution": self.resolution,
        }
