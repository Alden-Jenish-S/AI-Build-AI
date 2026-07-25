"""Text adapter used directly or as a multimodal component."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord
from .common import (
    read_manifest,
    read_task_config,
    records_to_bundle,
    render_profile_report,
    resolve_task_source,
    row_split,
)


class TextAdapter:
    name = "text"

    def discover(self, task_dir: Path) -> TaskSpec:
        config = read_task_config(task_dir)
        return TaskSpec.from_mapping(Path(task_dir).name, config)

    def build_bundle(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> DatasetBundle:
        input_spec = next(
            item
            for item in task_spec.inputs.values()
            if item.modality == "text"
        )
        source = resolve_task_source(task_dir, input_spec.source)
        frame = read_manifest(source)
        text_field = str(
            input_spec.options.get("text_field", input_spec.name)
        )
        if text_field not in frame.columns:
            for candidate in ("text", "content", "document", "caption"):
                if candidate in frame.columns:
                    text_field = candidate
                    break
            else:
                raise ValueError("text manifest does not contain a text field")
        target_field = (
            task_spec.target.field if task_spec.target else None
        )
        split_field = str(
            input_spec.options.get("split_field", "split")
        )
        records = []
        for index, series in frame.iterrows():
            row = series.to_dict()
            sample_id = str(
                row.get(task_spec.sample_id_field, index)
            )
            split = row_split(
                row,
                split_field=split_field,
                target_field=target_field,
            )
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    inputs={input_spec.name: str(row[text_field])},
                    target=(
                        None
                        if split == "test" or target_field is None
                        else row.get(target_field)
                    ),
                    entity_id=sample_id,
                    group_id=(
                        str(row.get(task_spec.group_id_field))
                        if task_spec.group_id_field
                        and row.get(task_spec.group_id_field) is not None
                        else None
                    ),
                    split=split,
                )
            )
        return records_to_bundle(task_spec, records)

    def profile(
        self, task_dir: Path, task_spec: TaskSpec
    ) -> dict[str, object]:
        bundle = self.build_bundle(task_dir, task_spec)
        lengths = [
            len(str(next(iter(record.inputs.values()))))
            for record in bundle.train_records
        ]
        return {
            **bundle.to_index_dict(),
            "character_length": {
                "min": min(lengths) if lengths else None,
                "max": max(lengths) if lengths else None,
                "mean": sum(lengths) / len(lengths) if lengths else None,
            },
            "target_distribution": dict(
                Counter(str(record.target) for record in bundle.train_records)
            ),
        }

    def render_report(
        self, task_dir: Path, task_spec: TaskSpec | None = None
    ) -> str:
        task = task_spec or self.discover(task_dir)
        return render_profile_report(task, self.profile(task_dir, task))
