"""Deterministic task-data scaffolds shared by every method-tree node."""

from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from typing import Any, Mapping


def _flatten_feature_row(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Mirror ``TaskDataLoader._feature_row`` without loading the dataset."""
    result: dict[str, Any] = {}
    for input_name, value in inputs.items():
        if isinstance(value, Mapping):
            for field, field_value in value.items():
                result[f"{input_name}__{field}"] = field_value
        else:
            result[input_name] = value
    return result


def _direct_tabular_runtime_contract(
    task_spec_path: Path, task_dir: Path
) -> dict[str, Any]:
    spec = json.loads(Path(task_spec_path).read_text(encoding="utf-8"))
    train_input = next(
        (
            item
            for item in spec.get("inputs", {}).values()
            if item.get("role") in {"train", "data"}
        ),
        None,
    )
    if not isinstance(train_input, Mapping):
        raise ValueError("tabular task contract has no train/data input")
    source = Path(str(train_input.get("source") or ""))
    candidates = (
        [source]
        if source.is_absolute()
        else [Path(task_dir) / source, Path(task_dir) / "input" / source]
    )
    train_path = next((path for path in candidates if path.is_file()), None)
    if train_path is None:
        raise FileNotFoundError(
            f"could not resolve tabular source {source!s}"
        )
    delimiter = "\t" if train_path.suffix.lower() == ".tsv" else ","
    with train_path.open("r", encoding="utf-8", newline="") as stream:
        columns = next(csv.reader(stream, delimiter=delimiter), [])
    target = spec.get("target") or {}
    excluded = {
        value
        for value in (
            target.get("field") if isinstance(target, Mapping) else None,
            spec.get("sample_id_field"),
            spec.get("entity_id_field"),
            spec.get("group_id_field"),
        )
        if value
    }
    feature_columns = [
        str(column) for column in columns if column not in excluded
    ]
    return {
        "schema_version": 2,
        "storage": "direct_tabular",
        "feature_container": "pandas.DataFrame",
        "feature_variables": [
            "X_train",
            "X_valid",
            "X_eval",
            "X_test",
        ],
        "feature_columns": feature_columns,
        "components": {
            "tabular": {
                "storage": "dataframe_columns",
                "columns": feature_columns,
            }
        },
        "array_variables": {
            "y_train": "numpy.ndarray",
            "y_valid": "numpy.ndarray",
            "y_eval": "numpy.ndarray",
            "row_ids": "numpy.ndarray",
            "fold_ids": "numpy.ndarray",
            "test_ids": "numpy.ndarray",
        },
        "indexing_rules": {
            "features": "Use .iloc for positional DataFrame row selection.",
            "arrays": "Use array[mask] or np.asarray(array)[mask]; never use .iloc or .loc.",
            "components": (
                "Tabular features retain their original CSV column names in "
                "one DataFrame; identity and target columns are excluded."
            ),
        },
    }


def build_runtime_data_contract(
    index_path: Path,
    *,
    task_spec_path: Path | None = None,
    task_dir: Path | None = None,
) -> dict[str, Any]:
    """Describe the concrete objects returned by the generated task loader."""
    if not Path(index_path).is_file():
        if task_spec_path is None or task_dir is None:
            raise FileNotFoundError(
                "dataset index is absent; direct tabular contracts require "
                "task_spec_path and task_dir"
            )
        return _direct_tabular_runtime_contract(
            Path(task_spec_path), Path(task_dir)
        )
    examples: dict[str, dict[str, Any]] = {}
    with Path(index_path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            split = str(record.get("split") or "")
            if split in {"train", "test"} and split not in examples:
                examples[split] = record
            if len(examples) == 2:
                break
    if "train" not in examples:
        raise ValueError("dataset index contains no training record")

    representative = examples["train"].get("inputs", {})
    if not isinstance(representative, Mapping):
        raise ValueError("dataset index record inputs must be an object")
    flat_row = _flatten_feature_row(representative)
    components = {}
    for name, value in representative.items():
        if isinstance(value, Mapping):
            columns = [
                column
                for column in flat_row
                if column.startswith(f"{name}__")
            ]
            components[name] = {
                "storage": "flattened_columns",
                "column_prefix": f"{name}__",
                "columns": columns,
            }
        else:
            components[name] = {
                "storage": "single_column",
                "columns": [name],
            }

    return {
        "schema_version": 1,
        "feature_container": "pandas.DataFrame",
        "feature_variables": ["X_eval", "X_test"],
        "feature_columns": list(flat_row),
        "components": components,
        "array_variables": {
            "y_eval": "numpy.ndarray",
            "row_ids": "numpy.ndarray",
            "fold_ids": "numpy.ndarray",
            "test_ids": "numpy.ndarray",
        },
        "indexing_rules": {
            "features": "Use .iloc for positional DataFrame row selection.",
            "arrays": "Use array[mask] or np.asarray(array)[mask]; never use .iloc or .loc.",
            "components": (
                "Nested input objects are not retained. Select a single-column "
                "component by its column name and a flattened component by its "
                "declared column prefix."
            ),
        },
    }


def write_runtime_data_contract(
    index_path: Path,
    output_path: Path,
    *,
    task_spec_path: Path | None = None,
    task_dir: Path | None = None,
) -> dict[str, Any]:
    contract = build_runtime_data_contract(
        index_path,
        task_spec_path=task_spec_path,
        task_dir=task_dir,
    )
    Path(output_path).write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract


def runtime_data_prompt(contract: Mapping[str, Any]) -> str:
    """Render a compact, exact schema for implementation and repair prompts."""
    return (
        "Concrete TaskDataLoader runtime objects (authoritative):\n"
        + json.dumps(dict(contract), indent=2, sort_keys=True)
        + "\nDo not infer a different container or nested component layout."
    )


def task_loader_source() -> str:
    """Return a self-contained loader for TaskAnalyzer's JSONL sample index."""
    return '''"""Harness-generated lazy task-data loader."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


class TaskDataLoader:
    def __init__(self):
        self._loaded = False
        self._train_data = None
        self._test_data = None

    @staticmethod
    def _feature_row(record):
        result = {}
        for input_name, value in record["inputs"].items():
            if isinstance(value, dict):
                for field, field_value in value.items():
                    result[f"{input_name}__{field}"] = field_value
            else:
                result[input_name] = value
        return result

    @staticmethod
    def _input_path(root, source):
        relative = Path(str(source))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe task input source: {source!r}")
        parts = relative.parts
        if parts and parts[0] == "input":
            relative = Path(*parts[1:])
        candidate = root / "input" / relative
        if not candidate.is_file():
            raise FileNotFoundError(
                f"task input is not linked into the node: {relative}"
            )
        return candidate

    @staticmethod
    def _read_table(path):
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        return pd.read_csv(path, sep=separator)

    def _load_direct_tabular(self, root, task_spec, index_manifest):
        inputs = task_spec.get("inputs", {})
        train_spec = next(
            (
                item for item in inputs.values()
                if item.get("role") in {"train", "data"}
            ),
            None,
        )
        test_spec = next(
            (
                item for item in inputs.values()
                if item.get("role") == "test"
            ),
            None,
        )
        if train_spec is None:
            raise ValueError("tabular task has no train/data input")
        train_frame = self._read_table(
            self._input_path(root, train_spec["source"])
        )
        test_frame = (
            self._read_table(self._input_path(root, test_spec["source"]))
            if test_spec is not None
            else pd.DataFrame()
        )
        target_spec = task_spec.get("target") or {}
        target_field = target_spec.get("field")
        unsupervised = task_spec.get("problem_type") == "unsupervised_clustering"
        if not unsupervised and (
            not target_field or target_field not in train_frame.columns
        ):
            raise ValueError(
                "supervised tabular target is missing from the training table"
            )
        y_full = (
            None
            if unsupervised
            else train_frame[target_field].to_numpy(copy=True)
        )
        sample_id_field = task_spec.get("sample_id_field")
        entity_id_field = task_spec.get("entity_id_field")
        group_id_field = task_spec.get("group_id_field")
        row_ids_full = (
            train_frame[sample_id_field].to_numpy(copy=True)
            if sample_id_field in train_frame.columns
            else np.arange(len(train_frame))
        )
        grouping_field = (
            group_id_field
            if group_id_field in train_frame.columns
            else entity_id_field
            if entity_id_field in train_frame.columns
            else None
        )
        group_ids_full = (
            train_frame[grouping_field].to_numpy(copy=True)
            if grouping_field
            else row_ids_full.copy()
        )
        excluded = {
            field
            for field in (
                target_field,
                sample_id_field,
                entity_id_field,
                group_id_field,
            )
            if field
        }
        feature_columns = [
            column for column in train_frame.columns
            if column not in excluded
        ]
        X_full = train_frame[feature_columns].copy()
        if test_frame.empty and test_spec is None:
            X_test = pd.DataFrame(columns=feature_columns)
            test_ids = np.asarray([], dtype=object)
        else:
            missing = [
                column for column in feature_columns
                if column not in test_frame.columns
            ]
            if missing:
                raise ValueError(
                    f"test table is missing training features: {missing}"
                )
            X_test = test_frame[feature_columns].copy()
            test_ids = (
                test_frame[sample_id_field].to_numpy(copy=True)
                if sample_id_field in test_frame.columns
                else np.arange(len(test_frame))
            )
        empty_y = (
            None if y_full is None else y_full[0:0].copy()
        )
        self._train_data = {
            "X": X_full.copy(),
            "y": None if y_full is None else y_full.copy(),
            "row_ids": row_ids_full.copy(),
            "X_val": X_full.iloc[0:0].copy(),
            "y_val": empty_y,
            "val_row_ids": row_ids_full[0:0].copy(),
            "X_full": X_full,
            "y_full": y_full,
            "row_ids_full": row_ids_full,
            "group_ids": group_ids_full.copy(),
            "group_ids_full": group_ids_full,
            "has_val": False,
            "cat_cols": [
                column for column in X_full.columns
                if not pd.api.types.is_numeric_dtype(X_full[column])
            ],
            "cont_cols": [
                column for column in X_full.columns
                if pd.api.types.is_numeric_dtype(X_full[column])
            ],
            "cat_dims": [],
            "n_cont": sum(
                pd.api.types.is_numeric_dtype(X_full[column])
                for column in X_full.columns
            ),
            "task_type": task_spec["problem_type"],
            "modality": task_spec["modality"],
            "component_modalities": task_spec["component_modalities"],
            "output_type": task_spec["output"]["type"],
            "class_names": task_spec["output"].get("class_names", []),
            "dataset_fingerprint": index_manifest["dataset_fingerprint"],
        }
        self._test_data = {
            "X_test": X_test,
            "test_ids": test_ids,
            "class_names": task_spec["output"].get("class_names", []),
        }
        self._loaded = True

    def _load(self):
        root = Path(__file__).resolve().parent
        assets_root = Path(
            os.environ.get("AIBUILDAI_TASK_ASSETS_DIR", str(root))
        ).resolve()
        task_spec = json.loads(
            (assets_root / "resolved_task_spec.json").read_text(
                encoding="utf-8"
            )
        )
        index_manifest = json.loads(
            (assets_root / "dataset_index_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        if index_manifest.get("storage") == "direct_tabular":
            self._load_direct_tabular(root, task_spec, index_manifest)
            return
        train_rows = []
        targets = []
        train_ids = []
        group_ids = []
        test_rows = []
        test_id_values = []
        with (assets_root / "dataset_index.jsonl").open(
            "r", encoding="utf-8"
        ) as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["split"] == "train":
                    train_rows.append(self._feature_row(record))
                    targets.append(record.get("target"))
                    train_ids.append(record["sample_id"])
                    group_id = record.get("group_id")
                    if group_id is None:
                        group_id = record.get("entity_id")
                    if group_id is None:
                        group_id = record["sample_id"]
                    group_ids.append(group_id)
                elif record["split"] == "test":
                    test_rows.append(self._feature_row(record))
                    test_id_values.append(record["sample_id"])
        X_full = pd.DataFrame(train_rows)
        y_full = np.asarray(targets)
        row_ids_full = np.asarray(train_ids)
        group_ids_full = np.asarray(group_ids)
        X_test = pd.DataFrame(test_rows)
        test_ids = np.asarray(test_id_values)
        self._train_data = {
            "X": X_full.copy(),
            "y": y_full.copy(),
            "row_ids": row_ids_full.copy(),
            "X_val": X_full.iloc[0:0].copy(),
            "y_val": y_full[0:0].copy(),
            "val_row_ids": row_ids_full[0:0].copy(),
            "X_full": X_full,
            "y_full": y_full,
            "row_ids_full": row_ids_full,
            "group_ids": group_ids_full.copy(),
            "group_ids_full": group_ids_full,
            "has_val": False,
            "cat_cols": [
                column
                for column in X_full.columns
                if not pd.api.types.is_numeric_dtype(X_full[column])
            ],
            "cont_cols": [
                column
                for column in X_full.columns
                if pd.api.types.is_numeric_dtype(X_full[column])
            ],
            "cat_dims": [],
            "n_cont": sum(
                pd.api.types.is_numeric_dtype(X_full[column])
                for column in X_full.columns
            ),
            "task_type": task_spec["problem_type"],
            "modality": task_spec["modality"],
            "component_modalities": task_spec["component_modalities"],
            "output_type": task_spec["output"]["type"],
            "class_names": task_spec["output"].get("class_names", []),
            "dataset_fingerprint": index_manifest["dataset_fingerprint"],
        }
        self._test_data = {
            "X_test": X_test,
            "test_ids": test_ids,
            "class_names": task_spec["output"].get("class_names", []),
        }
        self._loaded = True

    def get_data(self):
        if not self._loaded:
            self._load()
        return self._train_data, self._test_data

    def __call__(self):
        return self.get_data()
'''
