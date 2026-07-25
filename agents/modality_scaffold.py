"""Deterministic generated-code scaffolds for indexed media and multimodal tasks."""

from __future__ import annotations


def indexed_loader_source() -> str:
    """Return a self-contained loader for TaskAnalyzer's JSONL sample index."""
    return '''"""Harness-generated lazy sample-index loader."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class MyDataLoader:
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

    def _load(self):
        root = Path(__file__).resolve().parent
        task_spec = json.loads(
            (root / "resolved_task_spec.json").read_text(encoding="utf-8")
        )
        index_manifest = json.loads(
            (root / "dataset_index_manifest.json").read_text(encoding="utf-8")
        )
        records = [
            json.loads(line)
            for line in (root / "dataset_index.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        train_records = [
            record for record in records if record["split"] == "train"
        ]
        test_records = [
            record for record in records if record["split"] == "test"
        ]
        X_full = pd.DataFrame(
            [self._feature_row(record) for record in train_records]
        )
        y_full = np.asarray(
            [record.get("target") for record in train_records]
        )
        row_ids_full = np.asarray(
            [record["sample_id"] for record in train_records]
        )
        group_ids_full = np.asarray(
            [
                record.get("group_id")
                or record.get("entity_id")
                or record["sample_id"]
                for record in train_records
            ]
        )
        X_test = pd.DataFrame(
            [self._feature_row(record) for record in test_records]
        )
        test_ids = np.asarray(
            [record["sample_id"] for record in test_records]
        )
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
            "dataset_fingerprint": index_manifest["dataset_fingerprint"],
        }
        self._test_data = {
            "X_test": X_test,
            "test_ids": test_ids,
        }
        self._loaded = True

    def get_data(self):
        if not self._loaded:
            self._load()
        return self._train_data, self._test_data

    def __call__(self):
        return self.get_data()
'''
