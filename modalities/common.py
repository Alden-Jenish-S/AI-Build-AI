"""Shared manifest and path helpers for modality adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from core.contracts import TaskSpec
from core.runtime_contracts import DatasetBundle, SampleRecord


def read_task_config(task_dir: Path) -> dict[str, object]:
    path = Path(task_dir) / "task_config.json"
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("task_config.json must contain a JSON object")
    return loaded


def task_input_root(task_dir: Path) -> Path:
    task_dir = Path(task_dir).resolve()
    candidate = task_dir / "input"
    return candidate if candidate.is_dir() else task_dir


def resolve_task_source(
    task_dir: Path,
    source: str | Path,
    *,
    relative_to: Path | None = None,
) -> Path:
    """Resolve a configured source while rejecting task-directory escapes."""
    task_root = Path(task_dir).resolve()
    input_root = task_input_root(task_root)
    raw = Path(str(source))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        parts = raw.parts
        if parts and parts[0] == "input":
            candidates.append(input_root.joinpath(*parts[1:]))
        if relative_to is not None:
            candidates.append(Path(relative_to) / raw)
        candidates.extend((input_root / raw, task_root / raw))
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved != task_root and task_root not in resolved.parents:
            continue
        if resolved.exists():
            return resolved
    display = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"configured task source {source!r} was not found within "
        f"{task_root}; checked {display}"
    )


def canonical_input_reference(task_dir: Path, path: Path) -> str:
    """Return the run-local path created by ``expose_task_data``."""
    root = task_input_root(task_dir)
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"input path escapes task input root: {path}") from exc
    return (Path("input") / relative).as_posix()


def read_manifest(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            loaded = loaded.get("records", loaded.get("data", loaded))
        return pd.DataFrame(loaded)
    if suffix in {".parquet", ".feather"}:
        return (
            pd.read_parquet(path)
            if suffix == ".parquet"
            else pd.read_feather(path)
        )
    raise ValueError(f"unsupported manifest format: {path}")


def json_safe(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if pd.isna(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [json_safe(item) for item in value]
    return str(value)


def row_split(
    row: Mapping[str, object],
    *,
    split_field: str | None,
    target_field: str | None,
    default: str = "train",
) -> str:
    if split_field and split_field in row:
        value = str(row.get(split_field) or "").strip().lower()
        if value in {"test", "holdout", "inference", "predict"}:
            return "test"
        if value in {"train", "training", "validation", "valid", "val"}:
            return "train"
    if target_field and target_field in row and pd.isna(row[target_field]):
        return "test"
    return default


def records_to_bundle(
    task: TaskSpec,
    records: Iterable[SampleRecord],
    *,
    metadata: Mapping[str, object] | None = None,
) -> DatasetBundle:
    train = tuple(record for record in records if record.split == "train")
    test = tuple(record for record in records if record.split == "test")
    return DatasetBundle(
        task=task,
        train_records=train,
        test_records=test,
        metadata=dict(metadata or {}),
    )


def render_profile_report(
    task: TaskSpec, profile: Mapping[str, object]
) -> str:
    return (
        "=== AUTOMATIC DATASET ANALYSIS REPORT ===\n"
        f"Resolved modality: {task.modality}\n"
        f"Component modalities: {list(task.component_modalities)}\n"
        f"Problem type: {task.problem_type}\n"
        f"Output type: {task.output.type}\n"
        f"Primary metric: {task.primary_metric} "
        f"({task.metric_direction})\n"
        "Structured Profile:\n"
        f"{json.dumps(dict(profile), indent=2, default=str)}\n"
        "========================================="
    )
