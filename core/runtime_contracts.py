"""Runtime data, evaluation, prediction, and model artifact contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import TaskSpec


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SampleRecord:
    """One lazy sample reference; media values are paths, not decoded arrays."""

    sample_id: str
    inputs: Mapping[str, object]
    target: object | None = None
    entity_id: str | None = None
    group_id: str | None = None
    timestamp: object | None = None
    split: str = "train"

    def __post_init__(self) -> None:
        if not str(self.sample_id).strip():
            raise ValueError("sample_id must be non-empty")
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError(
                f"sample {self.sample_id!r} must contain named inputs"
            )
        normalized_split = str(self.split).strip().lower()
        if normalized_split not in {"train", "test"}:
            raise ValueError("sample split must be 'train' or 'test'")
        object.__setattr__(self, "split", normalized_split)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "inputs": dict(self.inputs),
            "target": self.target,
            "entity_id": self.entity_id,
            "group_id": self.group_id,
            "timestamp": self.timestamp,
            "split": self.split,
        }


@dataclass(frozen=True)
class DatasetBundle:
    """Indexed task data shared by analyzers and evaluation runners."""

    task: TaskSpec
    train_records: tuple[SampleRecord, ...]
    test_records: tuple[SampleRecord, ...] = ()
    dataset_fingerprint: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.train_records:
            raise ValueError("dataset bundle requires at least one train record")
        all_records = (*self.train_records, *self.test_records)
        sample_ids = [record.sample_id for record in all_records]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample IDs must be unique across train and test")
        if self.task.target is not None:
            missing = [
                record.sample_id
                for record in self.train_records
                if record.target is None
            ]
            if missing:
                raise ValueError(
                    "supervised train records are missing targets: "
                    f"{missing[:5]}"
                )
        if any(record.split != "train" for record in self.train_records):
            raise ValueError("train_records must use split='train'")
        if any(record.split != "test" for record in self.test_records):
            raise ValueError("test_records must use split='test'")
        if not self.dataset_fingerprint:
            fingerprint_payload = {
                "task": self.task.to_dict(),
                "records": [record.to_dict() for record in all_records],
            }
            object.__setattr__(
                self,
                "dataset_fingerprint",
                _canonical_digest(fingerprint_payload),
            )

    @property
    def train_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.train_records)

    @property
    def test_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.test_records)

    def to_index_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "task_id": self.task.task_id,
            "modality": self.task.modality,
            "problem_type": self.task.problem_type,
            "dataset_fingerprint": self.dataset_fingerprint,
            "train_count": len(self.train_records),
            "test_count": len(self.test_records),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SplitPlan:
    """Harness-owned fold assignments keyed by stable sample ID."""

    assignments: Mapping[str, int]
    strategy: str
    seed: int
    leakage_unit: str = "sample_id"
    group_field: str | None = None
    split_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.assignments:
            raise ValueError("split plan assignments may not be empty")
        normalized = {
            str(sample_id): int(fold)
            for sample_id, fold in self.assignments.items()
        }
        if any(fold < 0 for fold in normalized.values()):
            raise ValueError("fold IDs must be non-negative")
        object.__setattr__(self, "assignments", normalized)
        if not str(self.strategy).strip():
            raise ValueError("split strategy must be non-empty")
        if not self.split_fingerprint:
            object.__setattr__(
                self,
                "split_fingerprint",
                _canonical_digest(
                    {
                        "assignments": sorted(normalized.items()),
                        "strategy": self.strategy,
                        "seed": self.seed,
                        "leakage_unit": self.leakage_unit,
                        "group_field": self.group_field,
                    }
                ),
            )

    @property
    def folds(self) -> int:
        return len(set(self.assignments.values()))

    def to_dict(self) -> dict[str, object]:
        return {
            "assignments": dict(self.assignments),
            "strategy": self.strategy,
            "seed": self.seed,
            "leakage_unit": self.leakage_unit,
            "group_field": self.group_field,
            "folds": self.folds,
            "split_fingerprint": self.split_fingerprint,
        }


@dataclass(frozen=True)
class FidelityProfile:
    """Registered multi-dimensional evaluation fidelity."""

    name: str
    sample_fraction: float
    folds: int
    max_epochs: int
    max_trials: int
    early_stopping_patience: int
    max_estimator_iterations: int
    spatial_size: tuple[int, int] | None = None
    audio_sample_rate: int | None = None
    max_audio_seconds: float | None = None
    video_frames: int | None = None
    video_fps: float | None = None
    clips_per_video: int | None = None

    def __post_init__(self) -> None:
        if not 0 < float(self.sample_fraction) <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        for field_name in (
            "folds",
            "max_epochs",
            "max_trials",
            "early_stopping_patience",
            "max_estimator_iterations",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be positive")

    def to_dict(self) -> dict[str, object]:
        values: dict[str, object] = {
            "name": self.name,
            "sample_fraction": self.sample_fraction,
            "folds": self.folds,
            "max_epochs": self.max_epochs,
            "max_trials": self.max_trials,
            "early_stopping_patience": self.early_stopping_patience,
            "max_estimator_iterations": self.max_estimator_iterations,
        }
        # Retain old constructor fields for external API compatibility, but do
        # not expose category-specific hints unless a caller explicitly set
        # one from task evidence.
        optional = {
            "spatial_size": (
                list(self.spatial_size) if self.spatial_size else None
            ),
            "audio_sample_rate": self.audio_sample_rate,
            "max_audio_seconds": self.max_audio_seconds,
            "video_frames": self.video_frames,
            "video_fps": self.video_fps,
            "clips_per_video": self.clips_per_video,
        }
        values.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return values


@dataclass(frozen=True)
class PredictionBundle:
    """Typed prediction payload manifest."""

    task_fingerprint: str
    split_fingerprint: str
    output_type: str
    sample_ids: tuple[str, ...]
    payload_path: str
    payload_format: str
    class_names: tuple[str, ...] = ()
    target_path: str | None = None
    fold_ids_path: str | None = None
    schema_version: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("prediction schema_version must be positive")
        if not self.task_fingerprint or not self.split_fingerprint:
            raise ValueError(
                "prediction bundle requires task and split fingerprints"
            )
        if not self.sample_ids or len(set(self.sample_ids)) != len(
            self.sample_ids
        ):
            raise ValueError("prediction sample IDs must be non-empty and unique")
        if not self.payload_path or not self.payload_format:
            raise ValueError("prediction payload path/format are required")

    @property
    def compatibility_key(self) -> str:
        return _canonical_digest(
            {
                "task_fingerprint": self.task_fingerprint,
                "split_fingerprint": self.split_fingerprint,
                "output_type": self.output_type,
                "class_names": self.class_names,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "task_fingerprint": self.task_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "output_type": self.output_type,
            "sample_ids": list(self.sample_ids),
            "payload_path": self.payload_path,
            "payload_format": self.payload_format,
            "class_names": list(self.class_names),
            "target_path": self.target_path,
            "fold_ids_path": self.fold_ids_path,
            "compatibility_key": self.compatibility_key,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "PredictionBundle":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            task_fingerprint=str(raw.get("task_fingerprint") or ""),
            split_fingerprint=str(raw.get("split_fingerprint") or ""),
            output_type=str(raw.get("output_type") or ""),
            sample_ids=tuple(str(item) for item in raw.get("sample_ids", [])),
            payload_path=str(raw.get("payload_path") or ""),
            payload_format=str(raw.get("payload_format") or ""),
            class_names=tuple(
                str(item) for item in raw.get("class_names", [])
            ),
            target_path=(
                str(raw["target_path"])
                if raw.get("target_path") is not None
                else None
            ),
            fold_ids_path=(
                str(raw["fold_ids_path"])
                if raw.get("fold_ids_path") is not None
                else None
            ),
            metadata=(
                dict(raw.get("metadata", {}))
                if isinstance(raw.get("metadata", {}), Mapping)
                else {}
            ),
        )


@dataclass(frozen=True)
class ResultRecord:
    """Normalized measured result consumed by shared search policies."""

    status: str
    task_fingerprint: str
    split_fingerprint: str
    modality: str
    problem_type: str
    output_type: str
    primary_metric: str
    direction: str
    score: float
    cv_mean: float
    cv_std: float
    folds: int
    fidelity: str
    runtime_seconds: float = 0.0
    peak_ram_gb: float | None = None
    peak_vram_gb: float | None = None
    prediction_bundle: str | None = None
    model_bundle: str | None = None
    schema_version: int = 2
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("score", "cv_mean", "cv_std", "runtime_seconds"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError("result direction must be maximize or minimize")
        if self.folds < 1:
            raise ValueError("result folds must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ModelBundle:
    """Reloadable inference artifact produced by one model node."""

    model_family: str
    task_fingerprint: str
    output_type: str
    checkpoint_paths: tuple[str, ...]
    entrypoint: str
    dependencies: tuple[str, ...] = ()
    preprocessing: str | None = None
    bundle_type: str = "model"
    schema_version: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bundle_type != "model":
            raise ValueError("ModelBundle bundle_type must be 'model'")
        if not self.model_family or not self.task_fingerprint:
            raise ValueError("model family and task fingerprint are required")
        if not self.entrypoint:
            raise ValueError("model entrypoint is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_type": self.bundle_type,
            "model_family": self.model_family,
            "task_fingerprint": self.task_fingerprint,
            "output_type": self.output_type,
            "checkpoint_paths": list(self.checkpoint_paths),
            "preprocessing": self.preprocessing,
            "entrypoint": self.entrypoint,
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EnsembleBundle:
    """Manager-owned model ensemble referencing measured component bundles."""

    strategy: str
    component_nodes: tuple[str, ...]
    component_bundles: tuple[str, ...]
    output_type: str
    compatibility_key: str
    weights: tuple[float, ...]
    combiner_path: str | None = None
    inference_order: tuple[str, ...] = ()
    bundle_type: str = "ensemble"
    schema_version: int = 1
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bundle_type != "ensemble":
            raise ValueError("EnsembleBundle bundle_type must be 'ensemble'")
        if len(self.component_nodes) < 2:
            raise ValueError("ensemble requires at least two component nodes")
        if len(self.component_nodes) != len(self.component_bundles):
            raise ValueError(
                "ensemble nodes and component bundle paths must align"
            )
        if len(self.weights) != len(self.component_nodes):
            raise ValueError("ensemble weights must align with components")
        if any(weight < 0 or not math.isfinite(weight) for weight in self.weights):
            raise ValueError("ensemble weights must be finite and non-negative")
        if sum(self.weights) <= 0:
            raise ValueError("ensemble weights must have positive mass")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "bundle_type": self.bundle_type,
            "strategy": self.strategy,
            "component_nodes": list(self.component_nodes),
            "component_bundles": list(self.component_bundles),
            "output_type": self.output_type,
            "compatibility_key": self.compatibility_key,
            "weights": list(self.weights),
            "combiner_path": self.combiner_path,
            "inference_order": list(self.inference_order),
            "metadata": dict(self.metadata),
        }
