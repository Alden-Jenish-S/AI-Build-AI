"""Validated, modality-neutral task contracts.

The current runtime still accepts legacy tabular ``task_config.json`` files.
``TaskSpec.from_mapping`` translates them into the same canonical contract used
by future modality adapters, keeping configuration parsing out of the search
scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluation.metrics import (
    canonical_metric_name,
    infer_metric_direction,
    resolve_metric_name,
)


_MODALITY_ALIASES = {
    "images": "image",
    "vision": "image",
    "waveform": "audio",
    "videos": "video",
    "multi_modal": "multimodal",
    "multi_modality": "multimodal",
}
_SUPPORTED_MODALITIES = {
    "audio",
    "image",
    "multimodal",
    "tabular",
    "text",
    "video",
}
_PROBLEM_TYPE_ALIASES = {
    "binary_classification": "classification",
    "clustering": "unsupervised_clustering",
    "multiclass_classification": "classification",
    "unsupervised": "unsupervised_clustering",
}
_SUPPORTED_PROBLEM_TYPES = {
    "captioning",
    "classification",
    "detection",
    "multilabel_classification",
    "regression",
    "retrieval",
    "segmentation",
    "supervised",
    "temporal_localization",
    "unsupervised_clustering",
}
_SUPPORTED_OUTPUT_TYPES = {
    "boxes",
    "class_probabilities",
    "continuous",
    "embeddings",
    "labels",
    "masks",
    "ranked_items",
    "text",
}
_SUPPORTED_DIRECTIONS = {"maximize", "minimize"}


def _normalized_identifier(value: object, field_name: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def normalize_modality(value: object) -> str:
    """Return a canonical modality name and reject unsupported values."""
    normalized = _normalized_identifier(value, "modality")
    normalized = _MODALITY_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_MODALITIES:
        raise ValueError(
            "modality must be one of "
            f"{sorted(_SUPPORTED_MODALITIES)}; got {value!r}"
        )
    return normalized


def normalize_problem_type(value: object) -> str:
    """Return a canonical learning objective independent of modality."""
    normalized = _normalized_identifier(value, "problem_type")
    normalized = _PROBLEM_TYPE_ALIASES.get(normalized, normalized)
    if normalized not in _SUPPORTED_PROBLEM_TYPES:
        raise ValueError(
            "problem_type must be one of "
            f"{sorted(_SUPPORTED_PROBLEM_TYPES)}; got {value!r}"
        )
    return normalized


def _default_output_type(problem_type: str) -> str:
    if problem_type == "regression":
        return "continuous"
    if problem_type == "unsupervised_clustering":
        return "labels"
    if problem_type == "segmentation":
        return "masks"
    if problem_type == "detection":
        return "boxes"
    if problem_type == "retrieval":
        return "ranked_items"
    if problem_type == "captioning":
        return "text"
    return "class_probabilities"


def _inferred_format(source: str) -> str:
    suffix = Path(source).suffix.lower().lstrip(".")
    return suffix or "file"


@dataclass(frozen=True)
class InputSpec:
    """One named task input and its modality-specific options."""

    name: str
    modality: str
    role: str
    source: str
    format: str
    id_field: str | None = None
    required: bool = True
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("input name must be non-empty")
        object.__setattr__(self, "modality", normalize_modality(self.modality))
        if not str(self.role).strip():
            raise ValueError(f"input {self.name!r} role must be non-empty")
        if not str(self.source).strip():
            raise ValueError(f"input {self.name!r} source must be non-empty")
        if not str(self.format).strip():
            raise ValueError(f"input {self.name!r} format must be non-empty")
        if not isinstance(self.options, Mapping):
            raise ValueError(f"input {self.name!r} options must be an object")

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: object,
        *,
        default_modality: str,
    ) -> "InputSpec":
        if isinstance(raw, str):
            mapping: dict[str, object] = {"source": raw}
        elif isinstance(raw, Mapping):
            mapping = dict(raw)
        else:
            raise ValueError(f"input {name!r} must be a path or object")

        source = str(mapping.get("source") or "").strip()
        known = {
            "format",
            "id_field",
            "modality",
            "name",
            "options",
            "required",
            "role",
            "source",
        }
        options = mapping.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError(f"input {name!r} options must be an object")
        merged_options = dict(options)
        merged_options.update(
            {key: value for key, value in mapping.items() if key not in known}
        )
        return cls(
            name=str(name),
            modality=normalize_modality(
                mapping.get("modality", default_modality)
            ),
            role=str(mapping.get("role") or name),
            source=source,
            format=str(mapping.get("format") or _inferred_format(source)),
            id_field=(
                str(mapping["id_field"])
                if mapping.get("id_field") is not None
                else None
            ),
            required=bool(mapping.get("required", True)),
            options=merged_options,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "modality": self.modality,
            "role": self.role,
            "source": self.source,
            "format": self.format,
            "id_field": self.id_field,
            "required": self.required,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class TargetSpec:
    """Target location and semantic type for a supervised task."""

    field: str
    source: str | None = None
    type: str | None = None

    def __post_init__(self) -> None:
        if not str(self.field).strip():
            raise ValueError("target field must be non-empty")

    @classmethod
    def from_value(
        cls,
        raw: object,
        *,
        default_source: str | None = None,
    ) -> "TargetSpec | None":
        if raw is None:
            return None
        if isinstance(raw, str):
            return cls(field=raw, source=default_source)
        if not isinstance(raw, Mapping):
            raise ValueError("target must be a field name or object")
        field_name = raw.get("field") or raw.get("column")
        if field_name is None:
            return None
        return cls(
            field=str(field_name),
            source=(
                str(raw["source"])
                if raw.get("source") is not None
                else default_source
            ),
            type=(
                str(raw["type"]) if raw.get("type") is not None else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "source": self.source,
            "type": self.type,
        }


@dataclass(frozen=True)
class OutputSpec:
    """Normalized prediction/output shape expected from every candidate."""

    type: str
    class_names: tuple[str, ...] = ()
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = _normalized_identifier(self.type, "output.type")
        if normalized not in _SUPPORTED_OUTPUT_TYPES:
            raise ValueError(
                "output.type must be one of "
                f"{sorted(_SUPPORTED_OUTPUT_TYPES)}; got {self.type!r}"
            )
        object.__setattr__(self, "type", normalized)
        object.__setattr__(
            self, "class_names", tuple(str(item) for item in self.class_names)
        )
        if not isinstance(self.options, Mapping):
            raise ValueError("output options must be an object")

    @classmethod
    def from_value(
        cls, raw: object, *, problem_type: str
    ) -> "OutputSpec":
        if raw is None:
            return cls(type=_default_output_type(problem_type))
        if isinstance(raw, str):
            return cls(type=raw)
        if not isinstance(raw, Mapping):
            raise ValueError("output must be a type name or object")
        output_type = raw.get("type") or raw.get("kind")
        if output_type is None:
            output_type = _default_output_type(problem_type)
        known = {"type", "kind", "class_names", "options"}
        options = raw.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError("output options must be an object")
        merged_options = dict(options)
        merged_options.update(
            {key: value for key, value in raw.items() if key not in known}
        )
        return cls(
            type=str(output_type),
            class_names=tuple(str(item) for item in raw.get("class_names", [])),
            options=merged_options,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "class_names": list(self.class_names),
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class MetricSpec:
    """A named evaluation metric with explicit optimization direction."""

    name: str
    direction: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = canonical_metric_name(self.name)
        if not name:
            raise ValueError("metric name must be non-empty")
        object.__setattr__(self, "name", name)
        direction = str(self.direction).strip().lower()
        if direction not in _SUPPORTED_DIRECTIONS:
            raise ValueError(
                "metric direction must be 'maximize' or 'minimize'; "
                f"got {self.direction!r}"
            )
        object.__setattr__(self, "direction", direction)
        if not isinstance(self.options, Mapping):
            raise ValueError("metric options must be an object")

    @classmethod
    def from_value(
        cls, raw: object, *, default_direction: str | None = None
    ) -> "MetricSpec":
        if isinstance(raw, str):
            return cls(
                name=raw,
                direction=default_direction or infer_metric_direction(raw),
            )
        if not isinstance(raw, Mapping):
            raise ValueError("each metric must be a name or object")
        name = raw.get("name")
        if name is None:
            raise ValueError("each metric object must define name")
        known = {"name", "direction", "options"}
        options = raw.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError("metric options must be an object")
        merged_options = dict(options)
        merged_options.update(
            {key: value for key, value in raw.items() if key not in known}
        )
        return cls(
            name=str(name),
            direction=str(
                raw.get("direction")
                or default_direction
                or infer_metric_direction(name)
            ),
            options=merged_options,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "direction": self.direction,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class ResourceLimits:
    """Task-owned feasibility limits, independent of any model library."""

    preferred_accelerator: str = "auto"
    accelerators: tuple[str, ...] = ()
    max_ram_gb: float | None = None
    max_vram_gb: float | None = None
    max_cache_gb: float | None = None
    max_decode_workers: int | None = None

    def __post_init__(self) -> None:
        preference = str(self.preferred_accelerator).strip().lower()
        if preference not in {"auto", "cpu", "cuda", "gpu", "mps"}:
            raise ValueError(
                "preferred_accelerator must be auto, cpu, cuda, gpu, or mps"
            )
        object.__setattr__(self, "preferred_accelerator", preference)
        object.__setattr__(
            self,
            "accelerators",
            tuple(str(item).strip().lower() for item in self.accelerators),
        )
        for field_name in ("max_ram_gb", "max_vram_gb", "max_cache_gb"):
            value = getattr(self, field_name)
            if value is not None and float(value) <= 0:
                raise ValueError(f"{field_name} must be positive when provided")
        if self.max_decode_workers is not None and self.max_decode_workers < 1:
            raise ValueError(
                "max_decode_workers must be positive when provided"
            )

    @classmethod
    def from_mapping(cls, raw: object) -> "ResourceLimits":
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError("resource_limits must be an object")
        accelerators = raw.get("accelerators", ())
        if not isinstance(accelerators, Sequence) or isinstance(
            accelerators, (str, bytes)
        ):
            raise ValueError("resource_limits.accelerators must be a list")
        return cls(
            preferred_accelerator=str(
                raw.get("preferred_accelerator", "auto")
            ),
            accelerators=tuple(str(item) for item in accelerators),
            max_ram_gb=(
                float(raw["max_ram_gb"])
                if raw.get("max_ram_gb") is not None
                else None
            ),
            max_vram_gb=(
                float(raw["max_vram_gb"])
                if raw.get("max_vram_gb") is not None
                else None
            ),
            max_cache_gb=(
                float(raw["max_cache_gb"])
                if raw.get("max_cache_gb") is not None
                else None
            ),
            max_decode_workers=(
                int(raw["max_decode_workers"])
                if raw.get("max_decode_workers") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "preferred_accelerator": self.preferred_accelerator,
            "accelerators": list(self.accelerators),
            "max_ram_gb": self.max_ram_gb,
            "max_vram_gb": self.max_vram_gb,
            "max_cache_gb": self.max_cache_gb,
            "max_decode_workers": self.max_decode_workers,
        }


@dataclass(frozen=True)
class TaskSpec:
    """Canonical task description shared by every agent and adapter."""

    schema_version: int
    task_id: str
    modality: str
    component_modalities: tuple[str, ...]
    problem_type: str
    inputs: Mapping[str, InputSpec]
    target: TargetSpec | None
    sample_id_field: str
    entity_id_field: str | None
    group_id_field: str | None
    time_field: str | None
    output: OutputSpec
    metrics: tuple[MetricSpec, ...]
    primary_metric: str
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError("schema_version must be a positive integer")
        if not str(self.task_id).strip():
            raise ValueError("task_id must be non-empty")
        modality = normalize_modality(self.modality)
        problem_type = normalize_problem_type(self.problem_type)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "problem_type", problem_type)
        components = tuple(
            normalize_modality(item) for item in self.component_modalities
        )
        if not components:
            components = (modality,)
        object.__setattr__(self, "component_modalities", components)
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError("inputs must define at least one named task input")
        for name, input_spec in self.inputs.items():
            if not isinstance(input_spec, InputSpec):
                raise ValueError(f"input {name!r} must be an InputSpec")
            if str(name) != input_spec.name:
                raise ValueError(
                    f"input key {name!r} does not match name "
                    f"{input_spec.name!r}"
                )
        if modality == "multimodal":
            if len(self.inputs) < 2:
                raise ValueError(
                    "multimodal tasks require at least two named inputs"
                )
            if any(item == "multimodal" for item in components):
                raise ValueError(
                    "component_modalities must name concrete modalities"
                )
        elif any(item != modality for item in components):
            raise ValueError(
                "a unimodal task may only declare its own component modality"
            )
        if not str(self.sample_id_field).strip():
            raise ValueError("sample_id_field must be non-empty")
        if not self.metrics:
            raise ValueError("metrics must contain at least one metric")
        metric_names = [metric.name for metric in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metric names must be unique")
        if self.primary_metric not in metric_names:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} is not in metrics"
            )
        if (
            problem_type != "unsupervised_clustering"
            and self.target is None
            and problem_type not in {"retrieval"}
        ):
            # Legacy tabular tasks sometimes rely on the generated loader to
            # infer an ambiguous target. Preserve that behavior for schema v1,
            # while resolved v2 contracts must be explicit.
            if self.schema_version >= 2 and problem_type != "supervised":
                raise ValueError(
                    f"{problem_type} tasks must define a target"
                )

    @property
    def metric_direction(self) -> str:
        return next(
            metric.direction
            for metric in self.metrics
            if metric.name == self.primary_metric
        )

    @classmethod
    def from_mapping(
        cls,
        task_id: str,
        raw: Mapping[str, object] | None,
        *,
        inferred_problem_type: str | None = None,
        legacy_roles: Mapping[str, str] | None = None,
        inferred_target_field: str | None = None,
    ) -> "TaskSpec":
        """Build a canonical task from v2 or legacy tabular configuration."""
        config = dict(raw or {})
        schema_version = int(config.get("schema_version", 1))
        modality = normalize_modality(config.get("modality", "tabular"))
        problem_type = normalize_problem_type(
            config.get("problem_type")
            or config.get("task_type")
            or inferred_problem_type
            or "supervised"
        )

        configured_inputs = config.get("inputs")
        input_specs: dict[str, InputSpec] = {}
        if configured_inputs is not None:
            if not isinstance(configured_inputs, Mapping):
                raise ValueError("inputs must be an object")
            for name, value in configured_inputs.items():
                input_specs[str(name)] = InputSpec.from_mapping(
                    str(name), value, default_modality=modality
                )
        else:
            roles = dict(legacy_roles or {})
            if not roles:
                legacy_keys = {
                    "train": config.get("train_file"),
                    "test": config.get("test_file"),
                    "data": config.get("data_file"),
                    "sample_submission": config.get(
                        "sample_submission_file"
                    ),
                }
                roles = {
                    name: str(value)
                    for name, value in legacy_keys.items()
                    if value
                }
            for name, source in roles.items():
                input_specs[name] = InputSpec(
                    name=name,
                    modality="tabular",
                    role=name,
                    source=str(source),
                    format=_inferred_format(str(source)),
                    required=name
                    not in {"sample_submission", "test"},
                )
            if not input_specs:
                input_specs["train"] = InputSpec(
                    name="train",
                    modality=modality,
                    role="train",
                    source="train.csv",
                    format="csv",
                    required=False,
                )

        default_target_source = (
            input_specs["train"].source
            if "train" in input_specs
            else input_specs.get("data", None).source
            if "data" in input_specs
            else None
        )
        raw_target = config.get("target")
        if raw_target is None:
            raw_target = config.get("target_column") or inferred_target_field
        target = TargetSpec.from_value(
            raw_target, default_source=default_target_source
        )
        if problem_type == "unsupervised_clustering":
            target = None

        output = OutputSpec.from_value(
            config.get("output"), problem_type=problem_type
        )
        default_direction = (
            str(config["metric_direction"]).lower()
            if config.get("metric_direction") is not None
            else None
        )
        raw_metrics = config.get("metrics")
        if raw_metrics is None:
            raw_metrics = [
                resolve_metric_name(
                    config.get("metric_name"),
                    problem_type=problem_type,
                    output_type=output.type,
                )
            ]
        if not isinstance(raw_metrics, Sequence) or isinstance(
            raw_metrics, (str, bytes)
        ):
            raise ValueError("metrics must be a list")

        def resolved_metric_value(metric: object) -> object:
            if isinstance(metric, str):
                return resolve_metric_name(
                    metric,
                    problem_type=problem_type,
                    output_type=output.type,
                )
            if isinstance(metric, Mapping):
                resolved = dict(metric)
                if resolved.get("name") is not None:
                    resolved["name"] = resolve_metric_name(
                        resolved["name"],
                        problem_type=problem_type,
                        output_type=output.type,
                    )
                return resolved
            return metric

        metrics = tuple(
            MetricSpec.from_value(
                resolved_metric_value(metric),
                default_direction=default_direction,
            )
            for metric in raw_metrics
        )
        primary_metric = resolve_metric_name(
            config.get("primary_metric") or metrics[0].name,
            problem_type=problem_type,
            output_type=output.type,
        )

        component_values = config.get("component_modalities")
        if component_values is None:
            if modality == "multimodal":
                component_values = [
                    input_spec.modality
                    for input_spec in input_specs.values()
                ]
            else:
                component_values = [modality]
        if not isinstance(component_values, Sequence) or isinstance(
            component_values, (str, bytes)
        ):
            raise ValueError("component_modalities must be a list")

        return cls(
            schema_version=schema_version,
            task_id=str(task_id),
            modality=modality,
            component_modalities=tuple(
                normalize_modality(item) for item in component_values
            ),
            problem_type=problem_type,
            inputs=input_specs,
            target=target,
            sample_id_field=str(
                config.get("sample_id_field")
                or config.get("id_column")
                or "row_id"
            ),
            entity_id_field=(
                str(config["entity_id_field"])
                if config.get("entity_id_field") is not None
                else None
            ),
            group_id_field=(
                str(config["group_id_field"])
                if config.get("group_id_field") is not None
                else None
            ),
            time_field=(
                str(config["time_field"])
                if config.get("time_field") is not None
                else None
            ),
            output=output,
            metrics=metrics,
            primary_metric=primary_metric,
            resource_limits=ResourceLimits.from_mapping(
                config.get("resource_limits")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-serializable task representation."""
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "modality": self.modality,
            "component_modalities": list(self.component_modalities),
            "problem_type": self.problem_type,
            "inputs": {
                name: input_spec.to_dict()
                for name, input_spec in self.inputs.items()
            },
            "target": self.target.to_dict() if self.target else None,
            "sample_id_field": self.sample_id_field,
            "entity_id_field": self.entity_id_field,
            "group_id_field": self.group_id_field,
            "time_field": self.time_field,
            "output": self.output.to_dict(),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "primary_metric": self.primary_metric,
            "resource_limits": self.resource_limits.to_dict(),
        }
