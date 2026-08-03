"""Registry-driven task discovery and profiling."""

from __future__ import annotations

import json
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from core.contracts import TaskSpec, normalize_modality
from core.modality_registry import ModalityRegistry
from core.runtime_contracts import DatasetBundle
from evaluation.metrics import (
    infer_metric_direction,
    infer_metric_from_description,
)
from modalities import build_default_registry
from modalities.base import ModalityAdapter
from modalities.generic import GenericAdapter
from modalities.paired_directory import discover_paired_directory_layout
from modalities.tabular import discover_dataset_layout
from .dataset_diagnostics import (
    build_dataset_diagnostics,
    render_dataset_analysis_markdown,
)
from .task_inventory import build_task_inventory, verify_task_contract
from .llm_utils import call_llm_json


class UnresolvedTaskError(ValueError):
    """Raised when observed files do not establish a safe task contract."""


@dataclass(frozen=True)
class TaskAnalysis:
    """Canonical task contract plus machine and human-readable profiles."""

    task_spec: TaskSpec
    profile: Mapping[str, object]
    report: str
    bundle: DatasetBundle | None = None
    inventory: Mapping[str, object] | None = None
    verification: Mapping[str, object] | None = None


def _read_task_config(task_dir: Path) -> dict[str, object]:
    path = Path(task_dir) / "task_config.json"
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        loaded = json.load(stream)
    if not isinstance(loaded, dict):
        raise ValueError("task_config.json must contain a JSON object")
    return loaded


class TaskAnalyzer:
    """Resolve a task through the adapter registered for its modality."""

    def __init__(
        self,
        registry: ModalityRegistry | None = None,
        *,
        model_name: str | None = None,
        enable_agent_resolution: bool = False,
    ) -> None:
        self.registry = registry or build_default_registry()
        self.model_name = model_name
        self.enable_agent_resolution = bool(enable_agent_resolution)

    def _configured_adapter(
        self, modality: str, config: Mapping[str, object]
    ) -> ModalityAdapter:
        """Use a registered decoder or the contract-driven generic indexer."""
        if modality in self.registry:
            adapter = self.registry.get(modality)
        else:
            adapter = GenericAdapter(config)
        if not isinstance(adapter, ModalityAdapter):
            raise TypeError(
                f"registered {modality!r} adapter does not implement "
                "the ModalityAdapter contract"
            )
        return adapter

    def _adapter_for(
        self,
        task_dir: Path,
        inventory: Mapping[str, object],
    ) -> ModalityAdapter:
        config = _read_task_config(task_dir)
        modality = None
        adapter = None
        if config:
            configured_modality = config.get("modality")
            if configured_modality is not None:
                modality = normalize_modality(configured_modality)
                adapter = self._configured_adapter(modality, config)
            else:
                configured_inputs = config.get("inputs")
                input_modalities = {
                    normalize_modality(value.get("modality"))
                    for value in (
                        configured_inputs.values()
                        if isinstance(configured_inputs, Mapping)
                        else ()
                    )
                    if isinstance(value, Mapping)
                    and value.get("modality") is not None
                }
                if len(input_modalities) > 1:
                    modality = "multimodal"
                    adapter = self._configured_adapter(modality, config)
                elif len(input_modalities) == 1:
                    modality = next(iter(input_modalities))
                    adapter = self._configured_adapter(modality, config)
                elif any(
                    config.get(key) is not None
                    for key in (
                        "train_file",
                        "test_file",
                        "data_file",
                        "target_column",
                    )
                ):
                    # This is an explicit legacy loader contract, not a default
                    # inferred from the absence of task information.
                    modality = "tabular"
                    adapter = self._configured_adapter(modality, config)
                elif isinstance(configured_inputs, Mapping) and configured_inputs:
                    # A non-empty task contract with explicit inputs need not
                    # name a predefined data family. The generic adapter uses
                    # the declared paths and roles without category dispatch.
                    modality = "task_native"
                    adapter = GenericAdapter(config)

        if adapter is None:
            paired = discover_paired_directory_layout(task_dir)
            if paired is not None:
                modality = paired.modality
                adapter = self.registry.get(modality)
            else:
                multimodal = self.registry.get("multimodal")
                if (
                    hasattr(multimodal, "can_auto_discover")
                    and multimodal.can_auto_discover(task_dir)
                ):
                    adapter = multimodal
                    modality = "multimodal"
                else:
                    layout = discover_dataset_layout(task_dir)
                    roles = layout.get("roles", {})
                    meaningful_non_text = [
                        group
                        for group in inventory.get("file_groups", [])
                        if isinstance(group, Mapping)
                        and int(group.get("file_count", 0)) >= 2
                        and set(group.get("observed_content_kinds", {}))
                        - {"utf8_text"}
                    ]
                    if (
                        isinstance(roles, Mapping)
                        and any(role in roles for role in ("train", "data"))
                        and not meaningful_non_text
                    ):
                        modality = "tabular"
                        adapter = self.registry.get(modality)
        if adapter is None or modality is None:
            raise UnresolvedTaskError(
                "The task could not be verified from its observed files. No "
                "adapter matched the content and relationships in the neutral "
                "task inventory; refusing to assume a table, target, split, or "
                "data family. Supply an explicit task contract or a registered "
                "content-driven adapter."
            )
        if not isinstance(adapter, ModalityAdapter):
            raise TypeError(
                f"registered {modality!r} adapter does not implement "
                "the ModalityAdapter contract"
            )
        return adapter

    @staticmethod
    def _reasoned_contract_schema() -> dict[str, object]:
        """Schema for an evidence-grounded contract, not a category choice."""
        input_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "name",
                "role",
                "source",
                "format",
                "id_field",
                "path_field",
                "required",
                "evidence",
            ],
            "properties": {
                "name": {"type": "string"},
                "role": {"type": "string"},
                "source": {"type": "string"},
                "format": {"type": "string"},
                "id_field": {"type": "string"},
                "path_field": {"type": "string"},
                "required": {"type": "boolean"},
                "evidence": {"type": "string"},
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "resolved",
                "summary",
                "evidence",
                "uncertainties",
                "contract",
            ],
            "properties": {
                "resolved": {"type": "boolean"},
                "summary": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "uncertainties": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "contract": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "task_kind",
                        "objective",
                        "inputs",
                        "target",
                        "sample_id_field",
                        "output",
                        "metrics",
                        "primary_metric",
                    ],
                    "properties": {
                        "task_kind": {"type": "string"},
                        "objective": {"type": "string"},
                        "inputs": {
                            "type": "array",
                            "minItems": 1,
                            "items": input_schema,
                        },
                        "target": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "present",
                                "source",
                                "field",
                                "format",
                                "id_field",
                                "alignment",
                                "evidence",
                            ],
                            "properties": {
                                "present": {"type": "boolean"},
                                "source": {"type": "string"},
                                "field": {"type": "string"},
                                "format": {"type": "string"},
                                "id_field": {"type": "string"},
                                "alignment": {"type": "string"},
                                "evidence": {"type": "string"},
                            },
                        },
                        "sample_id_field": {"type": "string"},
                        "output": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["type", "shape", "encoding"],
                            "properties": {
                                "type": {"type": "string"},
                                "shape": {"type": "string"},
                                "encoding": {"type": "string"},
                            },
                        },
                        "metrics": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["name", "direction"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "direction": {
                                        "type": "string",
                                        "enum": ["maximize", "minimize"],
                                    },
                                },
                            },
                        },
                        "primary_metric": {"type": "string"},
                    },
                },
            },
        }

    def _reasoned_adapter(
        self,
        task_dir: Path,
        inventory: Mapping[str, object],
    ) -> ModalityAdapter:
        """Ask the analysis agent to explain a contract from observed facts."""
        if not self.enable_agent_resolution:
            raise UnresolvedTaskError(
                "The task could not be verified from its observed files; "
                "refusing to assume a table, target, split, or data family."
            )
        # Put narrative and schema evidence first so a very large directory
        # listing cannot push the task statement beyond the bounded prompt.
        prioritized_inventory = {
            "task_id": inventory.get("task_id"),
            "total_files": inventory.get("total_files"),
            "total_bytes": inventory.get("total_bytes"),
            "text_documents": inventory.get("text_documents", []),
            "table_summaries": inventory.get("table_summaries", []),
            "stem_relationships": inventory.get("stem_relationships", []),
            "file_groups": inventory.get("file_groups", []),
            "top_level_entries": inventory.get("top_level_entries", []),
            "inventory_fingerprint": inventory.get("inventory_fingerprint"),
        }
        inventory_text = json.dumps(
            prioritized_inventory, indent=2, default=str
        )
        response = call_llm_json(
            "You are the task-analysis agent. Resolve an executable ML task "
            "contract only after inspecting the supplied task-directory "
            "inventory. Reconcile the task narrative, byte signatures, table "
            "previews, document contents, counts, and cross-directory stem "
            "relationships. Do not map the task to a predefined modality or "
            "objective taxonomy. task_kind, objective, and output.type are "
            "free descriptive identifiers. Do not infer roles from conventional "
            "filenames. Every source must be an exact observed file or directory. "
            "Use role='train' for sources containing examples used to fit or "
            "derive a method, role='test' only for sources requiring final "
            "predictions, and descriptive roles for metadata/templates. An empty "
            "id_field/path_field/target field means it is not established. Set "
            "resolved=false whenever the files do not establish a single safe "
            "interpretation; list the ambiguity instead of guessing.",
            "TASK-DIRECTORY INVENTORY (authoritative observed evidence):\n"
            + inventory_text[:100_000],
            model=self.model_name,
            schema_name="evidence_grounded_task_contract",
            schema=self._reasoned_contract_schema(),
        )
        if not isinstance(response, Mapping) or not response.get("resolved"):
            uncertainties = (
                response.get("uncertainties", [])
                if isinstance(response, Mapping)
                else []
            )
            raise UnresolvedTaskError(
                "task-analysis agent could not verify one unambiguous contract: "
                + "; ".join(str(item) for item in uncertainties)
            )
        raw_contract = response.get("contract")
        if not isinstance(raw_contract, Mapping):
            raise UnresolvedTaskError("task-analysis agent returned no contract")
        raw_inputs = raw_contract.get("inputs")
        if not isinstance(raw_inputs, list) or not raw_inputs:
            raise UnresolvedTaskError("task-analysis agent returned no inputs")
        inputs: dict[str, object] = {}
        for raw_input in raw_inputs:
            if not isinstance(raw_input, Mapping):
                raise UnresolvedTaskError("task-analysis input must be an object")
            name = str(raw_input.get("name", "")).strip()
            if not name or name in inputs:
                raise UnresolvedTaskError(
                    "task-analysis inputs require unique non-empty names"
                )
            source = str(raw_input.get("source", "")).strip()
            if self._safe_task_source(task_dir, source) is None:
                raise UnresolvedTaskError(
                    f"task-analysis agent cited an unobserved source: {source!r}"
                )
            options = {
                "resolution_evidence": str(raw_input.get("evidence", ""))
            }
            path_field = str(raw_input.get("path_field", "")).strip()
            if path_field:
                options["path_field"] = path_field
            inputs[name] = {
                "modality": str(raw_contract.get("task_kind", "task_native")),
                "role": str(raw_input.get("role", "")),
                "source": source,
                "format": str(raw_input.get("format", "file")),
                "id_field": str(raw_input.get("id_field", "")).strip() or None,
                "required": bool(raw_input.get("required", True)),
                "options": options,
            }
        raw_target = raw_contract.get("target")
        target = None
        if isinstance(raw_target, Mapping) and raw_target.get("present"):
            target_source = str(raw_target.get("source", "")).strip()
            if not target_source:
                raise UnresolvedTaskError(
                    "task-analysis agent marked a target present without a source"
                )
            resolved_target_source = self._safe_task_source(
                task_dir, target_source
            )
            if resolved_target_source is None:
                raise UnresolvedTaskError(
                    "task-analysis agent cited an unobserved target source: "
                    f"{target_source!r}"
                )
            target = {
                "source": target_source,
                "field": str(raw_target.get("field", "")).strip() or None,
                "type": (
                    "file_reference"
                    if resolved_target_source.is_dir()
                    else str(raw_target.get("format", "file"))
                ),
                "format": str(raw_target.get("format", "file")),
                "options": {
                    "resolution_evidence": str(raw_target.get("evidence", "")),
                    "id_field": str(raw_target.get("id_field", "")).strip(),
                    "alignment": str(raw_target.get("alignment", "")).strip(),
                },
            }
        raw_output = raw_contract.get("output")
        output = dict(raw_output) if isinstance(raw_output, Mapping) else {}
        mapping = {
            "schema_version": 2,
            "modality": str(raw_contract.get("task_kind", "task_native")),
            "problem_type": str(raw_contract.get("objective", "ml_task")),
            "inputs": inputs,
            "target": target,
            "sample_id_field": str(
                raw_contract.get("sample_id_field", "sample_id")
            ),
            "output": output,
            "metrics": raw_contract.get("metrics"),
            "primary_metric": raw_contract.get("primary_metric"),
            "_resolution": {
                "resolution_summary": response.get("summary"),
                "resolution_evidence": response.get("evidence"),
                "resolution_uncertainties": response.get("uncertainties"),
            },
        }
        adapter = GenericAdapter(mapping)
        # Parse now so malformed identifiers, roles, metrics, and output facts
        # fail at the analysis boundary rather than during a method-tree node.
        adapter.discover(task_dir)
        return adapter

    @staticmethod
    def _explicit_metric_config(task_dir: Path) -> bool:
        config = _read_task_config(task_dir)
        return any(
            config.get(key) is not None
            for key in ("metrics", "metric_name", "primary_metric")
        )

    @staticmethod
    def _explicit_sample_id_config(task_dir: Path) -> bool:
        config = _read_task_config(task_dir)
        return any(
            config.get(key) is not None
            for key in ("sample_id_field", "id_column")
        )

    @staticmethod
    def _task_description(task_dir: Path) -> str:
        path = Path(task_dir) / "task_description.md"
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    @staticmethod
    def _safe_task_path(task_dir: Path, source: str) -> Path | None:
        root = Path(task_dir).resolve()
        raw = Path(str(source))
        candidates = (
            (raw,)
            if raw.is_absolute()
            else (root / raw, root / "input" / raw)
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved != root and root not in resolved.parents:
                continue
            if resolved.is_file():
                return resolved
        return None

    @staticmethod
    def _safe_task_source(task_dir: Path, source: str) -> Path | None:
        """Resolve a declared file or directory without leaving the task."""
        root = Path(task_dir).resolve()
        raw = Path(str(source))
        candidates = (
            (raw,)
            if raw.is_absolute()
            else (root / raw, root / "input" / raw)
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved != root and root not in resolved.parents:
                continue
            if resolved.exists():
                return resolved
        return None

    @classmethod
    def _submission_class_names(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> tuple[str, ...]:
        if (
            task_spec.output.type != "class_probabilities"
            or task_spec.output.class_names
        ):
            return task_spec.output.class_names
        candidates = [
            item.source
            for item in task_spec.inputs.values()
            if item.role == "sample_submission"
        ]
        for source in candidates:
            path = cls._safe_task_path(task_dir, source)
            if path is None:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            prediction_columns = tuple(
                str(column) for column in columns[1:] if str(column)
            )
            # A single probability column is normally the positive class, not
            # the complete binary class vocabulary.
            if len(prediction_columns) > 1:
                return prediction_columns
        return ()

    @classmethod
    def _submission_columns(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> tuple[str, ...]:
        candidates = [
            item.source
            for item in task_spec.inputs.values()
            if item.role == "sample_submission"
        ]
        for source in candidates:
            path = cls._safe_task_path(task_dir, source)
            if path is None:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            if len(columns) >= 2:
                return tuple(str(column) for column in columns)
        return ()

    @classmethod
    def _submission_id_field(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> str:
        """Infer the final-output ID only when it exists in declared test data."""
        if cls._explicit_sample_id_config(task_dir):
            return task_spec.sample_id_field
        template_sources = [
            item.source
            for item in task_spec.inputs.values()
            if item.role == "sample_submission"
        ]
        template_id = None
        for source in template_sources:
            path = cls._safe_task_path(task_dir, source)
            if path is None:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            if columns:
                template_id = str(columns[0])
                break
        if not template_id:
            return task_spec.sample_id_field
        for item in task_spec.inputs.values():
            if item.role != "test" and item.name != "test":
                continue
            path = cls._safe_task_path(task_dir, item.source)
            if path is None or path.suffix.lower() not in {".csv", ".tsv"}:
                continue
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as stream:
                columns = next(csv.reader(stream, delimiter=delimiter), [])
            if template_id in columns:
                return template_id
        return task_spec.sample_id_field

    @classmethod
    def _enrich_discovered_task(
        cls, task_dir: Path, task_spec: TaskSpec
    ) -> TaskSpec:
        mapping = task_spec.to_dict()
        changed = False
        if not cls._explicit_metric_config(task_dir):
            inferred_metric = infer_metric_from_description(
                cls._task_description(task_dir)
            )
            if inferred_metric and inferred_metric != task_spec.primary_metric:
                mapping["metrics"] = [
                    {
                        "name": inferred_metric,
                        "direction": infer_metric_direction(inferred_metric),
                    }
                ]
                mapping["primary_metric"] = inferred_metric
                changed = True
        inferred_classes = cls._submission_class_names(task_dir, task_spec)
        if inferred_classes and inferred_classes != task_spec.output.class_names:
            output = dict(mapping["output"])
            output["class_names"] = list(inferred_classes)
            mapping["output"] = output
            changed = True
        submission_columns = cls._submission_columns(task_dir, task_spec)
        if submission_columns:
            output = dict(mapping["output"])
            options = dict(output.get("options", {}))
            submission_contract = {
                "submission_id_column": submission_columns[0],
                "submission_prediction_columns": list(
                    submission_columns[1:]
                ),
            }
            if any(
                options.get(key) != value
                for key, value in submission_contract.items()
            ):
                options.update(submission_contract)
                output["options"] = options
                mapping["output"] = output
                changed = True
        inferred_id_field = cls._submission_id_field(task_dir, task_spec)
        if inferred_id_field != task_spec.sample_id_field:
            mapping["sample_id_field"] = inferred_id_field
            changed = True
        return (
            TaskSpec.from_mapping(task_spec.task_id, mapping)
            if changed
            else task_spec
        )

    def _discover(
        self, task_dir: Path
    ) -> tuple[ModalityAdapter, TaskSpec, Mapping[str, object]]:
        inventory = build_task_inventory(task_dir)
        try:
            adapter = self._adapter_for(task_dir, inventory)
        except UnresolvedTaskError as deterministic_error:
            try:
                adapter = self._reasoned_adapter(task_dir, inventory)
            except Exception as reasoning_error:
                if reasoning_error is deterministic_error:
                    raise
                raise UnresolvedTaskError(
                    f"{deterministic_error} Task-analysis reasoning also "
                    f"failed safely: {reasoning_error}"
                ) from reasoning_error
        task_spec = self._enrich_discovered_task(
            task_dir, adapter.discover(task_dir)
        )
        return adapter, task_spec, inventory

    def resolve(
        self, task_dir: Path, *, require_verification: bool = True
    ) -> TaskSpec:
        """Resolve only the canonical task contract without profiling data."""
        task_dir = Path(task_dir)
        _, task_spec, inventory = self._discover(task_dir)
        verification = verify_task_contract(task_dir, task_spec, inventory)
        if require_verification and not verification["verified"]:
            raise ValueError(
                "task contract verification failed before method-tree planning: "
                + "; ".join(verification["errors"])
            )
        return task_spec

    @staticmethod
    def _validate_output_boundary(
        task_dir: Path, output_dir: Path
    ) -> None:
        task_root = Path(task_dir).resolve()
        output_root = Path(output_dir).resolve()
        if output_root == task_root or task_root in output_root.parents:
            raise ValueError(
                "task analysis output must be outside the read-only task "
                f"directory: {output_dir}"
            )

    def analyze(
        self,
        task_dir: Path,
        *,
        output_dir: Path | None = None,
        include_index: bool = False,
        require_verification: bool = True,
    ) -> TaskAnalysis:
        """Discover, profile, and optionally persist a canonical task."""
        task_dir = Path(task_dir)
        adapter, task_spec, inventory = self._discover(task_dir)
        if (
            task_spec.modality != adapter.name
            and not getattr(adapter, "handles_arbitrary_task_identifiers", False)
        ):
            raise ValueError(
                f"adapter {adapter.name!r} returned modality "
                f"{task_spec.modality!r}"
            )
        profile = dict(adapter.profile(task_dir, task_spec))
        profile["profile_schema_version"] = 2
        # Tabular data already has an efficient columnar source (CSV/TSV).
        # Expanding every row into JSONL duplicates the complete dataset and
        # can add hundreds of megabytes before the first experiment starts.
        # Structured modalities still need the record index for path/component
        # joins, so retain it only for those adapters.
        direct_tabular = include_index and task_spec.modality == "tabular"
        bundle = (
            adapter.build_bundle(task_dir, task_spec)
            if include_index and not direct_tabular
            else None
        )
        verification = verify_task_contract(
            task_dir,
            task_spec,
            inventory,
            bundle=bundle,
        )
        if require_verification and not verification["verified"]:
            raise ValueError(
                "task contract verification failed before method-tree planning: "
                + "; ".join(verification["errors"])
            )
        profile["task_inventory"] = inventory
        profile["task_verification"] = verification
        profile["diagnostics"] = build_dataset_diagnostics(
            task_dir, task_spec, bundle=bundle
        )

        index_metadata: dict[str, object] | None = None
        if bundle is not None:
            index_metadata = {
                **bundle.to_index_dict(),
                "storage": "jsonl_index",
                "row_index_materialized": True,
            }
        elif direct_tabular:
            source_metadata = []
            for input_spec in task_spec.inputs.values():
                source_path = self._safe_task_path(
                    task_dir, input_spec.source
                )
                if source_path is None:
                    continue
                stat = source_path.stat()
                source_metadata.append(
                    {
                        "name": input_spec.name,
                        "role": input_spec.role,
                        "source": input_spec.source,
                        "size_bytes": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                    }
                )
            fingerprint_payload = {
                "task": task_spec.to_dict(),
                "sources": source_metadata,
            }
            index_metadata = {
                "schema_version": 2,
                "storage": "direct_tabular",
                "dataset_fingerprint": hashlib.sha256(
                    json.dumps(
                        fingerprint_payload,
                        sort_keys=True,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest(),
                "sources": source_metadata,
                "row_index_materialized": False,
            }
        if index_metadata is not None:
            # The runtime index contract is part of the canonical machine
            # profile; a separate manifest duplicated task and source metadata.
            profile["dataset_index"] = index_metadata
        report = render_dataset_analysis_markdown(task_spec, profile)
        report += (
            "\n## Task verification\n\n"
            "- Contract verified against task-owned files: "
            f"{'yes' if verification['verified'] else 'no'}\n"
            f"- Observed files: {inventory['total_files']}\n"
            f"- Inventory fingerprint: `{inventory['inventory_fingerprint']}`\n"
            f"- Indexed training records: {verification['train_record_count']}\n"
            f"- Indexed test records: {verification['test_record_count']}\n"
        )
        if verification["warnings"]:
            report += "- Warnings: " + "; ".join(verification["warnings"]) + "\n"
        analysis = TaskAnalysis(
            task_spec=task_spec,
            profile=profile,
            report=report,
            bundle=bundle,
            inventory=inventory,
            verification=verification,
        )
        if output_dir is not None:
            output_dir = Path(output_dir)
            self._validate_output_boundary(task_dir, output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "resolved_task_spec.json").write_text(
                json.dumps(task_spec.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            (output_dir / "dataset_profile.json").write_text(
                json.dumps(profile, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "dataset_analysis.md").write_text(
                report, encoding="utf-8"
            )
            (output_dir / "task_inventory.json").write_text(
                json.dumps(inventory, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (output_dir / "task_verification.json").write_text(
                json.dumps(verification, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if bundle is not None:
                with open(
                    output_dir / "dataset_index.jsonl",
                    "w",
                    encoding="utf-8",
                ) as stream:
                    for record in (
                        *bundle.train_records,
                        *bundle.test_records,
                    ):
                        stream.write(
                            json.dumps(record.to_dict(), default=str) + "\n"
                        )
        return analysis
