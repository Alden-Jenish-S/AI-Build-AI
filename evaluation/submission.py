"""Task-aware validation and alignment for generated submission artifacts."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def _task_mapping(task_spec: Any) -> dict[str, Any]:
    if hasattr(task_spec, "to_dict"):
        return dict(task_spec.to_dict())
    if isinstance(task_spec, Mapping):
        return dict(task_spec)
    raise TypeError("task_spec must be a TaskSpec or mapping")


def _safe_task_file(task_dir: Path, source: object) -> Path | None:
    root = Path(task_dir).resolve()
    raw = Path(str(source or ""))
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


def resolve_submission_template(
    task_dir: str | Path, task_spec: Any
) -> Path | None:
    """Resolve the task-owned submission template without assuming its basename."""
    root = Path(task_dir)
    spec = _task_mapping(task_spec)
    inputs = spec.get("inputs", {})
    candidates: list[object] = []
    if isinstance(inputs, Mapping):
        for _, raw in inputs.items():
            if not isinstance(raw, Mapping):
                continue
            if str(raw.get("role", "")) == "sample_submission":
                candidates.append(raw.get("source"))
    for source in candidates:
        path = _safe_task_file(root, source)
        if path is not None:
            return path
    return None


def task_requires_submission(task_dir: str | Path, task_spec: Any) -> bool:
    """Return whether the resolved task exposes a final prediction population."""
    spec = _task_mapping(task_spec)
    if resolve_submission_template(task_dir, spec) is not None:
        return True
    inputs = spec.get("inputs", {})
    if not isinstance(inputs, Mapping):
        return False
    for _, raw in inputs.items():
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("role", "")) == "test":
            return True
        options = raw.get("options", {})
        if isinstance(options, Mapping) and options.get("test_source"):
            return True
        if raw.get("test_source"):
            return True
    return False


def _discover_aligned_output_template(
    task_dir: Path, generated_path: Path
) -> Path | None:
    """Find an undeclared template by exact observed ID/row alignment."""
    try:
        generated = pd.read_csv(generated_path)
    except Exception:
        return None
    if generated.empty or len(generated.columns) < 2:
        return None
    generated_ids = set(generated.iloc[:, 0].dropna().astype(str))
    candidates: list[tuple[int, Path]] = []
    for path in sorted(Path(task_dir).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".tsv"}:
            continue
        try:
            frame = pd.read_csv(
                path,
                sep="\t" if path.suffix.lower() == ".tsv" else ",",
            )
        except Exception:
            continue
        if frame.empty or len(frame.columns) < 2 or len(frame) != len(generated):
            continue
        for column in frame.columns:
            candidate_ids = set(frame[column].dropna().astype(str))
            if candidate_ids == generated_ids and candidate_ids:
                # Prefer a first-column identity match; the filename is never
                # consulted.
                candidates.append((int(column == frame.columns[0]), path))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    best_score = candidates[0][0]
    best = [path for score, path in candidates if score == best_score]
    return best[0] if len(best) == 1 else None


def _aligned_submission(
    generated: pd.DataFrame, sample: pd.DataFrame
) -> tuple[pd.DataFrame, str, list[str], dict[str, str]]:
    if generated.empty or sample.empty:
        raise ValueError("submission files must not be empty")
    if len(sample.columns) < 2:
        raise ValueError("sample submission has no prediction columns")
    id_column = str(sample.columns[0])
    prediction_columns = [str(column) for column in sample.columns[1:]]
    if id_column not in generated.columns:
        raise ValueError(
            f"generated submission must include the template ID column {id_column!r}"
        )
    schema_renames: dict[str, str] = {}
    expected_columns = {id_column, *prediction_columns}
    generated_columns = set(str(column) for column in generated.columns)
    # For scalar-output tasks there is no semantic ambiguity: the only
    # non-ID generated column is the task's sole prediction column.
    if (
        generated_columns != expected_columns
        and len(prediction_columns) == 1
    ):
        generated_prediction_columns = [
            str(column)
            for column in generated.columns
            if str(column) != id_column
        ]
        if len(generated_prediction_columns) == 1:
            source = generated_prediction_columns[0]
            destination = prediction_columns[0]
            generated = generated.rename(columns={source: destination})
            schema_renames[source] = destination
            generated_columns = set(str(column) for column in generated.columns)
    if generated_columns != expected_columns:
        missing = sorted(expected_columns - generated_columns)
        unexpected = sorted(generated_columns - expected_columns)
        raise ValueError(
            "generated submission schema differs from the template; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if sample[id_column].duplicated().any():
        raise ValueError(f"sample submission contains duplicate {id_column!r} values")
    if generated[id_column].duplicated().any():
        raise ValueError(
            f"generated submission contains duplicate {id_column!r} values"
        )
    if sample[id_column].isna().any() or generated[id_column].isna().any():
        raise ValueError("submission IDs must not be missing")
    sample_keys = sample[id_column].astype(str)
    generated_keys = generated[id_column].astype(str)
    if sample_keys.duplicated().any() or generated_keys.duplicated().any():
        raise ValueError("string-normalized submission IDs are not unique")
    if set(sample_keys) != set(generated_keys):
        def numeric_key(value: object) -> str | None:
            try:
                parsed = Decimal(str(value))
            except (InvalidOperation, ValueError):
                return None
            if not parsed.is_finite():
                return None
            if parsed == 0:
                return "0"
            normalized = format(parsed.normalize(), "f")
            return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized

        sample_numeric = sample_keys.map(numeric_key)
        generated_numeric = generated_keys.map(numeric_key)
        # Preserve identifiers with meaningful formatting such as leading
        # zeros. Numeric equivalence is allowed only when template IDs are
        # already in their canonical numeric representation.
        if (
            sample_numeric.isna().any()
            or generated_numeric.isna().any()
            or not np.array_equal(
                sample_keys.to_numpy(), sample_numeric.to_numpy()
            )
            or set(sample_numeric) != set(generated_numeric)
        ):
            raise ValueError(
                "generated IDs do not exactly match sample submission IDs"
            )
        sample_keys = sample_numeric
        generated_keys = generated_numeric
        if sample_keys.duplicated().any() or generated_keys.duplicated().any():
            raise ValueError(
                "numeric-normalized submission IDs are not unique"
            )
    indexed = generated.copy()
    indexed["__submission_key__"] = generated_keys
    aligned = indexed.set_index("__submission_key__").reindex(sample_keys)
    result = sample[[id_column]].copy()
    for column in prediction_columns:
        result[column] = aligned[column].to_numpy()
    return result, id_column, prediction_columns, schema_renames


def validate_submission_file(
    submission_path: str | Path,
    *,
    task_dir: str | Path,
    task_spec: Any,
    normalize_probabilities: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate a generated submission and return template-aligned contents."""
    path = Path(submission_path)
    if not path.is_file():
        raise ValueError(f"generated submission is missing: {path}")
    spec = _task_mapping(task_spec)
    output = spec.get("output", {})
    output_type = (
        str(output.get("type", ""))
        if isinstance(output, Mapping)
        else str(output)
    )
    output_options = (
        output.get("options", {})
        if isinstance(output, Mapping)
        and isinstance(output.get("options", {}), Mapping)
        else {}
    )
    rle_submission = (
        output_options.get("submission_encoding")
        == "run_length_encoding"
    )
    template_path = resolve_submission_template(task_dir, spec)
    if template_path is None:
        template_path = _discover_aligned_output_template(
            Path(task_dir), path
        )
    if template_path is not None:
        template_columns = pd.read_csv(template_path, nrows=0).columns
        if len(template_columns) < 2:
            raise ValueError("sample submission has no prediction columns")
        template_id = str(template_columns[0])
        generated_columns = pd.read_csv(path, nrows=0).columns
        sample = pd.read_csv(
            template_path,
            dtype={template_id: "string"},
            keep_default_na=not rle_submission,
        )
        generated = pd.read_csv(
            path,
            dtype=(
                {template_id: "string"}
                if template_id in generated_columns
                else None
            ),
            keep_default_na=not rle_submission,
        )
        frame, id_column, prediction_columns, schema_renames = _aligned_submission(
            generated, sample
        )
    else:
        generated = pd.read_csv(path, keep_default_na=not rle_submission)
        if generated.empty or len(generated.columns) < 2:
            raise ValueError(
                "submission must contain an ID column and at least one output column"
            )
        frame = generated.copy()
        id_column = str(frame.columns[0])
        prediction_columns = [str(column) for column in frame.columns[1:]]
        if frame[id_column].duplicated().any():
            raise ValueError(
                f"generated submission contains duplicate {id_column!r} values"
            )
        schema_renames = {}

    if frame[prediction_columns].isnull().any().any():
        raise ValueError("generated submission contains missing predictions")
    problem_type = str(spec.get("problem_type", ""))
    normalization_applied = False
    if output_type in {"class_probabilities", "continuous"}:
        try:
            values = frame[prediction_columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("submission predictions must be numeric") from exc
        if not np.isfinite(values).all():
            raise ValueError("submission predictions contain NaN or infinity")
        if output_type == "class_probabilities":
            tolerance = 1e-12
            if (values < -tolerance).any() or (values > 1.0 + tolerance).any():
                raise ValueError(
                    "class-probability predictions must be within [0, 1]"
                )
            values = np.clip(values, 0.0, 1.0)
            if problem_type == "classification" and values.shape[1] > 1:
                row_sums = values.sum(axis=1, keepdims=True)
                if (row_sums <= 0).any():
                    raise ValueError(
                        "each multiclass probability row must have positive mass"
                    )
                if not np.allclose(row_sums, 1.0, rtol=1e-6, atol=1e-8):
                    if not normalize_probabilities:
                        raise ValueError(
                            "multiclass probability rows must sum to one"
                        )
                    values = values / row_sums
                    normalization_applied = True
            frame[prediction_columns] = values
    elif rle_submission:
        for column in prediction_columns:
            for row_index, raw_value in enumerate(frame[column].tolist()):
                text = str(raw_value or "").strip()
                if not text:
                    continue
                try:
                    encoded = [int(item) for item in text.split()]
                except ValueError as exc:
                    raise ValueError(
                        f"RLE column {column!r} row {row_index} contains "
                        "non-integer tokens"
                    ) from exc
                if len(encoded) % 2:
                    raise ValueError(
                        f"RLE column {column!r} row {row_index} must contain "
                        "start/length pairs"
                    )
                previous_end = 0
                index_base = int(output_options.get("rle_index_base", 1))
                for start, length in zip(encoded[::2], encoded[1::2]):
                    if start < index_base:
                        raise ValueError(
                            f"RLE start is below the configured index base in "
                            f"column {column!r} row {row_index}"
                        )
                    if length <= 0 or start <= previous_end:
                        raise ValueError(
                            "RLE runs must be positive, sorted, and "
                            f"non-overlapping in column {column!r} row {row_index}"
                        )
                    previous_end = start + length - 1

    return frame, {
        "submission_path": str(path),
        "template_path": str(template_path) if template_path else None,
        "row_count": int(len(frame)),
        "id_column": id_column,
        "prediction_columns": prediction_columns,
        "output_type": output_type,
        "normalization_applied": normalization_applied,
        "schema_renames": schema_renames,
    }


def validate_node_submission(
    node_dir: str | Path,
    *,
    task_dir: str | Path,
    task_spec: Any,
    require_full_training_manifest: bool = True,
) -> dict[str, Any]:
    """Enforce the per-node final-prediction contract used by the manager."""
    root = Path(node_dir)
    submission_path = root / "submission" / "submission.csv"
    frame, validation = validate_submission_file(
        submission_path,
        task_dir=task_dir,
        task_spec=task_spec,
        normalize_probabilities=True,
    )
    if require_full_training_manifest and task_requires_submission(
        task_dir, task_spec
    ):
        public_manifest_path = root / "final_training_manifest.json"
        contract_manifest_path = (
            root
            / ".evaluation_contract"
            / "final_training_manifest.json"
        )
        manifest_path = (
            contract_manifest_path
            if contract_manifest_path.is_file()
            else public_manifest_path
        )
        if not manifest_path.is_file():
            raise ValueError(
                "final training contract is missing; call "
                "prepare_final_training_data before fitting the test model"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("used_full_training_data"):
            raise ValueError("final test model was not prepared from full training data")
        if int(manifest.get("test_row_count", -1)) != len(frame):
            raise ValueError(
                "final training manifest test rows do not match the submission"
            )
        validation["final_training_manifest"] = str(manifest_path)
        validation["full_training_row_count"] = int(
            manifest.get("train_row_count", 0)
        )
        # Replace any generated ad-hoc manifest with the small canonical proof.
        public_manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    # Persist canonical order and normalized probabilities for downstream merges.
    frame.to_csv(submission_path, index=False)
    return validation
