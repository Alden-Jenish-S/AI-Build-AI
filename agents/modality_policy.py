"""Detect predictive modalities and validate comparable modality ablations."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


_KIND_TO_MODALITY = {
    "table": "tabular",
    "structured_text": "structured_text",
    "text": "text",
    "image": "image",
    "audio": "audio",
    "video": "video",
}
_NON_PREDICTIVE_STEMS = {
    "readme",
    "task",
    "task_description",
    "description",
    "overview",
    "instructions",
    "sample_submission",
    "sample_output",
    "submission_format",
}

# Identifier/join-key columns are placeholders, not predictive features. Tables
# whose only columns are keys and/or mirror of the submission (label) columns
# are index/label files, not a second predictive modality.
_IDENTIFIER_COLUMNS = {
    "id",
    "ids",
    "image_id",
    "file_id",
    "file_name",
    "filename",
    "sample_id",
    "row_id",
    "rowid",
    "object_id",
    "photo_id",
    "user_id",
    "index",
    "key",
    "uid",
    "uuid",
    "path",
    "filepath",
    "image",
    "image_name",
}
_TARGET_LIKE_COLUMNS = {
    "label",
    "labels",
    "target",
    "targets",
    "class",
    "classes",
    "y",
    "answer",
    "answers",
    "ground_truth",
    "groundtruth",
}
_OUTPUT_MIRROR_STEMS = {
    "sample_submission",
    "sample_output",
    "submission_format",
    "submission_template",
    "output_format",
}


def _normalized_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def _column_is_identifier(value: object) -> bool:
    normalized = _normalized_column(value)
    if not normalized:
        return True
    if normalized in _IDENTIFIER_COLUMNS:
        return True
    return normalized.endswith(("_id", "_uid", "_key"))


def _column_is_target_like(value: object) -> bool:
    normalized = _normalized_column(value)
    if normalized in _TARGET_LIKE_COLUMNS:
        return True
    return normalized.startswith(("label", "target", "ground_truth", "groundtruth")) or normalized.endswith(
        ("_label", "_target", "_class", "_answer")
    )


def _looks_non_predictive(path_value: object) -> bool:
    stem = Path(str(path_value or "")).stem.casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    if normalized in _NON_PREDICTIVE_STEMS:
        return True
    if re.fullmatch(
        r"(?:(?:train|test|validation|valid|holdout)_)?(?:labels?|targets?|answers?|ground_truth)",
        normalized,
    ):
        return True
    if normalized in _OUTPUT_MIRROR_STEMS:
        return True
    return normalized.startswith("sample_submission") or normalized.startswith(
        "sample_output"
    )


def _table_feature_columns(
    item: Mapping[str, Any], output_columns: set[str]
) -> list[str] | None:
    """Return the predictive feature columns of a table file.

    Returns None when the table has no inspectable column profile (unknown;
    caller keeps its conservative default), and an empty list when every
    non-identifier column is a target column or a mirror of the sample
    submission output columns (an index/label table, not a feature modality).
    """
    profile = item.get("profile")
    columns = profile.get("columns") if isinstance(profile, Mapping) else None
    if not isinstance(columns, list) or not columns:
        return None
    features: list[str] = []
    for column in columns:
        normalized = _normalized_column(column)
        if _column_is_identifier(column) or _column_is_target_like(column):
            continue
        if normalized in output_columns:
            continue
        features.append(str(column))
    return features


def predictive_modality_inventory(
    files: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return data modalities while excluding instructions and output templates.

    A table only counts as a predictive `tabular` modality when it exposes at
    least one feature column: a non-identifier, non-target column that is not a
    mirror of the sample-submission output columns. Index/label tables (e.g.
    `train.csv` holding only `image_id` plus the submission label columns) are
    annotations of another modality, not an independent predictive modality, so
    an image task with a label CSV is never treated as multimodal.
    """
    output_columns: set[str] = set()
    inspectable: dict[str, Mapping[str, Any]] = {}
    for item in files:
        path_value = str(item.get("path") or "")
        profile = item.get("profile")
        columns = profile.get("columns") if isinstance(profile, Mapping) else None
        if not isinstance(columns, list):
            continue
        normalized_stem = _normalized_column(Path(path_value).stem)
        if normalized_stem in _OUTPUT_MIRROR_STEMS or normalized_stem.startswith(
            ("sample_submission", "sample_output")
        ):
            output_columns.update(_normalized_column(column) for column in columns)
        inspectable[path_value] = item
    counts: Counter[str] = Counter()
    for item in files:
        if _looks_non_predictive(item.get("path")):
            continue
        modality = _KIND_TO_MODALITY.get(str(item.get("kind") or "").casefold())
        if not modality:
            continue
        if modality == "tabular":
            inspectable_item = inspectable.get(str(item.get("path") or ""))
            if inspectable_item is not None:
                features = _table_feature_columns(inspectable_item, output_columns)
                if features is not None and not features:
                    continue
        counts[modality] += 1
    modalities = sorted(counts)
    return {
        "modalities": modalities,
        "file_counts": dict(sorted(counts.items())),
        "is_multimodal": len(modalities) > 1,
    }


def transfer_learning_applicable(modalities: Iterable[str]) -> bool:
    """Return True if the modalities justify transfer learning (image, text, audio, video)."""
    applicable = {"image", "text", "audio", "video", "structured_text"}
    return any(str(mod).casefold() in applicable for mod in modalities)


def validate_modality_ablation_report(
    payload: Mapping[str, Any],
    modalities: Iterable[str],
    *,
    expected_folds: int | None = None,
) -> list[str]:
    """Require full, single-source, and leave-one-out evidence on identical folds."""
    expected = {str(item) for item in modalities if str(item)}
    if len(expected) <= 1:
        return []
    records = payload.get("modality_ablation_scores")
    if not isinstance(records, list) or not records:
        return [
            "result.json must contain non-empty modality_ablation_scores for a multimodal task"
        ]
    observed: list[set[str]] = []
    validation_hashes: set[str] = set()
    errors: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(f"modality_ablation_scores[{index}] must be an object")
            continue
        used = record.get("modalities")
        if not isinstance(used, list) or not used:
            errors.append(
                f"modality_ablation_scores[{index}].modalities must be a non-empty list"
            )
            continue
        used_set = {str(item) for item in used}
        unknown = used_set - expected
        if unknown:
            errors.append(
                f"modality_ablation_scores[{index}] contains unknown modalities: "
                + ", ".join(sorted(unknown))
            )
        try:
            score = record.get("score")
            if score is None or isinstance(score, bool) or not math.isfinite(float(score)):
                raise ValueError("non-finite score")
        except (TypeError, ValueError):
            errors.append(f"modality_ablation_scores[{index}].score must be finite")
        fold_scores = record.get("fold_scores")
        required_folds = max(1, int(expected_folds)) if expected_folds is not None else None
        if not isinstance(fold_scores, list) or not fold_scores:
            errors.append(
                f"modality_ablation_scores[{index}].fold_scores must be a non-empty list"
            )
        else:
            if required_folds is not None and len(fold_scores) != required_folds:
                errors.append(
                    f"modality_ablation_scores[{index}].fold_scores must contain "
                    f"exactly {required_folds} values"
                )
            try:
                if not all(
                    not isinstance(value, bool) and math.isfinite(float(value))
                    for value in fold_scores
                ):
                    raise ValueError("non-finite fold score")
            except (TypeError, ValueError):
                errors.append(
                    f"modality_ablation_scores[{index}].fold_scores must be finite"
                )
        validation_hash = str(record.get("validation_indices_hash") or "").strip()
        if not validation_hash:
            errors.append(
                f"modality_ablation_scores[{index}].validation_indices_hash is required"
            )
        else:
            validation_hashes.add(validation_hash)
        observed.append(used_set)
    if expected not in observed:
        errors.append("modality_ablation_scores must include the all-modalities model")
    for modality in sorted(expected):
        if {modality} not in observed:
            errors.append(
                f"modality_ablation_scores must include the {modality!r}-only model"
            )
        leave_one_out = expected - {modality}
        if leave_one_out and leave_one_out not in observed:
            errors.append(
                f"modality_ablation_scores must include the leave-one-out comparison "
                f"excluding {modality!r}"
            )
    if len(validation_hashes) > 1:
        errors.append(
            "all modality_ablation_scores must use the same validation_indices_hash"
        )
    return errors
