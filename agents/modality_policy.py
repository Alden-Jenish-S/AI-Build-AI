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
    return normalized.startswith("sample_submission") or normalized.startswith(
        "sample_output"
    )


def predictive_modality_inventory(
    files: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return data modalities while excluding instructions and output templates."""
    counts: Counter[str] = Counter()
    for item in files:
        if _looks_non_predictive(item.get("path")):
            continue
        modality = _KIND_TO_MODALITY.get(str(item.get("kind") or "").casefold())
        if modality:
            counts[modality] += 1
    modalities = sorted(counts)
    return {
        "modalities": modalities,
        "file_counts": dict(sorted(counts.items())),
        "is_multimodal": len(modalities) > 1,
    }


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
