"""Harness-side compact error summaries for evidence-driven follow-ups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    infer_metric_direction,
    metric_value,
    resolve_metric_name,
)
from .prediction_io import (
    legacy_prediction_payload,
    load_prediction_bundle,
    load_prediction_table,
)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _evaluation_frame(root: Path, mode: str) -> pd.DataFrame | None:
    if mode == "cross_validation":
        try:
            return load_prediction_table(root / "oof_predictions")
        except FileNotFoundError:
            return None
    try:
        frame = load_prediction_table(root / "validation_predictions")
    except FileNotFoundError:
        return None
    if mode != "holdout":
        return frame
    proof_path = root / ".evaluation_contract" / "validation_targets.npz"
    if not proof_path.is_file() or "row_id" not in frame.columns:
        return None
    with np.load(proof_path, allow_pickle=False) as proof:
        expected_ids = proof["row_ids"].astype(str).tolist()
        targets = proof["targets"]
    aligned = frame.set_index("row_id").reindex(expected_ids).reset_index()
    if len(aligned) != len(targets):
        return None
    aligned["target"] = targets
    return aligned


def _typed_evaluation_payload(
    root: Path,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    manifest = root / "predictions" / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        bundle, predictions, targets, _ = load_prediction_bundle(manifest)
    except (FileNotFoundError, TypeError, ValueError):
        return None
    sample_ids = np.asarray(bundle.sample_ids, dtype=str)
    if mode != "holdout":
        if targets is None:
            return None
        return sample_ids, np.asarray(targets), np.asarray(predictions)
    proof_path = root / ".evaluation_contract" / "validation_targets.npz"
    if not proof_path.is_file():
        return None
    with np.load(proof_path, allow_pickle=False) as proof:
        proof_ids = proof["row_ids"].astype(str)
        proof_targets = proof["targets"]
    positions = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    if set(positions) != set(proof_ids.tolist()):
        return None
    order = np.asarray([positions[sample_id] for sample_id in proof_ids], dtype=int)
    return proof_ids, proof_targets, np.asarray(predictions)[order]


def _structured_error_analysis(
    sample_ids: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    *,
    metric_name: str,
    problem_type: str,
    max_examples: int,
) -> dict[str, object]:
    resolved_metric = resolve_metric_name(
        metric_name or "score",
        problem_type=problem_type,
    )
    sample_scores = []
    invalid = 0
    for index in range(len(sample_ids)):
        try:
            sample_scores.append(
                float(
                    metric_value(
                        resolved_metric,
                        targets[index : index + 1],
                        predictions[index : index + 1],
                    )
                )
            )
        except (IndexError, TypeError, ValueError):
            sample_scores.append(float("nan"))
            invalid += 1
    scores = np.asarray(sample_scores, dtype=float)
    finite = np.isfinite(scores)
    if not finite.any():
        return {
            "available": False,
            "reason": "structured per-sample metrics could not be computed",
        }
    direction = infer_metric_direction(resolved_metric)
    priority = np.where(
        finite,
        scores if direction == "minimize" else -scores,
        -np.inf,
    )
    order = np.argsort(-priority)[: max(1, int(max_examples))]
    details: dict[str, object] = {
        "available": True,
        "row_count": len(sample_ids),
        "metric": resolved_metric,
        "metric_direction": direction,
        "invalid_sample_count": invalid,
        "per_sample_score_quantiles": {
            str(quantile): float(np.quantile(scores[finite], quantile))
            for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "worst_examples": [
            {
                "row_id": str(sample_ids[index]),
                "sample_score": float(scores[index]),
            }
            for index in order
            if np.isfinite(scores[index])
        ],
    }
    if problem_type == "segmentation" and np.asarray(targets).ndim >= 3:
        truth = np.asarray(targets) > 0
        predicted = np.asarray(predictions)
        if predicted.shape == truth.shape:
            predicted = predicted >= 0.5 if np.issubdtype(
                predicted.dtype, np.floating
            ) else predicted > 0
            spatial_axes = tuple(range(1, truth.ndim))
            details["empty_target_fraction"] = float(
                np.mean(~truth.any(axis=spatial_axes))
            )
            details["false_positive_pixel_rate"] = float(
                np.mean(np.logical_and(~truth, predicted))
            )
            details["false_negative_pixel_rate"] = float(
                np.mean(np.logical_and(truth, ~predicted))
            )
    return details


def build_error_analysis(
    node_dir: str | Path,
    *,
    evaluation_mode: str,
    problem_type: str,
    metric_name: str | None = None,
    max_examples: int = 20,
) -> dict[str, object]:
    """Summarize residual/error structure without reloading the dataset."""
    root = Path(node_dir)
    typed = _typed_evaluation_payload(root, evaluation_mode)
    if typed is not None:
        sample_ids, targets, predictions = typed
        structured = _structured_error_analysis(
            sample_ids,
            targets,
            predictions,
            metric_name=metric_name or "score",
            problem_type=problem_type,
            max_examples=max_examples,
        )
        return {
            "analysis_version": 2,
            "evaluation_mode": evaluation_mode,
            "problem_type": problem_type,
            **structured,
        }
    frame = _evaluation_frame(root, evaluation_mode)
    base: dict[str, object] = {
        "analysis_version": 1,
        "evaluation_mode": evaluation_mode,
        "problem_type": problem_type,
    }
    if frame is None or frame.empty:
        return {**base, "available": False, "reason": "no aligned predictions"}
    if evaluation_mode == "task_native":
        counts = frame.get("prediction", pd.Series(dtype=object)).value_counts()
        return {
            **base,
            "available": True,
            "row_count": len(frame),
            "cluster_size_distribution": {
                str(key): int(value) for key, value in counts.items()
            },
        }
    if "target" not in frame.columns or "row_id" not in frame.columns:
        return {**base, "available": False, "reason": "targets or row IDs missing"}
    try:
        prediction, class_names = legacy_prediction_payload(frame)
    except (TypeError, ValueError) as exc:
        return {**base, "available": False, "reason": str(exc)}
    target = frame["target"].to_numpy()
    row_ids = frame["row_id"].astype(str).to_numpy()
    if problem_type == "regression" and np.asarray(prediction).ndim == 1:
        try:
            numeric_target = np.asarray(target, dtype=float)
            numeric_prediction = np.asarray(prediction, dtype=float)
        except (TypeError, ValueError):
            return {**base, "available": False, "reason": "non-numeric regression output"}
        residual = numeric_target - numeric_prediction
        absolute = np.abs(residual)
        order = np.argsort(-absolute)[: max(1, int(max_examples))]
        return {
            **base,
            "available": True,
            "row_count": len(frame),
            "residual_bias": float(np.mean(residual)),
            "absolute_error_quantiles": {
                str(quantile): float(np.quantile(absolute, quantile))
                for quantile in (0.5, 0.75, 0.9, 0.95, 0.99)
            },
            "underprediction_rate": float(np.mean(residual > 0)),
            "worst_examples": [
                {
                    "row_id": row_ids[index],
                    "target": float(numeric_target[index]),
                    "prediction": float(numeric_prediction[index]),
                    "residual": float(residual[index]),
                    "absolute_error": float(absolute[index]),
                }
                for index in order
            ],
        }

    values = np.asarray(prediction)
    confidence = None
    if values.ndim == 2:
        winner = np.argmax(values, axis=1)
        labels = (
            np.asarray(class_names, dtype=object)[winner]
            if class_names
            else winner
        )
        confidence = np.max(values.astype(float), axis=1)
    else:
        labels = values.reshape(-1)
        unique_targets = list(pd.unique(pd.Series(target)))
        if (
            len(unique_targets) == 2
            and np.issubdtype(labels.dtype, np.number)
            and not set(pd.unique(labels)).issubset(set(unique_targets))
        ):
            confidence = np.maximum(labels.astype(float), 1.0 - labels.astype(float))
            labels = np.where(
                labels.astype(float) >= 0.5,
                unique_targets[1],
                unique_targets[0],
            )
    incorrect = np.asarray(labels).astype(str) != np.asarray(target).astype(str)
    by_class = {}
    target_series = pd.Series(target).astype(str)
    for label in sorted(target_series.unique()):
        mask = target_series.to_numpy() == label
        by_class[label] = {
            "count": int(mask.sum()),
            "error_rate": float(np.mean(incorrect[mask])),
        }
    if confidence is not None:
        priority = np.asarray(confidence) * incorrect.astype(float)
    else:
        priority = incorrect.astype(float)
    order = np.argsort(-priority)[: max(1, int(max_examples))]
    worst = [
        {
            "row_id": row_ids[index],
            "target": _json_value(target[index]),
            "prediction": _json_value(labels[index]),
            **(
                {"confidence": float(confidence[index])}
                if confidence is not None
                else {}
            ),
        }
        for index in order
        if incorrect[index]
    ]
    return {
        **base,
        "available": True,
        "row_count": len(frame),
        "error_rate": float(np.mean(incorrect)),
        "class_error_buckets": by_class,
        "confidence_quantiles": (
            {
                str(quantile): float(np.quantile(confidence, quantile))
                for quantile in (0.1, 0.5, 0.9)
            }
            if confidence is not None
            else None
        ),
        "worst_examples": worst,
    }
