"""Harness-side compact error summaries for evidence-driven follow-ups."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .prediction_io import legacy_prediction_payload


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _evaluation_frame(root: Path, mode: str) -> pd.DataFrame | None:
    if mode == "cross_validation":
        path = root / "oof_predictions.csv"
        return pd.read_csv(path) if path.is_file() else None
    path = root / "validation_predictions.csv"
    if not path.is_file():
        return None
    frame = pd.read_csv(path, dtype={"row_id": str})
    if mode != "holdout":
        return frame
    proof_path = root / ".evaluation_contract" / "validation_targets.json"
    if not proof_path.is_file() or "row_id" not in frame.columns:
        return None
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    expected_ids = [str(item) for item in proof.get("row_ids", [])]
    aligned = frame.set_index("row_id").reindex(expected_ids).reset_index()
    targets = proof.get("targets", [])
    if len(aligned) != len(targets):
        return None
    aligned["target"] = targets
    return aligned


def build_error_analysis(
    node_dir: str | Path,
    *,
    evaluation_mode: str,
    problem_type: str,
    max_examples: int = 20,
) -> dict[str, object]:
    """Summarize residual/error structure without reloading the dataset."""
    root = Path(node_dir)
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
