"""Output-aware metric computation shared across modalities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def infer_metric_direction(metric_name: str) -> str:
    """Determine the optimization direction (maximize or minimize) for a metric."""
    metric = str(metric_name).strip().lower()
    if any(
        m in metric
        for m in ("mae", "rmse", "mse", "loss", "logloss", "error", "distance")
    ):
        return "minimize"
    return "maximize"


def _labels(prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(prediction)
    if values.ndim == 2 and values.shape[1] > 1:
        return np.argmax(values, axis=1)
    values = values.reshape(-1)
    if np.issubdtype(values.dtype, np.floating):
        return (values >= 0.5).astype(int)
    return values


def metric_value(
    metric_name: str,
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate classification/regression metrics for typed predictions."""
    metric = str(metric_name).strip().lower()
    truth = np.asarray(target)
    predicted = np.asarray(prediction)
    if not np.isfinite(np.asarray(predicted, dtype=float)).all():
        raise ValueError("predictions contain non-finite values")
    if "auc" in metric:
        if predicted.ndim == 2 and predicted.shape[1] > 2:
            return float(
                roc_auc_score(
                    truth, predicted, multi_class="ovr", average="macro"
                )
            )
        scores = (
            predicted[:, 1]
            if predicted.ndim == 2 and predicted.shape[1] == 2
            else predicted.reshape(-1)
        )
        return float(roc_auc_score(truth, scores))
    if "mae" in metric:
        return float(mean_absolute_error(truth, predicted.reshape(-1)))
    if "rmse" in metric:
        return float(
            mean_squared_error(truth, predicted.reshape(-1)) ** 0.5
        )
    if "accuracy" in metric:
        return float(accuracy_score(truth, _labels(predicted)))
    if "log_loss" in metric or "logloss" in metric:
        return float(log_loss(truth, predicted))
    if "f1" in metric:
        average = (
            "macro"
            if "macro" in metric or len(np.unique(truth)) > 2
            else "binary"
        )
        return float(f1_score(truth, _labels(predicted), average=average))
    if metric in {"r2", "r2_score"}:
        return float(r2_score(truth, predicted.reshape(-1)))
    raise ValueError(f"unsupported evaluation metric: {metric_name!r}")


def normalized_metric_value(
    metric_name: str,
    target: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Return a higher-is-better metric score for objective optimization."""
    val = metric_value(metric_name, target, prediction)
    direction = infer_metric_direction(metric_name)
    return val if direction == "maximize" else -val
