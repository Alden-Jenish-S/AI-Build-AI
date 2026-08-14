"""Capability-gated verification and blending of numeric prediction evidence.

The search core must not assume that every MLE task is classification, tabular,
or even supervised.  This module therefore recognizes a deliberately small,
explicit set of prediction/metric contracts.  Unknown metrics and task-native
artifacts remain valid search outputs, but they are not centrally rescored or
numerically ensembled.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def metric_token(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


_MINIMIZE_METRICS = {
    "logloss",
    "multiclasslogloss",
    "crossentropy",
    "meansquarederror",
    "mse",
    "rootmeansquarederror",
    "rmse",
    "meanabsoluteerror",
    "mae",
}
_MAXIMIZE_METRICS = {"accuracy", "classificationaccuracy", "r2", "r2score"}


def metric_is_supported(metric: object) -> bool:
    return metric_token(metric) in _MINIMIZE_METRICS | _MAXIMIZE_METRICS


def metric_direction(metric: object) -> str | None:
    token = metric_token(metric)
    if token in _MINIMIZE_METRICS:
        return "minimize"
    if token in _MAXIMIZE_METRICS:
        return "maximize"
    return None


def evaluate_predictions(metric: object, target: np.ndarray, prediction: np.ndarray) -> float:
    """Evaluate a metric only when its semantics are unambiguous.

    Ambiguous metrics (macro/micro F1, ranking metrics, task-native scores,
    structured generation, control rewards, and so on) intentionally raise
    ``ValueError`` so their declared evaluator remains authoritative.
    """
    token = metric_token(metric)
    y_true = np.asarray(target)
    y_pred = np.asarray(prediction)
    if len(y_true) != len(y_pred):
        raise ValueError("target and prediction row counts differ")

    if token in {"logloss", "multiclasslogloss", "crossentropy"}:
        from sklearn.metrics import log_loss

        if y_pred.ndim == 1:
            if np.any((y_pred < 0.0) | (y_pred > 1.0)):
                raise ValueError("binary probabilities fall outside [0, 1]")
            y_pred = np.column_stack((1.0 - y_pred, y_pred))
        if y_pred.ndim != 2 or y_pred.shape[1] < 2:
            raise ValueError("log loss requires a probability matrix")
        if np.any(y_pred < 0.0):
            raise ValueError("probabilities must be non-negative")
        row_sums = y_pred.sum(axis=1)
        if np.any(row_sums <= 0.0):
            raise ValueError("probability rows must have positive mass")
        normalized = np.clip(y_pred / row_sums[:, None], 1e-15, 1.0)
        normalized /= normalized.sum(axis=1, keepdims=True)
        labels = np.arange(normalized.shape[1])
        return float(log_loss(y_true, normalized, labels=labels))

    if token in {"meansquarederror", "mse", "rootmeansquarederror", "rmse"}:
        from sklearn.metrics import mean_squared_error

        value = float(mean_squared_error(y_true, y_pred))
        return math.sqrt(value) if token in {"rootmeansquarederror", "rmse"} else value
    if token in {"meanabsoluteerror", "mae"}:
        from sklearn.metrics import mean_absolute_error

        return float(mean_absolute_error(y_true, y_pred))
    if token in {"r2", "r2score"}:
        from sklearn.metrics import r2_score

        return float(r2_score(y_true, y_pred))
    if token in {"accuracy", "classificationaccuracy"}:
        from sklearn.metrics import accuracy_score

        labels = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
        return float(accuracy_score(y_true, labels))
    raise ValueError(f"metric {metric!r} has no unambiguous central evaluator")


@dataclass
class PredictionEvidence:
    path: Path
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    oof_pred: np.ndarray | None = None
    oof_target: np.ndarray | None = None
    oof_index: np.ndarray | None = None
    oof_fold: np.ndarray | None = None
    test_pred: np.ndarray | None = None
    test_index: np.ndarray | None = None
    score: float | None = None
    fold_scores: list[float] = field(default_factory=list)
    prediction_hash: str | None = None

    @property
    def centrally_scored(self) -> bool:
        return self.valid and self.score is not None

    @property
    def blendable(self) -> bool:
        return bool(
            self.centrally_scored
            and self.oof_pred is not None
            and self.oof_target is not None
            and self.oof_index is not None
            and self.test_pred is not None
            and self.test_index is not None
        )

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "centrally_scored": self.centrally_scored,
            "blendable": self.blendable,
            "score": self.score,
            "fold_scores": list(self.fold_scores),
            "prediction_hash": self.prediction_hash,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "oof_shape": list(self.oof_pred.shape) if self.oof_pred is not None else None,
            "test_shape": list(self.test_pred.shape) if self.test_pred is not None else None,
        }


def inspect_prediction_evidence(path: str | Path, metric: object) -> PredictionEvidence:
    source = Path(path)
    evidence = PredictionEvidence(path=source, valid=False)
    if not source.is_file():
        evidence.errors.append("prediction evidence file is missing")
        return evidence
    try:
        with np.load(source, allow_pickle=False) as payload:
            keys = set(payload.files)
            required = {"oof_pred", "oof_index"}
            missing = sorted(required - keys)
            if missing:
                evidence.errors.append(
                    "prediction evidence is missing " + ", ".join(missing)
                )
                return evidence
            evidence.oof_pred = np.asarray(payload["oof_pred"])
            evidence.oof_index = np.asarray(payload["oof_index"])
            evidence.oof_target = (
                np.asarray(payload["oof_target"]) if "oof_target" in keys else None
            )
            evidence.oof_fold = (
                np.asarray(payload["oof_fold"]) if "oof_fold" in keys else None
            )
            evidence.test_pred = (
                np.asarray(payload["test_pred"]) if "test_pred" in keys else None
            )
            evidence.test_index = (
                np.asarray(payload["test_index"]) if "test_index" in keys else None
            )
    except Exception as exc:
        evidence.errors.append(f"could not read prediction evidence: {type(exc).__name__}: {exc}")
        return evidence

    row_count = len(evidence.oof_index)
    if evidence.oof_pred.ndim not in {1, 2}:
        evidence.errors.append("oof_pred must be a one- or two-dimensional array")
    elif len(evidence.oof_pred) != row_count:
        evidence.errors.append("oof_pred and oof_index row counts differ")
    if row_count == 0:
        evidence.errors.append("prediction evidence contains no validation rows")
    if not np.issubdtype(evidence.oof_pred.dtype, np.number):
        evidence.errors.append("oof_pred must be numeric")
    elif not np.all(np.isfinite(evidence.oof_pred)):
        evidence.errors.append("oof_pred contains non-finite values")
    try:
        if len(np.unique(evidence.oof_index)) != row_count:
            evidence.errors.append("oof_index contains duplicate rows")
    except TypeError:
        evidence.errors.append("oof_index values are not comparable")

    for name, values in (
        ("oof_target", evidence.oof_target),
        ("oof_fold", evidence.oof_fold),
    ):
        if values is not None and len(values) != row_count:
            evidence.errors.append(f"{name} and oof_index row counts differ")
    if (evidence.test_pred is None) != (evidence.test_index is None):
        evidence.errors.append("test_pred and test_index must be stored together")
    if evidence.test_pred is not None:
        if evidence.test_pred.ndim not in {1, 2}:
            evidence.errors.append("test_pred must be a one- or two-dimensional array")
        elif len(evidence.test_pred) != len(evidence.test_index):
            evidence.errors.append("test_pred and test_index row counts differ")
        if not np.issubdtype(evidence.test_pred.dtype, np.number):
            evidence.errors.append("test_pred must be numeric")
        elif not np.all(np.isfinite(evidence.test_pred)):
            evidence.errors.append("test_pred contains non-finite values")
        if evidence.oof_pred.ndim == evidence.test_pred.ndim == 2 and (
            evidence.oof_pred.shape[1] != evidence.test_pred.shape[1]
        ):
            evidence.errors.append("OOF and test prediction widths differ")
        try:
            if len(np.unique(evidence.test_index)) != len(evidence.test_index):
                evidence.errors.append("test_index contains duplicate rows")
        except TypeError:
            evidence.errors.append("test_index values are not comparable")

    if evidence.errors:
        return evidence
    evidence.valid = True
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(evidence.oof_pred).tobytes())
    if evidence.test_pred is not None:
        digest.update(np.ascontiguousarray(evidence.test_pred).tobytes())
    evidence.prediction_hash = digest.hexdigest()

    if evidence.oof_target is None:
        evidence.warnings.append(
            "oof_target is absent; score cannot be independently recomputed"
        )
    elif not metric_is_supported(metric):
        evidence.warnings.append(
            f"metric {metric!r} uses its task-native evaluator and cannot be centrally recomputed"
        )
    else:
        try:
            evidence.score = evaluate_predictions(
                metric, evidence.oof_target, evidence.oof_pred
            )
            if evidence.oof_fold is not None:
                for fold in np.unique(evidence.oof_fold):
                    mask = evidence.oof_fold == fold
                    evidence.fold_scores.append(
                        evaluate_predictions(
                            metric,
                            evidence.oof_target[mask],
                            evidence.oof_pred[mask],
                        )
                    )
        except (TypeError, ValueError) as exc:
            evidence.errors.append(str(exc))
            evidence.valid = False
    return evidence


def evidence_compatible(items: Iterable[PredictionEvidence]) -> bool:
    values = list(items)
    if len(values) < 2 or not all(item.blendable for item in values):
        return False
    first = values[0]
    for item in values[1:]:
        if (
            item.oof_pred.shape != first.oof_pred.shape
            or item.test_pred.shape != first.test_pred.shape
            or not np.array_equal(item.oof_index, first.oof_index)
            or not np.array_equal(item.oof_target, first.oof_target)
            or not np.array_equal(item.test_index, first.test_index)
            or (item.oof_fold is None) != (first.oof_fold is None)
            or (
                item.oof_fold is not None
                and not np.array_equal(item.oof_fold, first.oof_fold)
            )
        ):
            return False
    return True


@dataclass
class BlendResult:
    weights: np.ndarray
    score: float
    fold_scores: list[float]
    oof_pred: np.ndarray
    test_pred: np.ndarray


def cross_fitted_blend(
    items: list[PredictionEvidence],
    metric: object,
    direction: str,
    *,
    seed: int = 42,
    trials: int = 256,
) -> BlendResult:
    """Choose regularized non-trivial weights using outer cross-fitting."""
    if not evidence_compatible(items):
        raise ValueError("prediction evidence sets are not blend-compatible")
    prediction_stack = np.stack([item.oof_pred for item in items], axis=0)
    test_stack = np.stack([item.test_pred for item in items], axis=0)
    target = items[0].oof_target
    count = len(items)

    folds = items[0].oof_fold
    if folds is None or len(np.unique(folds)) < 2:
        # Generic deterministic meta-folds. They are used only to estimate blend
        # weights; the underlying predictions remain honest OOF predictions.
        order = np.argsort(np.asarray(items[0].oof_index).astype(str), kind="stable")
        folds = np.empty(len(order), dtype=np.int64)
        folds[order] = np.arange(len(order)) % min(5, max(2, len(order) // 20))
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("at least two meta-folds are required to fit blend weights")

    rng = np.random.default_rng(seed)
    candidates: list[np.ndarray] = [np.full(count, 1.0 / count)]
    for left in range(count):
        for right in range(left + 1, count):
            for left_weight in np.linspace(0.05, 0.95, 19):
                weights = np.zeros(count)
                weights[left] = left_weight
                weights[right] = 1.0 - left_weight
                candidates.append(weights)
    for _ in range(max(0, int(trials))):
        weights = rng.dirichlet(np.full(count, 1.5))
        if np.count_nonzero(weights >= 0.05) >= 2:
            candidates.append(weights)

    def better(left: float, right: float) -> bool:
        return left < right if direction == "minimize" else left > right

    crossfit = np.zeros_like(items[0].oof_pred, dtype=float)
    selected: list[np.ndarray] = []
    fold_scores: list[float] = []
    for fold in unique_folds:
        validation_mask = folds == fold
        training_mask = ~validation_mask
        best_weights: np.ndarray | None = None
        best_score: float | None = None
        for weights in candidates:
            prediction = np.tensordot(
                weights, prediction_stack[:, training_mask], axes=(0, 0)
            )
            score = evaluate_predictions(metric, target[training_mask], prediction)
            # A small shrinkage term prevents unstable near-one-hot stacking.
            penalty = 1e-3 * float(np.sum((weights - 1.0 / count) ** 2))
            objective = score + penalty if direction == "minimize" else score - penalty
            if best_score is None or better(objective, best_score):
                best_score = objective
                best_weights = weights
        assert best_weights is not None
        selected.append(best_weights)
        fold_prediction = np.tensordot(
            best_weights, prediction_stack[:, validation_mask], axes=(0, 0)
        )
        crossfit[validation_mask] = fold_prediction
        fold_scores.append(
            evaluate_predictions(metric, target[validation_mask], fold_prediction)
        )

    weights = np.mean(selected, axis=0)
    weights[weights < 0.05] = 0.0
    if np.count_nonzero(weights) < 2:
        strongest = np.argsort(weights)[-2:]
        weights[:] = 0.0
        weights[strongest[-1]] = 0.95
        weights[strongest[-2]] = 0.05
    weights /= weights.sum()
    test_prediction = np.tensordot(weights, test_stack, axes=(0, 0))
    return BlendResult(
        weights=weights,
        score=evaluate_predictions(metric, target, crossfit),
        fold_scores=fold_scores,
        oof_pred=crossfit,
        test_pred=test_prediction,
    )
