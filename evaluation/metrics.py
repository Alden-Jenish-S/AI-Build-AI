"""Task-aware metric resolution and computation shared across modalities."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)


_ALIASES = {
    "acc": "accuracy",
    "auc": "roc_auc",
    "average_precision_score": "average_precision",
    "balanced_acc": "balanced_accuracy",
    "bce": "log_loss",
    "bleu_score": "bleu",
    "dice_coefficient": "dice",
    "dice_score": "dice",
    "f1_score": "f1",
    "iou": "mean_iou",
    "jaccard": "mean_iou",
    "jaccard_index": "mean_iou",
    "logloss": "log_loss",
    "map": "mean_average_precision",
    "mean_ap": "mean_average_precision",
    "mean_average_precision_score": "mean_average_precision",
    "mask_average_precision": "segmentation_average_precision",
    "segmentation_map": "segmentation_average_precision",
    "mean_squared_error": "mse",
    "mean_absolute_error": "mae",
    "mrr": "mean_reciprocal_rank",
    "ndcg": "ndcg@10",
    "pixel_acc": "pixel_accuracy",
    "r2_score": "r2",
    "root_mean_squared_error": "rmse",
    "rouge": "rouge_l",
    "rouge_l_score": "rouge_l",
    "silhouette": "silhouette_score",
    "temporal_iou_score": "temporal_iou",
    "token_f1_score": "token_f1",
}
_PLACEHOLDER_METRICS = {
    "",
    "metric",
    "objective",
    "performance",
    "score",
    "validation_metric",
    "validation_score",
}
_MINIMIZE_METRICS = {
    "cross_entropy",
    "hamming_loss",
    "log_loss",
    "mae",
    "mape",
    "median_absolute_error",
    "mse",
    "msle",
    "rmse",
    "rmsle",
}
_CUSTOM_METRICS: dict[str, tuple[Callable[..., float], str]] = {}

_DESCRIPTION_METRIC_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\b(?:multi[- ]?class\s+)?(?:logarithmic|log)\s*loss\b"
        r"|\bnegative\s+log[- ]?likelihood\b"
        r"|\bcross[- ]?entropy\b",
        "log_loss",
    ),
    (
        r"\broot\s+mean\s+squared\s+logarithmic\s+error\b|\brmsle\b",
        "rmsle",
    ),
    (
        r"\broot\s+mean\s+squared\s+error\b|\brmse\b",
        "rmse",
    ),
    (
        r"\bmean\s+squared\s+logarithmic\s+error\b|\bmsle\b",
        "msle",
    ),
    (r"\bmean\s+squared\s+error\b|\bmse\b", "mse"),
    (
        r"\bmean\s+absolute\s+percentage\s+error\b|\bmape\b",
        "mape",
    ),
    (r"\bmean\s+absolute\s+error\b|\bmae\b", "mae"),
    (
        r"\bmean\s+average\s+precision\b|\bmean\s+ap\b|\bmAP\b",
        "mean_average_precision",
    ),
    (
        r"\baverage\s+precision\b|\barea\s+under\s+(?:the\s+)?"
        r"precision[- ]recall\s+curve\b",
        "average_precision",
    ),
    (
        r"\b(?:roc[- ]?)?auc\b|\barea\s+under\s+(?:the\s+)?"
        r"(?:receiver\s+operating\s+characteristic|roc)\s+curve\b",
        "roc_auc",
    ),
    (r"\bbalanced\s+accuracy\b", "balanced_accuracy"),
    (r"\bmacro[- ]?f1\b|\bf1[- ]?macro\b", "f1_macro"),
    (r"\bmicro[- ]?f1\b|\bf1[- ]?micro\b", "f1_micro"),
    (r"\bweighted[- ]?f1\b|\bf1[- ]?weighted\b", "f1_weighted"),
    (r"\bf1(?:[- ]?score)?\b", "f1"),
    (r"\btop[- ]?(\d+)\s+accuracy\b", "top_{group}_accuracy"),
    (
        r"\bnormalized\s+discounted\s+cumulative\s+gain"
        r"(?:\s+at\s+(\d+))?\b|\bndcg(?:@(\d+))?\b",
        "ndcg@{group}",
    ),
    (r"\bmean\s+reciprocal\s+rank\b|\bmrr\b", "mean_reciprocal_rank"),
    (r"\bdice(?:\s+coefficient|\s+score)?\b", "dice"),
    (
        r"\bintersection\s+over\s+union\b|\bmean\s+iou\b|\bmIoU\b",
        "mean_iou",
    ),
    (r"\bsilhouette(?:\s+score)?\b", "silhouette_score"),
    (r"\bcohen(?:'s)?\s+kappa\b", "cohen_kappa"),
    (
        r"\bmatthews\s+correlation\s+coefficient\b|\bmcc\b",
        "matthews_corrcoef",
    ),
    (r"\br[- ]?squared\b|\br2(?:[- ]?score)?\b", "r2"),
    (r"\brouge[- ]?l\b", "rouge_l"),
    (r"\bbleu(?:\s+score)?\b", "bleu"),
    (r"\bexact\s+match\b", "exact_match"),
    (r"\baccuracy\b", "accuracy"),
)


def canonical_metric_name(metric_name: object) -> str:
    """Return a stable metric identifier while retaining ``@k`` suffixes."""
    metric = (
        str(metric_name or "")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    metric = re.sub(r"_+", "_", metric)
    return _ALIASES.get(metric, metric)


def infer_metric_from_description(description: object) -> str | None:
    """Infer a supported metric from explicit natural-language evaluation text.

    This deliberately recognizes only concrete metric names. It does not guess
    from vague words such as "score" or override an explicit task contract.
    """
    text = str(description or "")
    if not text.strip():
        return None
    normalized = text.lower()
    if (
        ("average precision" in normalized or "mean ap" in normalized)
        and (
            "iou threshold" in normalized
            or "intersection over union" in normalized
        )
        and any(token in normalized for token in ("mask", "pixel", "segment"))
    ):
        return "segmentation_average_precision"
    for pattern, metric_template in _DESCRIPTION_METRIC_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        groups = tuple(value for value in match.groups() if value)
        group = groups[0] if groups else "10"
        return canonical_metric_name(metric_template.format(group=group))
    return None


def default_metric_for_problem(
    problem_type: object,
    output_type: object | None = None,
) -> str:
    """Choose a concrete, locally recomputable default for a task contract."""
    problem = str(problem_type or "supervised").strip().lower().replace("-", "_")
    output = str(output_type or "").strip().lower().replace("-", "_")
    defaults = {
        "captioning": "token_f1",
        "classification": "accuracy",
        "detection": "box_iou",
        "multilabel_classification": "f1_macro",
        "regression": "rmse",
        "retrieval": "ndcg@10",
        "segmentation": "dice",
        "temporal_localization": "temporal_iou",
        "unsupervised_clustering": "silhouette_score",
    }
    if problem in defaults:
        return defaults[problem]
    output_defaults = {
        "boxes": "box_iou",
        "class_probabilities": "accuracy",
        "continuous": "rmse",
        "labels": "accuracy",
        "masks": "dice",
        "ranked_items": "ndcg@10",
        "text": "token_f1",
    }
    return output_defaults.get(output, "accuracy")


def resolve_metric_name(
    metric_name: object,
    *,
    problem_type: object | None = None,
    output_type: object | None = None,
) -> str:
    """Resolve legacy placeholders such as ``score`` to a real task metric."""
    metric = canonical_metric_name(metric_name)
    if metric in _PLACEHOLDER_METRICS:
        return default_metric_for_problem(problem_type, output_type)
    return metric


def infer_metric_direction(metric_name: object) -> str:
    """Determine the natural optimization direction for a metric."""
    metric = canonical_metric_name(metric_name)
    if metric in _CUSTOM_METRICS:
        return _CUSTOM_METRICS[metric][1]
    if (
        metric in _MINIMIZE_METRICS
        or metric.endswith("_loss")
        or metric.endswith("_error")
        or any(
            token in metric
            for token in ("distance", "deviance", "perplexity")
        )
    ):
        return "minimize"
    return "maximize"


def register_metric(
    name: str,
    evaluator: Callable[..., float],
    *,
    direction: str,
    aliases: Iterable[str] = (),
) -> None:
    """Register a project-specific metric without changing orchestration code."""
    canonical = canonical_metric_name(name)
    normalized_direction = str(direction).strip().lower()
    if not canonical or canonical in _PLACEHOLDER_METRICS:
        raise ValueError("custom metric name must be concrete and non-empty")
    if normalized_direction not in {"maximize", "minimize"}:
        raise ValueError("custom metric direction must be maximize or minimize")
    if not callable(evaluator):
        raise TypeError("custom metric evaluator must be callable")
    _CUSTOM_METRICS[canonical] = (evaluator, normalized_direction)
    for alias in aliases:
        normalized_alias = canonical_metric_name(alias)
        if normalized_alias:
            _ALIASES[normalized_alias] = canonical


def _ensure_aligned(truth: np.ndarray, predicted: np.ndarray) -> None:
    if truth.ndim == 0 or predicted.ndim == 0:
        raise ValueError("targets and predictions must have a sample axis")
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError(
            "targets and predictions must contain the same number of samples"
        )


def _ensure_finite(values: np.ndarray, field: str) -> None:
    if np.issubdtype(values.dtype, np.number):
        if not np.isfinite(values.astype(float, copy=False)).all():
            raise ValueError(f"{field} contain non-finite values")


def _class_labels(
    prediction: np.ndarray,
    *,
    class_names: Sequence[object] = (),
) -> np.ndarray:
    values = np.asarray(prediction)
    if values.ndim >= 2 and values.shape[-1] > 1:
        indices = np.argmax(values, axis=-1)
        if class_names and len(class_names) == values.shape[-1]:
            names = np.asarray(tuple(class_names), dtype=object)
            return names[indices]
        return indices
    values = values.reshape(-1)
    if np.issubdtype(values.dtype, np.floating):
        return (values >= 0.5).astype(int)
    return values


def _compatible_class_names(
    class_names: Sequence[object], truth: np.ndarray
) -> tuple[object, ...]:
    if not class_names:
        return ()
    target = np.asarray(truth).reshape(-1)
    if np.issubdtype(target.dtype, np.number):
        try:
            return tuple(np.asarray(class_names, dtype=target.dtype).tolist())
        except (TypeError, ValueError):
            return tuple(class_names)
    return tuple(str(item) for item in class_names)


def _classification_predictions(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    class_names: Sequence[object] = (),
) -> np.ndarray:
    values = np.asarray(prediction)
    target = np.asarray(truth)
    if target.ndim > 1 and values.shape == target.shape:
        return (
            (values >= 0.5).astype(int)
            if np.issubdtype(values.dtype, np.floating)
            else values
        )
    return _class_labels(values, class_names=class_names)


def _classification_average(metric: str, truth: np.ndarray) -> str:
    for average in ("micro", "macro", "weighted", "samples"):
        if metric.endswith(f"_{average}") or f"@{average}" in metric:
            return average
    return "binary" if len(np.unique(truth)) <= 2 else "macro"


def _ranking_k(metric: str, default: int | None = None) -> int | None:
    match = re.search(r"@(\d+)$", metric)
    if match:
        return max(1, int(match.group(1)))
    return default


def _as_item_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, str):
        return value.split()
    return [value]


def _relevance_set(value: Any) -> set[Any]:
    return set(_as_item_sequence(value))


def _retrieval_scores(
    truth: np.ndarray,
    predicted: np.ndarray,
    metric: str,
) -> float:
    targets = [_relevance_set(item) for item in truth]
    rankings = [_as_item_sequence(item) for item in predicted]
    if len(targets) != len(rankings):
        raise ValueError("retrieval targets and rankings must align")
    k = _ranking_k(metric)
    values = []
    for relevant, ranking in zip(targets, rankings):
        ranked = ranking[:k] if k is not None else ranking
        hits = np.asarray([item in relevant for item in ranked], dtype=float)
        if metric.startswith("precision"):
            values.append(float(hits.sum() / max(len(ranked), 1)))
        elif metric.startswith("recall"):
            values.append(float(hits.sum() / max(len(relevant), 1)))
        elif metric.startswith("hit_rate"):
            values.append(float(bool(hits.any())))
        elif metric == "mean_reciprocal_rank":
            hit_indices = np.flatnonzero(hits)
            values.append(
                0.0 if not len(hit_indices) else 1.0 / (int(hit_indices[0]) + 1)
            )
        elif metric == "mean_average_precision" or metric.startswith("map@"):
            precisions = [
                hits[: index + 1].sum() / (index + 1)
                for index in np.flatnonzero(hits)
            ]
            values.append(
                float(sum(precisions) / max(len(relevant), 1))
            )
        elif metric.startswith("ndcg"):
            discounts = 1.0 / np.log2(np.arange(len(hits)) + 2.0)
            dcg = float((hits * discounts).sum())
            ideal_count = min(len(relevant), len(hits))
            idcg = float(discounts[:ideal_count].sum())
            values.append(dcg / idcg if idcg > 0 else 0.0)
        else:
            raise ValueError(f"unsupported retrieval metric: {metric!r}")
    return float(np.mean(values)) if values else 0.0


def _tokens(value: Any) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_token == right_token
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _text_score(truth: np.ndarray, predicted: np.ndarray, metric: str) -> float:
    values = []
    for target, prediction in zip(truth.reshape(-1), predicted.reshape(-1)):
        target_tokens = _tokens(target)
        prediction_tokens = _tokens(prediction)
        if metric == "exact_match":
            values.append(float(str(target).strip() == str(prediction).strip()))
            continue
        if metric == "rouge_l":
            overlap = _lcs_length(target_tokens, prediction_tokens)
        else:
            target_counts: dict[str, int] = {}
            for token in target_tokens:
                target_counts[token] = target_counts.get(token, 0) + 1
            overlap = 0
            for token in prediction_tokens:
                if target_counts.get(token, 0) > 0:
                    overlap += 1
                    target_counts[token] -= 1
        precision = overlap / max(len(prediction_tokens), 1)
        recall = overlap / max(len(target_tokens), 1)
        if metric in {"token_f1", "rouge_l"}:
            values.append(
                0.0
                if precision + recall == 0
                else 2.0 * precision * recall / (precision + recall)
            )
        elif metric == "bleu":
            if not prediction_tokens:
                values.append(0.0)
            else:
                brevity = min(
                    1.0,
                    math.exp(
                        1.0
                        - len(target_tokens) / max(len(prediction_tokens), 1)
                    ),
                )
                values.append(brevity * precision)
        else:
            raise ValueError(f"unsupported text metric: {metric!r}")
    return float(np.mean(values)) if values else 0.0


def _mask_labels(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    values = np.asarray(predicted)
    if values.shape == truth.shape:
        if np.issubdtype(values.dtype, np.floating):
            return (values >= 0.5).astype(int)
        return values
    if (
        values.ndim == truth.ndim + 1
        and values.shape[0] == truth.shape[0]
    ):
        # Support both channel-first and channel-last spatial logits.
        if values.shape[2:] == truth.shape[1:]:
            if values.shape[1] == 1:
                squeezed = values[:, 0]
                return (
                    (squeezed >= 0.5).astype(int)
                    if np.issubdtype(squeezed.dtype, np.floating)
                    else squeezed
                )
            return np.argmax(values, axis=1)
        if values.shape[1:-1] == truth.shape[1:]:
            if values.shape[-1] == 1:
                squeezed = values[..., 0]
                return (
                    (squeezed >= 0.5).astype(int)
                    if np.issubdtype(squeezed.dtype, np.floating)
                    else squeezed
                )
            return np.argmax(values, axis=-1)
    raise ValueError("segmentation predictions do not align with target masks")


def _overlap_score(
    truth: np.ndarray,
    predicted: np.ndarray,
    metric: str,
) -> float:
    labels = _mask_labels(predicted, truth)
    truth_flat = truth.reshape(-1)
    predicted_flat = labels.reshape(-1)
    if metric == "pixel_accuracy":
        return float(np.mean(truth_flat == predicted_flat))
    classes = np.union1d(truth_flat, predicted_flat)
    per_class = []
    for value in classes:
        target_mask = truth_flat == value
        prediction_mask = predicted_flat == value
        intersection = int(np.logical_and(target_mask, prediction_mask).sum())
        if metric == "dice":
            denominator = int(target_mask.sum() + prediction_mask.sum())
            per_class.append(1.0 if denominator == 0 else 2 * intersection / denominator)
        else:
            union = int(np.logical_or(target_mask, prediction_mask).sum())
            per_class.append(1.0 if union == 0 else intersection / union)
    return float(np.mean(per_class)) if per_class else 1.0


def _segmentation_average_precision(
    truth: np.ndarray,
    predicted: np.ndarray,
    *,
    thresholds: Sequence[float] | None = None,
) -> float:
    """Mean per-image precision across binary-mask IoU thresholds.

    This is the common competition segmentation metric in which an image is a
    hit at a threshold only when its foreground IoU is strictly greater than
    that threshold.  Empty/empty masks are perfect matches; a one-sided empty
    mask scores zero.
    """
    target = np.asarray(truth)
    labels = _mask_labels(np.asarray(predicted), target)
    if target.shape != labels.shape or target.ndim < 2:
        raise ValueError(
            "segmentation average precision requires aligned mask arrays"
        )
    cutoffs = np.asarray(
        tuple(thresholds)
        if thresholds is not None
        else tuple(np.arange(0.50, 1.00, 0.05)),
        dtype=float,
    )
    if cutoffs.ndim != 1 or not len(cutoffs) or (
        (cutoffs < 0.0).any() or (cutoffs > 1.0).any()
    ):
        raise ValueError("segmentation IoU thresholds must be within [0, 1]")
    values = []
    for target_mask, predicted_mask in zip(target, labels):
        target_positive = np.asarray(target_mask) > 0
        predicted_positive = np.asarray(predicted_mask) > 0
        intersection = int(
            np.logical_and(target_positive, predicted_positive).sum()
        )
        union = int(np.logical_or(target_positive, predicted_positive).sum())
        iou = 1.0 if union == 0 else intersection / union
        values.append(float(np.mean(iou > cutoffs)))
    return float(np.mean(values)) if values else 0.0


def _box_iou(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if truth.shape != predicted.shape or truth.ndim < 2 or truth.shape[-1] != 4:
        raise ValueError(
            "box IoU requires aligned arrays with final dimension "
            "[x_min, y_min, x_max, y_max]"
        )
    left = np.maximum(truth[..., :2], predicted[..., :2])
    right = np.minimum(truth[..., 2:], predicted[..., 2:])
    intersection = np.prod(np.maximum(right - left, 0.0), axis=-1)
    truth_area = np.prod(np.maximum(truth[..., 2:] - truth[..., :2], 0.0), axis=-1)
    predicted_area = np.prod(
        np.maximum(predicted[..., 2:] - predicted[..., :2], 0.0), axis=-1
    )
    union = truth_area + predicted_area - intersection
    return float(np.mean(np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)))


def _temporal_iou(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if truth.shape != predicted.shape or truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError(
            "temporal IoU requires aligned [start, end] pairs"
        )
    intersection = np.maximum(
        0.0,
        np.minimum(truth[:, 1], predicted[:, 1])
        - np.maximum(truth[:, 0], predicted[:, 0]),
    )
    union = (
        np.maximum(truth[:, 1], predicted[:, 1])
        - np.minimum(truth[:, 0], predicted[:, 0])
    )
    return float(
        np.mean(
            np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
        )
    )


def metric_value(
    metric_name: object,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    class_names: Sequence[object] = (),
    options: dict[str, object] | None = None,
) -> float:
    """Calculate a metric from scalar, probabilistic, or structured outputs."""
    metric = canonical_metric_name(metric_name)
    truth = np.asarray(target)
    predicted = np.asarray(prediction)
    _ensure_aligned(truth, predicted)
    _ensure_finite(predicted, "predictions")
    options = dict(options or {})
    class_names = _compatible_class_names(class_names, truth)

    if metric in _CUSTOM_METRICS:
        value = float(
            _CUSTOM_METRICS[metric][0](
                truth,
                predicted,
                class_names=tuple(class_names),
                **options,
            )
        )
        if not math.isfinite(value):
            raise ValueError(f"custom metric {metric!r} returned a non-finite value")
        return value

    if metric == "accuracy":
        labels = _classification_predictions(
            truth, predicted, class_names=class_names
        )
        return float(np.mean(truth.reshape(-1) == labels.reshape(-1)))
    if metric == "balanced_accuracy":
        return float(
            balanced_accuracy_score(
                truth.reshape(-1),
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ).reshape(-1),
            )
        )
    if metric.startswith("top_") and metric.endswith("_accuracy"):
        k_match = re.match(r"top_(\d+)_accuracy", metric)
        k = int(k_match.group(1)) if k_match else int(options.get("k", 5))
        return float(
            top_k_accuracy_score(
                truth.reshape(-1),
                predicted,
                k=k,
                labels=list(class_names) or None,
            )
        )
    if "auc" in metric:
        if predicted.ndim == 2 and predicted.shape[1] > 2:
            return float(
                roc_auc_score(
                    truth,
                    predicted,
                    labels=list(class_names) or None,
                    multi_class="ovo" if "ovo" in metric else "ovr",
                    average=(
                        "weighted" if "weighted" in metric else "macro"
                    ),
                )
            )
        scores = (
            predicted[:, 1]
            if predicted.ndim == 2 and predicted.shape[1] == 2
            else predicted.reshape(-1)
        )
        return float(roc_auc_score(truth.reshape(-1), scores))
    if metric == "average_precision":
        return float(average_precision_score(truth, predicted))
    if metric == "segmentation_average_precision":
        return _segmentation_average_precision(
            truth,
            predicted,
            thresholds=options.get("thresholds"),
        )
    if metric in {"log_loss", "cross_entropy"}:
        probabilities = np.asarray(predicted, dtype=float)
        tolerance = 1e-12
        if (
            (probabilities < -tolerance).any()
            or (probabilities > 1.0 + tolerance).any()
        ):
            raise ValueError(
                f"{metric} requires probability predictions within [0, 1]"
            )
        probabilities = np.clip(probabilities, 0.0, 1.0)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            row_sums = probabilities.sum(axis=1, keepdims=True)
            if (row_sums <= 0).any():
                raise ValueError(
                    f"{metric} requires positive probability mass per row"
                )
            probabilities = probabilities / row_sums
        return float(
            log_loss(
                truth.reshape(-1),
                probabilities,
                labels=list(class_names) or None,
            )
        )
    if metric.startswith("f1"):
        return float(
            f1_score(
                truth,
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ),
                average=_classification_average(metric, truth),
                zero_division=0,
            )
        )
    if metric.startswith("precision") and not re.search(r"@\d+$", metric):
        return float(
            precision_score(
                truth,
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ),
                average=_classification_average(metric, truth),
                zero_division=0,
            )
        )
    if metric.startswith("recall") and not re.search(r"@\d+$", metric):
        return float(
            recall_score(
                truth,
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ),
                average=_classification_average(metric, truth),
                zero_division=0,
            )
        )
    if metric == "matthews_corrcoef":
        return float(
            matthews_corrcoef(
                truth.reshape(-1),
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ).reshape(-1),
            )
        )
    if metric == "cohen_kappa":
        return float(
            cohen_kappa_score(
                truth.reshape(-1),
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ).reshape(-1),
            )
        )
    if metric == "hamming_loss":
        return float(
            hamming_loss(
                truth,
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ),
            )
        )
    if metric.startswith("jaccard_"):
        return float(
            jaccard_score(
                truth,
                _classification_predictions(
                    truth, predicted, class_names=class_names
                ),
                average=_classification_average(metric, truth),
                zero_division=0,
            )
        )

    numeric_truth = None
    numeric_prediction = None
    if metric in {
        "mae",
        "mape",
        "median_absolute_error",
        "mse",
        "msle",
        "rmse",
        "rmsle",
        "r2",
        "explained_variance",
    }:
        numeric_truth = np.asarray(truth, dtype=float)
        numeric_prediction = np.asarray(predicted, dtype=float)
        if numeric_truth.shape != numeric_prediction.shape:
            numeric_prediction = numeric_prediction.reshape(numeric_truth.shape)
    if metric == "mae":
        return float(mean_absolute_error(numeric_truth, numeric_prediction))
    if metric == "mape":
        return float(
            mean_absolute_percentage_error(numeric_truth, numeric_prediction)
        )
    if metric == "median_absolute_error":
        return float(median_absolute_error(numeric_truth, numeric_prediction))
    if metric == "mse":
        return float(mean_squared_error(numeric_truth, numeric_prediction))
    if metric == "rmse":
        return float(
            mean_squared_error(numeric_truth, numeric_prediction) ** 0.5
        )
    if metric == "msle":
        return float(
            mean_squared_log_error(numeric_truth, numeric_prediction)
        )
    if metric == "rmsle":
        return float(
            mean_squared_log_error(numeric_truth, numeric_prediction) ** 0.5
        )
    if metric == "r2":
        return float(r2_score(numeric_truth, numeric_prediction))
    if metric == "explained_variance":
        return float(
            explained_variance_score(numeric_truth, numeric_prediction)
        )

    if (
        metric.startswith(("precision@", "recall@", "hit_rate@", "ndcg", "map@"))
        or metric in {"mean_average_precision", "mean_reciprocal_rank"}
    ):
        return _retrieval_scores(truth, predicted, metric)
    if metric in {"bleu", "exact_match", "rouge_l", "token_f1"}:
        return _text_score(truth, predicted, metric)
    if metric in {"dice", "mean_iou", "pixel_accuracy"}:
        return _overlap_score(truth, predicted, metric)
    if metric in {"box_iou"}:
        return _box_iou(truth, predicted)
    if metric == "temporal_iou":
        return _temporal_iou(truth, predicted)

    raise ValueError(
        f"unsupported evaluation metric: {metric_name!r}. Configure a built-in "
        "metric or register a project-specific evaluator with register_metric()."
    )


def normalized_metric_value(
    metric_name: object,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    class_names: Sequence[object] = (),
    options: dict[str, object] | None = None,
) -> float:
    """Return a higher-is-better score for optimization."""
    value = metric_value(
        metric_name,
        target,
        prediction,
        class_names=class_names,
        options=options,
    )
    return value if infer_metric_direction(metric_name) == "maximize" else -value
