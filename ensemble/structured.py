"""Structured output merging strategies for boxes, masks, embeddings, and items."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np


def merge_embeddings(
    embedding_arrays: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Combine structured feature embeddings across model nodes."""
    if not embedding_arrays:
        raise ValueError("embedding_arrays must be non-empty")

    if weights is None:
        norm_weights = np.full(len(embedding_arrays), 1.0 / len(embedding_arrays))
    else:
        norm_weights = np.asarray(weights, dtype=float)
        norm_weights /= norm_weights.sum()

    stacked = np.stack(embedding_arrays)  # (K, N, D)
    merged = np.tensordot(norm_weights, stacked, axes=(0, 0))

    if normalize:
        norms = np.linalg.norm(merged, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        merged = merged / norms
    return merged


def merge_segmentation_logits(
    logits_list: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Weighted blend of spatial logit maps or pixel probabilities."""
    if not logits_list:
        raise ValueError("logits_list must be non-empty")

    if weights is None:
        norm_weights = np.full(len(logits_list), 1.0 / len(logits_list))
    else:
        norm_weights = np.asarray(weights, dtype=float)
        norm_weights /= norm_weights.sum()

    stacked = np.stack(logits_list)
    return np.tensordot(norm_weights, stacked, axes=(0, 0))


def merge_structured_outputs(
    output_type: str,
    payloads: Sequence[Any],
    weights: Sequence[float] | None = None,
) -> Any:
    """Route structured output payload merging based on output_type."""
    if output_type == "embeddings":
        return merge_embeddings(payloads, weights=weights)
    elif output_type in {"segmentation_logits", "masks"}:
        return merge_segmentation_logits(payloads, weights=weights)
    else:
        # Default weighted average fallback
        if weights is None:
            norm_weights = np.full(len(payloads), 1.0 / len(payloads))
        else:
            norm_weights = np.asarray(weights, dtype=float)
            norm_weights /= norm_weights.sum()
        stacked = np.stack(payloads)
        return np.tensordot(norm_weights, stacked, axes=(0, 0))
