"""Output-type strategy mapping and ensemble registry."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence
import numpy as np


class EnsembleStrategyRegistry:
    """Maps output types to preferred and fallback ensemble strategy functions."""

    def __init__(self) -> None:
        self._strategies: dict[str, dict[str, str]] = {
            "class_probabilities": {
                "preferred": "cross_validated_stacking",
                "fallback": "rank_average",
            },
            "scalar_predictions": {
                "preferred": "cross_validated_stacking",
                "fallback": "average",
            },
            "continuous": {
                "preferred": "regularized_stacking",
                "fallback": "average",
            },
            "multilabel_probabilities": {
                "preferred": "joint_stacking",
                "fallback": "average",
            },
            "segmentation_logits": {
                "preferred": "calibrated_pixel_blend",
                "fallback": "mean_logits",
            },
            "detection_boxes": {
                "preferred": "weighted_box_fusion",
                "fallback": "nms_merge",
            },
            "embeddings": {
                "preferred": "normalized_learned_fusion",
                "fallback": "normalized_average",
            },
            "ranked_items": {
                "preferred": "learned_reranker",
                "fallback": "rank_aggregation",
            },
            "generated_text": {
                "preferred": "validation_selection",
                "fallback": "strongest_candidate",
            },
        }

    def resolve(self, output_type: str) -> dict[str, str]:
        """Resolve strategy for an output type, with fallback for unknown types."""
        return self._strategies.get(
            output_type,
            {
                "preferred": "cross_validated_stacking",
                "fallback": "average",
            },
        )

    def register(
        self, output_type: str, preferred: str, fallback: str
    ) -> None:
        """Register or update an output-type strategy mapping."""
        self._strategies[output_type] = {
            "preferred": preferred,
            "fallback": fallback,
        }


_DEFAULT_REGISTRY = EnsembleStrategyRegistry()


def default_ensemble_registry() -> EnsembleStrategyRegistry:
    return _DEFAULT_REGISTRY
