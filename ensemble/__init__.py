"""Ensemble strategies and output-type registry for AIBuildAI."""

from .registry import EnsembleStrategyRegistry, default_ensemble_registry
from .stacking import fit_cross_validated_stacker, optimize_constrained_blend
from .structured import merge_structured_outputs

__all__ = [
    "EnsembleStrategyRegistry",
    "default_ensemble_registry",
    "fit_cross_validated_stacker",
    "optimize_constrained_blend",
    "merge_structured_outputs",
]
