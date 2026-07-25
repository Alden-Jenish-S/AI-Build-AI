"""Modality-neutral runtime contracts for AIBuildAI."""

from .contracts import (
    InputSpec,
    MetricSpec,
    OutputSpec,
    ResourceLimits,
    TargetSpec,
    TaskSpec,
)
from .modality_registry import ModalityRegistry
from .runtime_contracts import (
    DatasetBundle,
    EnsembleBundle,
    FidelityProfile,
    ModelBundle,
    PredictionBundle,
    ResultRecord,
    SampleRecord,
    SplitPlan,
)

__all__ = [
    "InputSpec",
    "DatasetBundle",
    "EnsembleBundle",
    "FidelityProfile",
    "MetricSpec",
    "ModelBundle",
    "ModalityRegistry",
    "OutputSpec",
    "PredictionBundle",
    "ResourceLimits",
    "ResultRecord",
    "SampleRecord",
    "SplitPlan",
    "TargetSpec",
    "TaskSpec",
]
