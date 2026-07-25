"""Backward-compatible access to the tabular task analyzer.

New orchestration should use :mod:`agents.task_analyzer`. Existing generated
code and tests can continue importing these functions while the runtime moves
to modality-neutral contracts.
"""

from modalities.tabular import (
    TabularAdapter,
    build_task_profile,
    discover_dataset_layout,
    run_dataset_analysis,
)

__all__ = [
    "TabularAdapter",
    "build_task_profile",
    "discover_dataset_layout",
    "run_dataset_analysis",
]
