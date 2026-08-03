"""Concise evidence-driven ML design knowledge without task categories."""

from __future__ import annotations

from typing import Mapping


EVIDENCE_PATTERNS = (
    "Begin with the verified sources, target structure, leakage unit, value shapes, missingness, cardinality, and distribution diagnostics; add transformations only when those observations justify them.",
    "Fit every learned transform on protocol training samples only and reuse exactly that fitted state for validation, full refit, and inference.",
    "Derive decoding, batching, representation, loss, and prediction shape from actual runtime values and the output contract, never from a data-category label.",
    "Keep identity/join values out of learned features unless the task evidence establishes predictive semantics, while preserving sample alignment through every transformation.",
    "Separate deterministic preparation from stochastic training operations; validation and inference must be reproducible and structurally identical.",
    "Stream file-backed values and choose bounded batches or truncation from measured sizes and the active fidelity limits rather than loading an uninspected corpus eagerly.",
    "Treat the simplest sound pipeline as a control and make each branch test a concrete representation, robustness, or optimization hypothesis grounded in the inspected task files.",
)


def strategy_patterns(task_spec: Mapping[str, object] | None) -> list[str]:
    """Return universal high-level patterns; task evidence selects the details."""
    return list(EVIDENCE_PATTERNS)


def render_strategy_patterns(
    task_spec: Mapping[str, object] | None,
) -> str:
    return "\n".join(
        f"- {pattern}" for pattern in strategy_patterns(task_spec)
    )
