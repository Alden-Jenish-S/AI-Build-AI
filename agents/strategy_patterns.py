"""Concise reusable ML design knowledge without executable artifacts."""

from __future__ import annotations

from typing import Mapping


BASE_PATTERNS = (
    "Begin with schema, target, leakage-unit, missingness, cardinality, and distribution diagnostics; only add transformations justified by those diagnostics.",
    "Fit every learned transform on protocol training rows only and reuse exactly that fitted state for validation, full refit, and test.",
    "Use explicit reproducible seeds. Average seeds only for materially stochastic models when the fidelity budget supports it.",
    "Treat the simplest sound pipeline as a control and make each branch test a concrete representation, robustness, or optimization hypothesis.",
)

TABULAR_PATTERNS = (
    "For categoricals, compare rare-level grouping plus frequency/count encoding; use smoothed target encoding only inside the harness training split or fold.",
    "Consider missingness indicators, robust clipping/scaling, log transforms, and domain-plausible interactions only when the observed distributions support them.",
    "Keep identity columns out of features and preserve train/validation/test feature parity after every transformation.",
    "For wide tables, use leakage-free feature selection or regularization inside the selected evaluation protocol rather than on all labeled rows.",
)

STRUCTURED_PATTERNS = (
    "Separate deterministic decoding/normalization from learned augmentation; stochastic augmentation is training-only.",
    "Prefer pretrained representations when appropriate, but validate task-specific heads and calibration on the harness split.",
    "Control memory through streaming, bounded batches, and fidelity-aware resolution or sequence length.",
)


def strategy_patterns(task_spec: Mapping[str, object] | None) -> list[str]:
    """Return relevant high-level patterns, never executable source code."""
    spec = dict(task_spec or {})
    if not spec:
        return list(BASE_PATTERNS)
    modality = str(spec.get("modality") or "tabular").lower()
    problem_type = str(spec.get("problem_type") or "supervised").lower()
    patterns = list(BASE_PATTERNS)
    if modality == "tabular":
        patterns.extend(TABULAR_PATTERNS)
    else:
        patterns.extend(STRUCTURED_PATTERNS)
    if problem_type == "unsupervised_clustering":
        patterns.append(
            "Choose representations and cluster stability checks without inventing supervised labels or target encodings."
        )
    return patterns


def render_strategy_patterns(
    task_spec: Mapping[str, object] | None,
) -> str:
    return "\n".join(
        f"- {pattern}" for pattern in strategy_patterns(task_spec)
    )
