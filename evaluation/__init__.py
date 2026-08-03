"""Modality-neutral evaluation services.

Imports are intentionally lazy: the canonical task contracts use metric-name
normalization, while fidelity and runner services depend on those contracts.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "build_error_analysis": (
        ".error_analysis",
        "build_error_analysis",
    ),
    "create_split_plan": (".splitters", "create_split_plan"),
    "evaluate_prediction_bundle": (
        ".runner",
        "evaluate_prediction_bundle",
    ),
    "get_fidelity_profile": (".fidelity", "get_fidelity_profile"),
    "normalize_evaluation_mode": (
        ".policy",
        "normalize_evaluation_mode",
    ),
    "load_prediction_bundle": (
        ".prediction_io",
        "load_prediction_bundle",
    ),
    "load_prediction_table": (
        ".prediction_io",
        "load_prediction_table",
    ),
    "prepare_evaluation_bundle": (
        ".runner",
        "prepare_evaluation_bundle",
    ),
    "select_evaluation_policy": (
        ".policy",
        "select_evaluation_policy",
    ),
    "write_prediction_bundle": (
        ".prediction_io",
        "write_prediction_bundle",
    ),
    "write_prediction_table": (
        ".prediction_io",
        "write_prediction_table",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
