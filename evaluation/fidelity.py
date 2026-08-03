"""Task-neutral experiment fidelity profiles."""

from __future__ import annotations

from core.runtime_contracts import FidelityProfile


_COMMON = {
    "screen": {
        "sample_fraction": 0.25,
        "folds": 2,
        "max_trials": 8,
        "max_epochs": 8,
        "early_stopping_patience": 2,
        "max_estimator_iterations": 500,
    },
    "medium": {
        "sample_fraction": 0.60,
        "folds": 3,
        "max_trials": 20,
        "max_epochs": 20,
        "early_stopping_patience": 4,
        "max_estimator_iterations": 1500,
    },
    "full": {
        "sample_fraction": 1.0,
        "folds": 5,
        "max_trials": 40,
        "max_epochs": 50,
        "early_stopping_patience": 7,
        "max_estimator_iterations": 4000,
    },
}

def get_fidelity_profile(
    modality: str, name: str
) -> FidelityProfile:
    """Resolve task-neutral limits; ``modality`` is retained for API compatibility."""
    normalized_name = str(name).strip().lower()
    if normalized_name not in _COMMON:
        raise ValueError(f"unknown fidelity: {name!r}")
    values = dict(_COMMON[normalized_name])
    return FidelityProfile(name=normalized_name, **values)


def legacy_profiles() -> dict[str, dict[str, object]]:
    """Return the historical mapping consumed by legacy generated code."""
    return {
        name: {
            "data_fraction": values["sample_fraction"],
            "cv_folds": values["folds"],
            "max_tuning_trials": values["max_trials"],
            "max_epochs": values["max_epochs"],
            "early_stopping_patience": values[
                "early_stopping_patience"
            ],
            "max_estimator_iterations": values[
                "max_estimator_iterations"
            ],
        }
        for name, values in _COMMON.items()
    }
