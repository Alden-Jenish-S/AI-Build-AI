"""Registered fidelity profiles for each modality."""

from __future__ import annotations

from core.contracts import normalize_modality
from core.runtime_contracts import FidelityProfile


_COMMON = {
    "screen": {
        "sample_fraction": 0.25,
        "folds": 5,
        "max_trials": 20,
        "max_epochs": 20,
        "early_stopping_patience": 5,
        "max_estimator_iterations": 2000,
    },
    "medium": {
        "sample_fraction": 0.60,
        "folds": 8,
        "max_trials": 40,
        "max_epochs": 40,
        "early_stopping_patience": 7,
        "max_estimator_iterations": 2500,
    },
    "full": {
        "sample_fraction": 1.0,
        "folds": 10,
        "max_trials": 60,
        "max_epochs": 60,
        "early_stopping_patience": 10,
        "max_estimator_iterations": 3000,
    },
}

_OVERRIDES = {
    "image": {
        "screen": {"spatial_size": (128, 128)},
        "medium": {"spatial_size": (224, 224)},
        "full": {"spatial_size": (384, 384)},
    },
    "audio": {
        "screen": {
            "audio_sample_rate": 8000,
            "max_audio_seconds": 10.0,
        },
        "medium": {
            "audio_sample_rate": 16000,
            "max_audio_seconds": 30.0,
        },
        "full": {
            "audio_sample_rate": 32000,
            "max_audio_seconds": None,
        },
    },
    "video": {
        "screen": {
            "spatial_size": (112, 112),
            "video_frames": 8,
            "video_fps": 4.0,
            "clips_per_video": 1,
        },
        "medium": {
            "spatial_size": (160, 160),
            "video_frames": 16,
            "video_fps": 8.0,
            "clips_per_video": 2,
        },
        "full": {
            "spatial_size": (224, 224),
            "video_frames": 32,
            "video_fps": None,
            "clips_per_video": 4,
        },
    },
}


def get_fidelity_profile(
    modality: str, name: str
) -> FidelityProfile:
    """Resolve a fidelity without exposing modality conditionals to scheduling."""
    normalized_modality = normalize_modality(modality)
    normalized_name = str(name).strip().lower()
    if normalized_name not in _COMMON:
        raise ValueError(f"unknown fidelity: {name!r}")
    values = dict(_COMMON[normalized_name])
    # Multimodal evaluation uses the common sample/fold/optimization limits.
    # Each component adapter receives its own concrete fidelity when decoding.
    values.update(
        _OVERRIDES.get(normalized_modality, {}).get(normalized_name, {})
    )
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
