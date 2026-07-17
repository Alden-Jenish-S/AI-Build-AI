"""Harness-owned data, fold, and validation contracts for generated experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit


FIDELITY_PROFILES = {
    "screen": {
        "data_fraction": 0.25,
        "cv_folds": 2,
        "max_tuning_trials": 8,
        "max_epochs": 8,
        "early_stopping_patience": 2,
        "max_estimator_iterations": 500,
    },
    "medium": {
        "data_fraction": 0.60,
        "cv_folds": 3,
        "max_tuning_trials": 20,
        "max_epochs": 20,
        "early_stopping_patience": 4,
        "max_estimator_iterations": 1500,
    },
    "full": {
        "data_fraction": 1.0,
        "cv_folds": 5,
        "max_tuning_trials": 40,
        "max_epochs": 50,
        "early_stopping_patience": 7,
        "max_estimator_iterations": 4000,
    },
}


def _array(value: Any) -> np.ndarray:
    if isinstance(value, pd.Series):
        return value.to_numpy()
    return np.asarray(value)


def _full_training_data(train_data: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Recover the complete training set, including a loader-held validation split."""
    if train_data.get("X_full") is not None and train_data.get("y_full") is not None:
        X = pd.DataFrame(train_data["X_full"]).reset_index(drop=True)
        y = _array(train_data["y_full"])
        row_ids = _array(train_data.get("row_ids_full", np.arange(len(X))))
    else:
        parts = [pd.DataFrame(train_data["X"])]
        targets = [_array(train_data["y"])]
        row_parts = [
            _array(train_data.get("row_ids", np.arange(len(parts[0]))))
        ]
        if train_data.get("X_val") is not None and train_data.get("y_val") is not None:
            validation = pd.DataFrame(train_data["X_val"])
            parts.append(validation)
            targets.append(_array(train_data["y_val"]))
            fallback_ids = np.arange(len(parts[0]), len(parts[0]) + len(validation))
            row_parts.append(_array(train_data.get("val_row_ids", fallback_ids)))
        X = pd.concat(parts, ignore_index=True)
        y = np.concatenate(targets)
        row_ids = np.concatenate(row_parts)
    if len(X) != len(y) or len(X) != len(row_ids):
        raise ValueError("full training features, targets, and row IDs must align")
    if pd.Series(row_ids).duplicated().any():
        # Legacy generated loaders often reset both split indices. Stable synthetic
        # IDs are safer than silently joining different rows under the same key.
        row_ids = np.arange(len(X))
    return X, y, row_ids


def _is_classification(y: np.ndarray) -> bool:
    unique = np.unique(y)
    return len(unique) <= max(20, int(np.sqrt(max(len(y), 1))))


def prepare_evaluation_data(
    train_data: dict,
    fidelity: str,
    *,
    seed: int = 42,
    output_dir: str | Path = ".",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return deterministic data/folds and persist the protocol used by a node.

    Generated algorithms must train and score on these rows and folds. This keeps
    fidelity decisions outside LLM control and restores loader-held validation rows
    for full-fidelity training.
    """
    if fidelity not in FIDELITY_PROFILES:
        raise ValueError(f"unknown fidelity: {fidelity!r}")
    profile = FIDELITY_PROFILES[fidelity]
    X_full, y_full, row_ids_full = _full_training_data(train_data)
    fraction = float(profile["data_fraction"])
    selected = np.arange(len(X_full))
    classification = _is_classification(y_full)
    if fraction < 1.0:
        if classification:
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=fraction, random_state=seed
            )
            selected, _ = next(splitter.split(X_full, y_full))
        else:
            rng = np.random.default_rng(seed)
            selected = np.sort(
                rng.choice(len(X_full), size=max(2, int(len(X_full) * fraction)), replace=False)
            )
    X = X_full.iloc[selected].reset_index(drop=True)
    y = y_full[selected]
    row_ids = row_ids_full[selected]
    folds = int(profile["cv_folds"])
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        if classification
        else KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    for fold, (_, validation_indices) in enumerate(splitter.split(X, y)):
        fold_ids[validation_indices] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("failed to assign every evaluation row to a fold")

    assignments = pd.DataFrame({"row_id": row_ids, "fold_id": fold_ids})
    digest = hashlib.sha256(
        assignments.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    metadata = {
        "protocol_version": 1,
        "fidelity": fidelity,
        "seed": seed,
        "source_row_count": int(len(X_full)),
        "row_count": int(len(X)),
        "data_fraction": fraction,
        "cv_folds": folds,
        "classification": classification,
        "fold_assignment_sha256": digest,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return X, y, row_ids, fold_ids, metadata


def _metric_value(metric_name: str, target: np.ndarray, prediction: np.ndarray) -> float:
    metric = metric_name.lower()
    if "auc" in metric:
        return float(roc_auc_score(target, prediction))
    if "mae" in metric:
        return float(mean_absolute_error(target, prediction))
    if "rmse" in metric:
        return float(mean_squared_error(target, prediction) ** 0.5)
    if "accuracy" in metric:
        return float(accuracy_score(target, prediction >= 0.5))
    if "log_loss" in metric or "logloss" in metric:
        return float(log_loss(target, prediction))
    if metric in {"f1", "f1_score"}:
        return float(f1_score(target, prediction >= 0.5))
    if metric in {"r2", "r2_score"}:
        return float(r2_score(target, prediction))
    raise ValueError(f"unsupported harness metric for OOF validation: {metric_name!r}")


def validate_evaluation_outputs(
    node_dir: str | Path,
    fidelity: str,
    metric_name: str,
) -> dict:
    """Validate the generated protocol and recompute fold statistics from OOF."""
    root = Path(node_dir)
    manifest = json.loads((root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    expected = FIDELITY_PROFILES[fidelity]
    if manifest.get("fidelity") != fidelity:
        raise ValueError("evaluation manifest fidelity does not match the scheduled fidelity")
    if int(manifest.get("cv_folds", 0)) != int(expected["cv_folds"]):
        raise ValueError("evaluation manifest fold count violates the fidelity profile")
    if abs(float(manifest.get("data_fraction", -1)) - float(expected["data_fraction"])) > 1e-12:
        raise ValueError("evaluation manifest data fraction violates the fidelity profile")

    oof = pd.read_csv(root / "oof_predictions.csv")
    required = {"row_id", "target", "prediction"}
    if not required.issubset(oof.columns):
        raise ValueError(f"OOF output is missing columns: {sorted(required - set(oof.columns))}")
    if len(oof) != int(manifest["row_count"]) or oof["row_id"].duplicated().any():
        raise ValueError("OOF output does not cover each scheduled evaluation row exactly once")
    assignments = pd.read_csv(root / "fold_assignments.csv")
    if assignments["row_id"].duplicated().any():
        raise ValueError("fold assignment row IDs are duplicated")
    assignments = assignments.rename(columns={"fold_id": "expected_fold_id"})
    merged = oof.merge(assignments, on="row_id", how="left", validate="one_to_one")
    if merged["expected_fold_id"].isna().any():
        raise ValueError("OOF rows do not match the harness-owned fold assignment")
    if "fold_id" in merged and not np.array_equal(
        merged["fold_id"].to_numpy(), merged["expected_fold_id"].to_numpy()
    ):
        raise ValueError("OOF fold IDs differ from the harness-owned assignment")
    target = merged["target"].to_numpy()
    prediction = merged["prediction"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("OOF predictions contain non-finite values")
    fold_scores = [
        _metric_value(
            metric_name,
            group["target"].to_numpy(),
            group["prediction"].to_numpy(dtype=float),
        )
        for _, group in merged.groupby("expected_fold_id", sort=True)
    ]
    return {
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "folds": len(fold_scores),
        "fold_scores": fold_scores,
        "seed": int(manifest["seed"]),
        "fidelity": fidelity,
        "row_count": int(manifest["row_count"]),
        "source_row_count": int(manifest["source_row_count"]),
        "fold_assignment_sha256": manifest["fold_assignment_sha256"],
    }
