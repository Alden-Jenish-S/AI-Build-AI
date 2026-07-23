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
    silhouette_score,
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


def _full_training_data(
    train_data: dict,
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray]:
    """Recover the complete training set, including a loader-held validation split."""
    unsupervised = (
        train_data.get("task_type") == "unsupervised_clustering"
        or (
            train_data.get("y") is None
            and train_data.get("y_full") is None
        )
    )
    if train_data.get("X_full") is not None and (
        train_data.get("y_full") is not None or unsupervised
    ):
        X = pd.DataFrame(train_data["X_full"]).reset_index(drop=True)
        y = (
            None
            if unsupervised
            else _array(train_data["y_full"])
        )
        row_ids = _array(train_data.get("row_ids_full", np.arange(len(X))))
    else:
        parts = [pd.DataFrame(train_data["X"])]
        targets = [] if unsupervised else [_array(train_data["y"])]
        row_parts = [
            _array(train_data.get("row_ids", np.arange(len(parts[0]))))
        ]
        if train_data.get("X_val") is not None and (
            train_data.get("y_val") is not None or unsupervised
        ):
            validation = pd.DataFrame(train_data["X_val"])
            parts.append(validation)
            if not unsupervised:
                targets.append(_array(train_data["y_val"]))
            fallback_ids = np.arange(len(parts[0]), len(parts[0]) + len(validation))
            row_parts.append(_array(train_data.get("val_row_ids", fallback_ids)))
        X = pd.concat(parts, ignore_index=True)
        y = None if unsupervised else np.concatenate(targets)
        row_ids = np.concatenate(row_parts)
    if (y is not None and len(X) != len(y)) or len(X) != len(row_ids):
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
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Return deterministic data/folds and persist the protocol used by a node.

    Generated algorithms must train and score on these rows and folds. This keeps
    fidelity decisions outside LLM control and restores loader-held validation rows
    for full-fidelity training.
    """
    if fidelity not in FIDELITY_PROFILES:
        raise ValueError(f"unknown fidelity: {fidelity!r}")
    profile = FIDELITY_PROFILES[fidelity]
    X_full, y_full, row_ids_full = _full_training_data(train_data)
    unsupervised = y_full is None
    fraction = float(profile["data_fraction"])
    selected = np.arange(len(X_full))
    declared_task_type = str(train_data.get("task_type", "")).strip().lower()
    if unsupervised or declared_task_type == "regression":
        classification = False
    elif declared_task_type == "classification":
        classification = True
    else:
        classification = _is_classification(y_full)
    if fraction < 1.0:
        if unsupervised:
            rng = np.random.default_rng(seed)
            selected = np.sort(
                rng.choice(
                    len(X_full),
                    size=max(2, int(len(X_full) * fraction)),
                    replace=False,
                )
            )
        elif classification and pd.Series(y_full).value_counts().min() >= 2:
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
    y = None if unsupervised else y_full[selected]
    row_ids = row_ids_full[selected]
    folds = int(profile["cv_folds"])
    can_stratify_folds = (
        classification
        and not unsupervised
        and pd.Series(y).value_counts().min() >= folds
    )
    splitter = (
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        if can_stratify_folds
        else KFold(n_splits=folds, shuffle=True, random_state=seed)
    )
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    split_iterator = (
        splitter.split(X)
        if unsupervised
        else splitter.split(X, y)
    )
    for fold, (_, validation_indices) in enumerate(split_iterator):
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
        "task_type": (
            "unsupervised_clustering" if unsupervised else "supervised"
        ),
        "fold_assignment_sha256": digest,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    (output / "evaluation_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return X, y, row_ids, fold_ids, metadata


def _numeric_clustering_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Deterministically encode a mixed table for internal cluster validation."""
    encoded = []
    for column in pd.DataFrame(frame).columns:
        series = pd.DataFrame(frame)[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            values = pd.to_numeric(series, errors="coerce").to_numpy(
                dtype=np.float64
            )
            values[~np.isfinite(values)] = np.nan
            finite = values[np.isfinite(values)]
            fill = float(np.median(finite)) if finite.size else 0.0
            values = np.nan_to_num(
                values, nan=fill, posinf=fill, neginf=fill
            )
        else:
            normalized = series.astype("string").fillna("<MISSING>")
            categories = sorted(normalized.unique().tolist())
            mapping = {value: index for index, value in enumerate(categories)}
            values = normalized.map(mapping).to_numpy(dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std())
        encoded.append((values - mean) / (std if std > 1e-12 else 1.0))
    if not encoded:
        raise ValueError("clustering evaluation requires at least one feature")
    matrix = np.column_stack(encoded).astype(np.float32, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("encoded clustering features contain non-finite values")
    return matrix


def evaluate_clustering_predictions(
    X: pd.DataFrame,
    labels: Any,
    row_ids: Any,
    fold_ids: Any,
    *,
    fidelity: str,
    output_dir: str | Path = ".",
    seed: int = 42,
    max_validation_rows: int = 5000,
) -> dict:
    """Persist and score deterministic clustering outputs without hidden labels.

    Adjusted Rand Index cannot be computed locally when the competition withholds
    its ground-truth clusters. This helper therefore produces a bounded silhouette
    proxy that the parent harness independently recomputes.
    """
    frame = pd.DataFrame(X).reset_index(drop=True)
    predicted = _array(labels)
    ids = _array(row_ids)
    folds = _array(fold_ids)
    if not (len(frame) == len(predicted) == len(ids) == len(folds)):
        raise ValueError(
            "clustering features, labels, row IDs, and fold IDs must align"
        )
    if len(frame) < 3:
        raise ValueError("clustering evaluation requires at least three rows")
    if pd.Series(ids).duplicated().any():
        raise ValueError("clustering row IDs must be unique")
    if pd.isna(predicted).any():
        raise ValueError("cluster labels may not contain missing values")
    codes, _ = pd.factorize(predicted, sort=True)
    if len(np.unique(codes)) < 2 or len(np.unique(codes)) >= len(codes):
        raise ValueError(
            "clustering must produce between 2 and n_rows - 1 clusters"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "row_id": ids,
            "prediction": codes.astype(np.int64),
            "fold_id": folds.astype(np.int64),
        }
    ).to_csv(output / "oof_predictions.csv", index=False)

    rng = np.random.default_rng(seed)
    sample_parts = []
    unique_folds = np.unique(folds)
    per_fold = max(2, int(max_validation_rows) // max(len(unique_folds), 1))
    for fold in unique_folds:
        indices = np.flatnonzero(folds == fold)
        if len(indices) > per_fold:
            indices = np.sort(
                rng.choice(indices, size=per_fold, replace=False)
            )
        sample_parts.append(indices)
    sample_indices = np.sort(np.concatenate(sample_parts))
    matrix = _numeric_clustering_matrix(frame.iloc[sample_indices])
    np.savez_compressed(
        output / "clustering_validation.npz",
        sample_indices=sample_indices.astype(np.int64),
        features=matrix,
    )

    sampled_labels = codes[sample_indices]
    sampled_folds = folds[sample_indices]
    fold_scores = []
    for fold in unique_folds:
        mask = sampled_folds == fold
        fold_labels = sampled_labels[mask]
        cluster_count = len(np.unique(fold_labels))
        if mask.sum() >= 3 and 1 < cluster_count < mask.sum():
            fold_scores.append(
                float(silhouette_score(matrix[mask], fold_labels))
            )
    if not fold_scores:
        raise ValueError(
            "no evaluation fold contains a valid multi-cluster assignment"
        )
    return {
        "score": float(np.mean(fold_scores)),
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "folds": len(fold_scores),
        "fold_scores": fold_scores,
        "metric": "silhouette_score",
        "direction": "maximize",
        "fidelity": fidelity,
    }


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
    if manifest.get("task_type") == "unsupervised_clustering":
        required = {"row_id", "prediction"}
        if not required.issubset(oof.columns):
            raise ValueError(
                f"clustering output is missing columns: "
                f"{sorted(required - set(oof.columns))}"
            )
        if len(oof) != int(manifest["row_count"]) or oof["row_id"].duplicated().any():
            raise ValueError(
                "clustering output does not cover every scheduled row exactly once"
            )
        assignments = pd.read_csv(root / "fold_assignments.csv").rename(
            columns={"fold_id": "expected_fold_id"}
        )
        merged = oof.merge(
            assignments, on="row_id", how="left", validate="one_to_one"
        )
        if merged["expected_fold_id"].isna().any():
            raise ValueError(
                "clustering rows do not match the harness fold assignment"
            )
        if "fold_id" in merged and not np.array_equal(
            merged["fold_id"].to_numpy(),
            merged["expected_fold_id"].to_numpy(),
        ):
            raise ValueError(
                "clustering fold IDs differ from the harness assignment"
            )
        predictions = merged["prediction"].to_numpy()
        if pd.isna(predictions).any():
            raise ValueError("cluster predictions contain missing values")
        labels, _ = pd.factorize(predictions, sort=True)
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
            raise ValueError("invalid number of predicted clusters")
        validation = np.load(
            root / "clustering_validation.npz", allow_pickle=False
        )
        sample_indices = validation["sample_indices"].astype(np.int64)
        features = validation["features"].astype(np.float64)
        if (
            sample_indices.ndim != 1
            or features.ndim != 2
            or len(sample_indices) != len(features)
            or (sample_indices < 0).any()
            or (sample_indices >= len(merged)).any()
            or not np.isfinite(features).all()
        ):
            raise ValueError("invalid clustering validation sample")
        sampled_labels = labels[sample_indices]
        sampled_folds = merged["expected_fold_id"].to_numpy()[sample_indices]
        fold_scores = []
        for fold in sorted(np.unique(sampled_folds)):
            mask = sampled_folds == fold
            fold_labels = sampled_labels[mask]
            cluster_count = len(np.unique(fold_labels))
            if mask.sum() >= 3 and 1 < cluster_count < mask.sum():
                fold_scores.append(
                    float(silhouette_score(features[mask], fold_labels))
                )
        if not fold_scores:
            raise ValueError(
                "no clustering fold has a valid silhouette score"
            )
        return {
            "cv_mean": float(np.mean(fold_scores)),
            "cv_std": float(np.std(fold_scores)),
            "folds": len(fold_scores),
            "fold_scores": fold_scores,
            "seed": int(manifest["seed"]),
            "fidelity": fidelity,
            "row_count": int(manifest["row_count"]),
            "source_row_count": int(manifest["source_row_count"]),
            "fold_assignment_sha256": manifest[
                "fold_assignment_sha256"
            ],
            "task_type": "unsupervised_clustering",
            "metric_proxy_for": "adjusted_rand_index",
        }

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
