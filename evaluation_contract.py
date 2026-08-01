"""Harness-owned data, fold, and validation contracts for generated experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from evaluation.fidelity import legacy_profiles
from evaluation.metrics import (
    infer_metric_direction,
    metric_value,
    resolve_metric_name,
)
from evaluation.policy import normalize_evaluation_mode
from evaluation.runner import (
    evaluate_prediction_bundle,
    prepare_evaluation_bundle,
)
from evaluation.prediction_io import legacy_prediction_payload

FIDELITY_PROFILES = legacy_profiles()
CONTRACT_STATE_DIR = ".evaluation_contract"


def prepare_modality_evaluation(*args, **kwargs):
    """Compatibility facade for the modality-neutral evaluation runner."""
    return prepare_evaluation_bundle(*args, **kwargs)


def validate_prediction_bundle(*args, **kwargs):
    """Recompute metrics from a typed non-tabular prediction bundle."""
    return evaluate_prediction_bundle(*args, **kwargs)


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


def prepare_final_training_data(
    train_data: dict,
    test_data: dict,
    *,
    output_dir: str | Path = ".",
) -> tuple[pd.DataFrame, np.ndarray | None, pd.DataFrame, np.ndarray]:
    """Return full training and test rows and persist a refit audit manifest."""
    X_full, y_full, row_ids_full = _full_training_data(train_data)
    if test_data.get("X_test") is None:
        raise ValueError("test_data must provide X_test for final prediction")
    X_test = pd.DataFrame(test_data["X_test"]).reset_index(drop=True)
    test_ids = _array(
        test_data.get("test_ids", np.arange(len(X_test)))
    )
    if len(X_test) != len(test_ids):
        raise ValueError("test features and IDs must align")
    if pd.Series(test_ids).duplicated().any():
        raise ValueError("test IDs must be unique")

    def digest(values: np.ndarray) -> str:
        payload = pd.Series(values).astype(str).to_csv(
            index=False, header=False
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_version": 1,
        "used_full_training_data": True,
        "train_row_count": int(len(X_full)),
        "test_row_count": int(len(X_test)),
        "train_id_sha256": digest(row_ids_full),
        "test_id_sha256": digest(test_ids),
    }
    (output / "final_training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    # Keep a harness-owned proof separate from the public convenience file.
    # Generated implementations have repeatedly overwritten the latter with
    # ad-hoc, multi-megabyte payloads after correctly calling this helper.
    contract_state = output / CONTRACT_STATE_DIR
    contract_state.mkdir(parents=True, exist_ok=True)
    (contract_state / "final_training_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return X_full, y_full, X_test, test_ids


def _is_classification(y: np.ndarray) -> bool:
    """Infer only when a legacy loader did not declare its problem type."""
    values = pd.Series(np.asarray(y).reshape(-1)).dropna()
    if values.empty:
        return False
    if (
        pd.api.types.is_bool_dtype(values.dtype)
        or pd.api.types.is_object_dtype(values.dtype)
        or pd.api.types.is_string_dtype(values.dtype)
        or isinstance(values.dtype, pd.CategoricalDtype)
    ):
        return True
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        return True
    unique_count = int(numeric.nunique())
    integer_like = bool(
        np.allclose(numeric.to_numpy(), np.round(numeric.to_numpy()))
    )
    return integer_like and (
        unique_count <= 100
        or unique_count / max(len(numeric), 1) <= 0.20
    )


def prepare_evaluation_data(
    train_data: dict,
    fidelity: str,
    *,
    seed: int = 42,
    output_dir: str | Path = ".",
    evaluation_mode: str = "cross_validation",
) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Return deterministic data/folds and persist the protocol used by a node.

    Generated algorithms must train and score on these rows and folds. This keeps
    fidelity decisions outside LLM control and restores loader-held validation rows
    for full-fidelity training.
    """
    if fidelity not in FIDELITY_PROFILES:
        raise ValueError(f"unknown fidelity: {fidelity!r}")
    evaluation_mode = normalize_evaluation_mode(
        evaluation_mode, allow_auto=False
    )
    profile = FIDELITY_PROFILES[fidelity]
    X_full, y_full, row_ids_full = _full_training_data(train_data)
    group_ids_full = train_data.get("group_ids_full")
    if group_ids_full is not None:
        group_ids_full = _array(group_ids_full)
        if len(group_ids_full) != len(X_full):
            raise ValueError(
                "full training group IDs must align with features"
            )
    grouped_leakage = (
        group_ids_full is not None
        and pd.Series(group_ids_full).duplicated().any()
    )
    unsupervised = y_full is None
    fraction = float(profile["data_fraction"])
    folds = int(profile["cv_folds"])
    selected = np.arange(len(X_full))
    declared_task_type = str(train_data.get("task_type", "")).strip().lower()
    if unsupervised or declared_task_type == "regression":
        classification = False
    elif declared_task_type in {
        "classification",
        "multilabel_classification",
    }:
        classification = True
    else:
        classification = _is_classification(y_full)
    stratifiable_classification = (
        classification
        and not unsupervised
        and np.asarray(y_full).ndim == 1
    )
    if fraction < 1.0:
        if grouped_leakage:
            rng = np.random.default_rng(seed)
            groups = np.asarray(
                sorted(set(str(item) for item in group_ids_full))
            )
            rng.shuffle(groups)
            selected_groups = set()
            selected_count = 0
            target_count = max(2, int(len(X_full) * fraction))
            for group in groups:
                selected_groups.add(str(group))
                selected_count += int(
                    np.sum(
                        np.asarray(
                            [str(item) for item in group_ids_full]
                        )
                        == group
                    )
                )
                if selected_count >= target_count:
                    break
            selected = np.flatnonzero(
                np.asarray(
                    [
                        str(item) in selected_groups
                        for item in group_ids_full
                    ],
                    dtype=bool,
                )
            )
        elif unsupervised:
            rng = np.random.default_rng(seed)
            selected = np.sort(
                rng.choice(
                    len(X_full),
                    size=max(2, int(len(X_full) * fraction)),
                    replace=False,
                )
            )
        elif stratifiable_classification:
            # Preserve enough observations per class for every scheduled fold.
            # A fixed global fraction can otherwise leave high-cardinality tasks
            # with one training example (or none) for most classes.
            rng = np.random.default_rng(seed)
            labels = pd.Series(y_full).reset_index(drop=True)
            minimum_per_class = max(folds, 4)
            selected_parts: list[np.ndarray] = []
            for _, class_indices in labels.groupby(
                labels, sort=True, dropna=False
            ).groups.items():
                indices = np.asarray(list(class_indices), dtype=int)
                requested = int(np.ceil(len(indices) * fraction))
                take = min(
                    len(indices),
                    max(1, requested, minimum_per_class),
                )
                chosen = (
                    indices
                    if take == len(indices)
                    else rng.choice(indices, size=take, replace=False)
                )
                selected_parts.append(np.asarray(chosen, dtype=int))
            selected = np.sort(np.concatenate(selected_parts))
        else:
            rng = np.random.default_rng(seed)
            selected = np.sort(
                rng.choice(len(X_full), size=max(2, int(len(X_full) * fraction)), replace=False)
            )
    X = X_full.iloc[selected].reset_index(drop=True)
    y = None if unsupervised else y_full[selected]
    row_ids = row_ids_full[selected]
    group_ids = (
        group_ids_full[selected]
        if group_ids_full is not None
        else None
    )
    split_group_ids = group_ids if grouped_leakage else None
    can_stratify_folds = (
        stratifiable_classification
        and not unsupervised
        and pd.Series(y).value_counts().min() >= folds
    )
    if split_group_ids is not None:
        if len(set(str(item) for item in split_group_ids)) < folds:
            raise ValueError(
                "group/entity count is smaller than the fidelity fold count"
            )
        splitter = (
            StratifiedGroupKFold(
                n_splits=folds, shuffle=True, random_state=seed
            )
            if can_stratify_folds
            else GroupKFold(n_splits=folds)
        )
    else:
        splitter = (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            if can_stratify_folds
            else KFold(n_splits=folds, shuffle=True, random_state=seed)
        )
    fold_ids = np.full(len(X), -1, dtype=np.int16)
    if split_group_ids is not None:
        split_iterator = splitter.split(X, y, split_group_ids)
    else:
        split_iterator = (
            splitter.split(X)
            if unsupervised
            else splitter.split(X, y)
        )
    for fold, (_, validation_indices) in enumerate(split_iterator):
        fold_ids[validation_indices] = fold
    if (fold_ids < 0).any():
        raise RuntimeError("failed to assign every evaluation row to a fold")
    if split_group_ids is not None:
        for group in set(str(item) for item in split_group_ids):
            group_folds = {
                int(fold_ids[index])
                for index, value in enumerate(split_group_ids)
                if str(value) == group
            }
            if len(group_folds) != 1:
                raise RuntimeError(
                    f"group/entity {group!r} crossed evaluation folds"
                )

    assignments = pd.DataFrame({"row_id": row_ids, "fold_id": fold_ids})
    digest = hashlib.sha256(
        assignments.to_csv(index=False).encode("utf-8")
    ).hexdigest()
    metadata = {
        "protocol_version": 1,
        "evaluation_mode": evaluation_mode,
        "fidelity": fidelity,
        "seed": seed,
        "source_row_count": int(len(X_full)),
        "row_count": int(len(X)),
        "data_fraction": fraction,
        "effective_data_fraction": float(len(X) / max(len(X_full), 1)),
        "cv_folds": folds,
        "classification": classification,
        "output_type": train_data.get(
            "output_type",
            "class_probabilities" if classification else "continuous",
        ),
        "leakage_unit": (
            "group_or_entity" if group_ids is not None else "row_id"
        ),
        "task_type": (
            "unsupervised_clustering"
            if unsupervised
            else declared_task_type
            if declared_task_type and declared_task_type != "supervised"
            else "classification"
            if classification
            else "regression"
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


def _json_compatible(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    return value


def prepare_holdout_evaluation_data(
    train_data: dict,
    fidelity: str,
    *,
    seed: int = 42,
    output_dir: str | Path = ".",
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    dict,
]:
    """Return one harness-owned train/validation split without requiring OOF.

    The fidelity sampler and leakage-aware split construction remain harness
    controlled. Only fold zero is exposed as validation; all remaining rows
    form one training set, so expensive or non-fold-independent models train
    exactly once.
    """
    X, y, row_ids, fold_ids, metadata = prepare_evaluation_data(
        train_data,
        fidelity,
        seed=seed,
        output_dir=output_dir,
        evaluation_mode="holdout",
    )
    if y is None:
        raise ValueError(
            "holdout evaluation requires supervised targets; use "
            "task_native evaluation for unsupervised tasks"
        )
    validation_mask = np.asarray(fold_ids) == 0
    training_mask = ~validation_mask
    if not training_mask.any() or not validation_mask.any():
        raise ValueError("holdout split must contain training and validation rows")

    X_train = X.iloc[np.flatnonzero(training_mask)].reset_index(drop=True)
    X_validation = X.iloc[
        np.flatnonzero(validation_mask)
    ].reset_index(drop=True)
    y_array = _array(y)
    y_train = y_array[training_mask]
    y_validation = y_array[validation_mask]
    validation_row_ids = _array(row_ids)[validation_mask]

    output = Path(output_dir)
    assignments = pd.DataFrame({"row_id": validation_row_ids})
    assignments.to_csv(
        output / "validation_assignments.csv", index=False
    )
    proof_dir = output / CONTRACT_STATE_DIR
    proof_dir.mkdir(parents=True, exist_ok=True)
    target_proof = {
        "row_ids": [str(item) for item in validation_row_ids],
        "targets": [
            _json_compatible(item) for item in y_validation
        ],
    }
    (proof_dir / "validation_targets.json").write_text(
        json.dumps(target_proof, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metadata.update(
        {
            "evaluation_mode": "holdout",
            "training_row_count": int(training_mask.sum()),
            "validation_row_count": int(validation_mask.sum()),
            "validation_fold": 0,
        }
    )
    (output / "evaluation_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        validation_row_ids,
        metadata,
    )


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
    ).to_csv(output / "validation_predictions.csv", index=False)

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
        "validation_score": float(np.mean(fold_scores)),
        "score_std": float(np.std(fold_scores)),
        "evaluation_mode": "task_native",
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "folds": len(fold_scores),
        "fold_scores": fold_scores,
        "metric": "silhouette_score",
        "direction": "maximize",
        "fidelity": fidelity,
    }


def _metric_value(metric_name: str, target: np.ndarray, prediction: np.ndarray) -> float:
    return metric_value(metric_name, target, prediction)


def validate_evaluation_outputs(
    node_dir: str | Path,
    fidelity: str,
    metric_name: str,
    *,
    expected_class_names: tuple[str, ...] = (),
    expected_evaluation_mode: str | None = None,
) -> dict:
    """Validate and independently score the selected evaluation protocol."""
    root = Path(node_dir)
    manifest = json.loads((root / "evaluation_manifest.json").read_text(encoding="utf-8"))
    evaluation_mode = normalize_evaluation_mode(
        manifest.get("evaluation_mode", "cross_validation"),
        allow_auto=False,
    )
    if expected_evaluation_mode is not None:
        expected_mode = normalize_evaluation_mode(
            expected_evaluation_mode, allow_auto=False
        )
        if evaluation_mode != expected_mode:
            raise ValueError(
                f"evaluation manifest mode {evaluation_mode!r} does not "
                f"match scheduled mode {expected_mode!r}"
            )
    expected = FIDELITY_PROFILES[fidelity]
    protocol_version = int(manifest.get("protocol_version", 1))
    if protocol_version >= 2:
        fidelity_payload = manifest.get("fidelity")
        if not isinstance(fidelity_payload, dict):
            raise ValueError("v2 evaluation manifest fidelity must be an object")
        if fidelity_payload.get("name") != fidelity:
            raise ValueError(
                "evaluation manifest fidelity does not match the scheduled fidelity"
            )
        if int(fidelity_payload.get("folds", 0)) != int(
            expected["cv_folds"]
        ):
            raise ValueError(
                "evaluation manifest fold count violates the fidelity profile"
            )
        if abs(
            float(fidelity_payload.get("sample_fraction", -1))
            - float(expected["data_fraction"])
        ) > 1e-12:
            raise ValueError(
                "evaluation manifest data fraction violates the fidelity profile"
            )
    else:
        if manifest.get("fidelity") != fidelity:
            raise ValueError(
                "evaluation manifest fidelity does not match the scheduled fidelity"
            )
        if int(manifest.get("cv_folds", 0)) != int(expected["cv_folds"]):
            raise ValueError(
                "evaluation manifest fold count violates the fidelity profile"
            )
        if abs(
            float(manifest.get("data_fraction", -1))
            - float(expected["data_fraction"])
        ) > 1e-12:
            raise ValueError(
                "evaluation manifest data fraction violates the fidelity profile"
            )

    if evaluation_mode == "holdout":
        validation_path = root / "validation_predictions.csv"
        if not validation_path.is_file():
            raise ValueError(
                "holdout evaluation requires validation_predictions.csv; "
                "OOF output is neither required nor accepted"
            )
        frame = pd.read_csv(validation_path, dtype={"row_id": str})
        if "row_id" not in frame.columns:
            aliases = [
                alias for alias in ("sample_id", "id")
                if alias in frame.columns
            ]
            if len(aliases) == 1:
                frame = frame.rename(columns={aliases[0]: "row_id"})
        required = {"row_id"}
        if not required.issubset(frame.columns):
            raise ValueError(
                "holdout predictions are missing the row_id column"
            )
        if frame["row_id"].duplicated().any():
            raise ValueError("holdout prediction row IDs are duplicated")
        assignments = pd.read_csv(
            root / "validation_assignments.csv",
            dtype={"row_id": str},
        )
        if assignments["row_id"].duplicated().any():
            raise ValueError("validation assignment row IDs are duplicated")
        if len(frame) != len(assignments):
            raise ValueError(
                "holdout predictions do not cover every validation row"
            )
        aligned = frame.set_index("row_id").reindex(assignments["row_id"])
        if aligned.index.hasnans or aligned.isna().all(axis=1).any():
            raise ValueError(
                "holdout prediction IDs differ from the harness validation split"
            )
        proof = json.loads(
            (
                root
                / CONTRACT_STATE_DIR
                / "validation_targets.json"
            ).read_text(encoding="utf-8")
        )
        expected_ids = [str(item) for item in proof.get("row_ids", [])]
        if expected_ids != assignments["row_id"].astype(str).tolist():
            raise ValueError(
                "validation targets differ from the harness assignment"
            )
        authoritative_target = np.asarray(proof.get("targets", []))
        if len(authoritative_target) != len(aligned):
            raise ValueError("validation target proof has the wrong length")
        aligned = aligned.reset_index()
        aligned["target"] = authoritative_target
        resolved_metric = resolve_metric_name(
            metric_name,
            problem_type=manifest.get("task_type"),
            output_type=manifest.get("output_type"),
        )
        prediction, class_names = legacy_prediction_payload(aligned)
        if expected_class_names and tuple(class_names) != tuple(
            expected_class_names
        ):
            raise ValueError(
                "holdout class-probability columns differ from the task "
                "output contract"
            )
        if np.issubdtype(prediction.dtype, np.number) and not np.isfinite(
            prediction.astype(float, copy=False)
        ).all():
            raise ValueError("holdout predictions contain non-finite values")
        score = float(
            metric_value(
                resolved_metric,
                authoritative_target,
                prediction,
                class_names=class_names,
            )
        )
        bootstrap_scores = []
        if len(aligned) >= 2:
            rng = np.random.default_rng(int(manifest.get("seed", 42)))
            for _ in range(64):
                sample = rng.integers(0, len(aligned), len(aligned))
                try:
                    value = float(
                        metric_value(
                            resolved_metric,
                            authoritative_target[sample],
                            prediction[sample],
                            class_names=class_names,
                        )
                    )
                except (TypeError, ValueError, IndexError):
                    continue
                if np.isfinite(value):
                    bootstrap_scores.append(value)
        score_std = (
            float(np.std(bootstrap_scores))
            if bootstrap_scores
            else 0.0
        )
        return {
            "score": score,
            "validation_score": score,
            "score_std": score_std,
            "evaluation_mode": "holdout",
            "folds": 1,
            "fold_scores": [score],
            "seed": int(manifest["seed"]),
            "fidelity": fidelity,
            "row_count": len(aligned),
            "validation_row_count": len(aligned),
            "source_row_count": int(manifest["source_row_count"]),
            "fold_assignment_sha256": manifest[
                "fold_assignment_sha256"
            ],
            "metric": resolved_metric,
            "direction": infer_metric_direction(resolved_metric),
        }

    typed_manifest = root / "predictions" / "manifest.json"
    legacy_oof = root / "oof_predictions.csv"
    if typed_manifest.is_file() and (
        protocol_version >= 2 or not legacy_oof.is_file()
    ):
        resolved_metric = resolve_metric_name(
            metric_name,
            problem_type=manifest.get("problem_type"),
            output_type=manifest.get("output_type"),
        )
        validated = evaluate_prediction_bundle(
            typed_manifest, resolved_metric
        )
        expected_count = int(
            manifest.get("sample_count", manifest.get("row_count", 0))
        )
        from evaluation.prediction_io import load_prediction_bundle

        bundle, predictions, _, fold_ids = load_prediction_bundle(
            typed_manifest
        )
        if expected_class_names and tuple(bundle.class_names) != tuple(
            expected_class_names
        ):
            raise ValueError(
                "typed OOF class columns differ from the task output contract"
            )
        if expected_count and len(predictions) != expected_count:
            raise ValueError(
                "typed predictions do not cover every scheduled sample"
            )
        if (
            manifest.get("split_fingerprint")
            and bundle.split_fingerprint != manifest["split_fingerprint"]
        ):
            raise ValueError(
                "typed predictions use a different harness split"
            )
        assignments = pd.read_csv(root / "fold_assignments.csv")
        id_column = (
            "sample_id"
            if "sample_id" in assignments.columns
            else "row_id"
        )
        expected_folds = assignments.set_index(id_column).reindex(
            list(bundle.sample_ids)
        )["fold_id"]
        if expected_folds.isna().any() or not np.array_equal(
            expected_folds.to_numpy(dtype=int),
            np.asarray(fold_ids, dtype=int),
        ):
            raise ValueError(
                "typed prediction folds differ from the harness assignment"
            )
        return {
            **validated,
            "score": float(validated["cv_mean"]),
            "score_std": float(validated["cv_std"]),
            "evaluation_mode": evaluation_mode,
            "metric": resolved_metric,
            "direction": infer_metric_direction(resolved_metric),
            "fidelity": fidelity,
            "row_count": len(predictions),
            "source_row_count": int(
                manifest.get(
                    "source_sample_count",
                    manifest.get("source_row_count", len(predictions)),
                )
            ),
            "task_type": manifest.get("problem_type"),
        }

    evaluation_output = (
        root / "validation_predictions.csv"
        if evaluation_mode == "task_native"
        else legacy_oof
    )
    oof = pd.read_csv(evaluation_output)
    if "row_id" not in oof.columns:
        aliases = [
            alias
            for alias in ("sample_id", "id")
            if alias in oof.columns
        ]
        if (
            len(aliases) == 1
            and (
                "target" in oof.columns
                or "prediction" in oof.columns
                or any(
                    str(column).startswith("prediction::")
                    for column in oof.columns
                )
            )
        ):
            oof = oof.rename(columns={aliases[0]: "row_id"})
            # Persist the canonical form so pruning and diversity policies see
            # the same schema that the evaluator accepted.
            oof.to_csv(evaluation_output, index=False)
    if manifest.get("task_type") == "unsupervised_clustering":
        required = {"row_id", "prediction", "fold_id"}
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
            "score": float(np.mean(fold_scores)),
            "score_std": float(np.std(fold_scores)),
            "evaluation_mode": "task_native",
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

    required = {"row_id", "target", "fold_id"}
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
    resolved_metric = resolve_metric_name(
        metric_name,
        problem_type=manifest.get("task_type"),
        output_type=manifest.get("output_type"),
    )
    prediction, class_names = legacy_prediction_payload(merged)
    if expected_class_names and tuple(class_names) != tuple(
        expected_class_names
    ):
        raise ValueError(
            "OOF class-probability columns differ from the task output contract"
        )
    if np.issubdtype(prediction.dtype, np.number) and not np.isfinite(
        prediction.astype(float, copy=False)
    ).all():
        raise ValueError("OOF predictions contain non-finite values")
    target = merged["target"].to_numpy()
    fold_ids = merged["expected_fold_id"].to_numpy()
    fold_scores = [
        metric_value(
            resolved_metric,
            target[fold_ids == fold],
            prediction[fold_ids == fold],
            class_names=class_names,
        )
        for fold in sorted(np.unique(fold_ids))
    ]
    return {
        "score": float(np.mean(fold_scores)),
        "score_std": float(np.std(fold_scores)),
        "evaluation_mode": "cross_validation",
        "cv_mean": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "folds": len(fold_scores),
        "fold_scores": fold_scores,
        "seed": int(manifest["seed"]),
        "fidelity": fidelity,
        "row_count": int(manifest["row_count"]),
        "source_row_count": int(manifest["source_row_count"]),
        "fold_assignment_sha256": manifest["fold_assignment_sha256"],
        "metric": resolved_metric,
        "direction": infer_metric_direction(resolved_metric),
        "task_type": manifest.get("task_type"),
    }
