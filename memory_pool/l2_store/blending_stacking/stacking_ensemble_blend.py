"""OOF stacking blend with CUDA-preferred tree learners and CPU fallback."""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def _prefer_cuda(device=None):
    requested = str(
        device or os.environ.get("AIBUILDAI_ACCELERATOR", "cpu")
    ).lower()
    return requested in {"cuda", "gpu"}


def _make_model(name, use_cuda):
    if name == "catboost":
        from catboost import CatBoostClassifier

        params = {
            "verbose": False,
            "random_seed": 42,
            "allow_writing_files": False,
            "task_type": "GPU" if use_cuda else "CPU",
        }
        if use_cuda:
            params["devices"] = os.environ.get("AIBUILDAI_CUDA_DEVICES", "0")
        return CatBoostClassifier(**params)
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            verbose=-1,
            random_state=42,
            device_type="gpu" if use_cuda else "cpu",
        )
    return LogisticRegression(max_iter=1000, random_state=42)


def _fit_model(name, use_cuda, X, y):
    model = _make_model(name, use_cuda)
    try:
        model.fit(X, y)
        return model, use_cuda
    except Exception as exc:
        if not use_cuda or name == "lr":
            raise
        warnings.warn(
            f"{name} GPU backend failed ({exc}); retrying on CPU.",
            RuntimeWarning,
        )
        model = _make_model(name, False)
        model.fit(X, y)
        return model, False


def train_and_predict(X_train, y_train, X_test, device=None):
    """Train base learners out-of-fold, then return a blended stack prediction."""
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    model_names = ["catboost", "lgbm", "lr"]
    gpu_enabled = {
        name: _prefer_cuda(device) and name != "lr" for name in model_names
    }
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = (
        "cuda" if any(gpu_enabled.values()) else "cpu"
    )
    n_folds = 5
    oof = np.zeros((X_train.shape[0], len(model_names)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fitted_models = []

    for model_index, name in enumerate(model_names):
        for train_idx, hold_idx in skf.split(X_train, y_train):
            fold_model, used_gpu = _fit_model(
                name,
                gpu_enabled[name],
                X_train[train_idx],
                y_train[train_idx],
            )
            gpu_enabled[name] = used_gpu
            oof[hold_idx, model_index] = fold_model.predict_proba(
                X_train[hold_idx]
            )[:, 1]
        full_model, used_gpu = _fit_model(
            name, gpu_enabled[name], X_train, y_train
        )
        gpu_enabled[name] = used_gpu
        fitted_models.append(full_model)

    if not any(gpu_enabled.values()):
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"

    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(oof, y_train)
    base_probs = np.column_stack(
        [model.predict_proba(X_test)[:, 1] for model in fitted_models]
    )
    meta_prob = meta.predict_proba(base_probs)[:, 1]
    return 0.5 * meta_prob + 0.5 * base_probs.mean(axis=1)
