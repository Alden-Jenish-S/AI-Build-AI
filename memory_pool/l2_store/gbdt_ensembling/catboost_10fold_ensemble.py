"""Cross-validated CatBoost predictions with optional CUDA acceleration."""

from __future__ import annotations

import os
import warnings

import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import KFold, StratifiedKFold


def _requested_accelerator(device=None):
    requested = str(
        device or os.environ.get("AIBUILDAI_ACCELERATOR", "cpu")
    ).lower()
    # CatBoost's GPU backend is CUDA-only; Apple MPS therefore uses CPU.
    return "cuda" if requested in {"cuda", "gpu"} else "cpu"


def _model_params(device, overrides=None):
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = device
    params = {
        "iterations": 100,
        "learning_rate": 0.05,
        "depth": 6,
        "verbose": 0,
        "random_seed": 42,
        "allow_writing_files": False,
    }
    params.update(dict(overrides or {}))
    params.pop("task_type", None)
    params.pop("devices", None)
    if device == "cuda":
        params["task_type"] = "GPU"
        params["devices"] = os.environ.get("AIBUILDAI_CUDA_DEVICES", "0")
    else:
        params["task_type"] = "CPU"
    return params


def _fit_with_cpu_fallback(model_class, params, X_tr, y_tr, X_va, y_va, cat_features):
    model = model_class(**params)
    try:
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=20,
            cat_features=cat_features,
            verbose=False,
        )
        return model
    except Exception as exc:
        if params.get("task_type") != "GPU":
            raise
        warnings.warn(
            f"CatBoost CUDA backend failed ({exc}); retrying this fold on CPU.",
            RuntimeWarning,
        )
        cpu_params = dict(params)
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
        cpu_params["task_type"] = "CPU"
        cpu_params.pop("devices", None)
        model = model_class(**cpu_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=(X_va, y_va),
            early_stopping_rounds=20,
            cat_features=cat_features,
            verbose=False,
        )
        return model


def fit_predict(
    X_train,
    y_train,
    X_test,
    cat_features=None,
    n_folds=10,
    is_classification=True,
    device=None,
    catboost_params=None,
):
    """Return OOF predictions, averaged test predictions, and fitted models."""
    y_train = np.asarray(y_train)
    X_train = X_train.copy()
    X_test = X_test.copy()
    if cat_features is not None:
        for column in cat_features:
            X_train[column] = X_train[column].astype(str)
            X_test[column] = X_test[column].astype(str)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    model_list = []
    requested_device = _requested_accelerator(device)
    params = _model_params(requested_device, catboost_params)

    if is_classification:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        model_class = CatBoostClassifier
    else:
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        model_class = CatBoostRegressor

    for train_idx, val_idx in folds.split(X_train, y_train):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train[val_idx]
        model = _fit_with_cpu_fallback(
            model_class, params, X_tr, y_tr, X_va, y_va, cat_features
        )
        if is_classification:
            oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        else:
            oof_preds[val_idx] = model.predict(X_va)
            test_preds += model.predict(X_test) / n_folds
        model_list.append(model)

    return oof_preds, test_preds, model_list
