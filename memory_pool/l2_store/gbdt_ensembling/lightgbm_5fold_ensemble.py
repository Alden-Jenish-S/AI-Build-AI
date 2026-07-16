"""Cross-validated LightGBM predictions with optional GPU acceleration."""

from __future__ import annotations

import os
import warnings

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold


def _requested_accelerator(device=None):
    requested = str(
        device or os.environ.get("AIBUILDAI_ACCELERATOR", "cpu")
    ).lower()
    return "gpu" if requested in {"cuda", "gpu"} else "cpu"


def _model_params(device, overrides=None):
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = (
        "cuda" if device == "gpu" else "cpu"
    )
    params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "verbose": -1,
    }
    params.update(dict(overrides or {}))
    params["device_type"] = device
    return params


def _fit_with_cpu_fallback(
    model_class, params, X_tr, y_tr, X_va, y_va, cat_features
):
    model = model_class(**params)
    try:
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_features,
        )
        return model
    except Exception as exc:
        if params.get("device_type") != "gpu":
            raise
        warnings.warn(
            f"LightGBM GPU backend failed ({exc}); retrying this fold on CPU.",
            RuntimeWarning,
        )
        cpu_params = dict(params)
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
        cpu_params["device_type"] = "cpu"
        model = model_class(**cpu_params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_features,
        )
        return model


def fit_predict(
    X_train,
    y_train,
    X_test,
    cat_features=None,
    n_folds=5,
    is_classification=True,
    device=None,
    lightgbm_params=None,
):
    """Return OOF predictions, averaged test predictions, and fitted models."""
    y_train = np.asarray(y_train)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    model_list = []
    params = _model_params(_requested_accelerator(device), lightgbm_params)

    if is_classification:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        model_class = LGBMClassifier
    else:
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        model_class = LGBMRegressor

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
