"""Cross-validated XGBoost predictions with optional CUDA acceleration."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
import xgboost
from xgboost import XGBClassifier, XGBRegressor


def _requested_accelerator(device=None):
    requested = str(
        device or os.environ.get("AIBUILDAI_ACCELERATOR", "cpu")
    ).lower()
    if requested not in {"cuda", "gpu"}:
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
        return "cpu"
    try:
        cuda_built = bool(xgboost.build_info().get("USE_CUDA", False))
    except Exception:
        cuda_built = True
    if cuda_built:
        return "cuda"
    warnings.warn(
        "The installed XGBoost build has no CUDA support; using CPU.",
        RuntimeWarning,
    )
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
    return "cpu"


def _model_params(device, overrides=None):
    params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
        "enable_categorical": True,
        "verbosity": 0,
        "tree_method": "hist",
    }
    params.update(dict(overrides or {}))
    major_version = int(xgboost.__version__.split(".", 1)[0])
    if major_version >= 2:
        params["device"] = "cuda" if device == "cuda" else "cpu"
    elif device == "cuda":
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
    else:
        params.pop("device", None)
    return params


def _fit_with_cpu_fallback(model_class, params, X_tr, y_tr, X_va, y_va):
    model = model_class(**params)
    try:
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = (
            "cuda"
            if params.get("device") == "cuda"
            or params.get("tree_method") == "gpu_hist"
            else "cpu"
        )
        return model
    except Exception as exc:
        if params.get("device") != "cuda" and params.get("tree_method") != "gpu_hist":
            raise
        warnings.warn(
            f"XGBoost CUDA backend failed ({exc}); retrying this fold on CPU.",
            RuntimeWarning,
        )
        cpu_params = dict(params)
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
        if int(xgboost.__version__.split(".", 1)[0]) >= 2:
            cpu_params["device"] = "cpu"
        else:
            cpu_params.pop("device", None)
        cpu_params["tree_method"] = "hist"
        cpu_params.pop("predictor", None)
        model = model_class(**cpu_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return model


def fit_predict(
    X_train,
    y_train,
    X_test,
    cat_features=None,
    n_folds=5,
    is_classification=True,
    device=None,
    xgboost_params=None,
    fold_ids=None,
):
    """Return OOF predictions, averaged test predictions, and fitted models."""
    y_train = np.asarray(y_train)
    classes = np.unique(y_train) if is_classification else np.asarray([])
    multiclass = is_classification and len(classes) > 2
    oof_preds = np.zeros(
        (len(X_train), len(classes)) if multiclass else len(X_train)
    )
    test_preds = np.zeros(
        (len(X_test), len(classes)) if multiclass else len(X_test)
    )
    model_list = []

    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    if cat_features is not None:
        for col in cat_features:
            categories = pd.Index(
                X_train_xgb[col].astype(str).unique()
            ).sort_values().tolist()
            dtype = pd.CategoricalDtype(categories=categories)
            X_train_xgb[col] = X_train_xgb[col].astype(str).astype(dtype)
            X_test_xgb[col] = X_test_xgb[col].astype(str).astype(dtype)

    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids)
        if len(fold_ids) != len(X_train):
            raise ValueError("fold_ids must align with X_train")
        unique_folds = np.unique(fold_ids)
        if len(unique_folds) < 2:
            raise ValueError("fold_ids must contain at least two folds")
        splits = [
            (np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold))
            for fold in unique_folds
        ]
    elif is_classification:
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(folds.split(X_train_xgb, y_train))
        model_class = XGBClassifier
    else:
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(folds.split(X_train_xgb, y_train))
        model_class = XGBRegressor
    model_class = XGBClassifier if is_classification else XGBRegressor
    fold_count = len(splits)
    params = _model_params(_requested_accelerator(device), xgboost_params)

    for train_idx, val_idx in splits:
        X_tr = X_train_xgb.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train_xgb.iloc[val_idx]
        y_va = y_train[val_idx]
        model = _fit_with_cpu_fallback(model_class, params, X_tr, y_tr, X_va, y_va)
        if is_classification:
            validation_proba = model.predict_proba(X_va)
            test_proba = model.predict_proba(X_test_xgb)
            if multiclass:
                columns = np.searchsorted(classes, model.classes_)
                oof_preds[np.ix_(val_idx, columns)] = validation_proba
                test_preds[:, columns] += test_proba / fold_count
            else:
                oof_preds[val_idx] = validation_proba[:, 1]
                test_preds += test_proba[:, 1] / fold_count
        else:
            oof_preds[val_idx] = model.predict(X_va)
            test_preds += model.predict(X_test_xgb) / fold_count
        model_list.append(model)

    final_model = _fit_with_cpu_fallback(
        model_class,
        params,
        X_train_xgb,
        y_train,
        X_train_xgb,
        y_train,
    )
    if is_classification:
        final_proba = final_model.predict_proba(X_test_xgb)
        if multiclass:
            test_preds = np.zeros((len(X_test_xgb), len(classes)))
            columns = np.searchsorted(classes, final_model.classes_)
            test_preds[:, columns] = final_proba
        else:
            test_preds = final_proba[:, 1]
    else:
        test_preds = final_model.predict(X_test_xgb)
    model_list.append(final_model)
    return oof_preds, test_preds, model_list
