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
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
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
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = (
            "cuda" if params.get("device_type") == "gpu" else "cpu"
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
    fold_ids=None,
):
    """Return OOF predictions, averaged test predictions, and fitted models."""
    y_train = np.asarray(y_train)
    classes = np.unique(y_train) if is_classification else np.asarray([])
    multiclass = is_classification and len(classes) > 2
    prediction_shape = (
        (len(X_train), len(classes)) if multiclass else (len(X_train),)
    )
    test_shape = (
        (len(X_test), len(classes)) if multiclass else (len(X_test),)
    )
    oof_preds = np.zeros(prediction_shape)
    test_preds = np.zeros(test_shape)
    model_list = []
    params = _model_params(_requested_accelerator(device), lightgbm_params)

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
        splits = list(folds.split(X_train, y_train))
        model_class = LGBMClassifier
    else:
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(folds.split(X_train, y_train))
        model_class = LGBMRegressor
    model_class = LGBMClassifier if is_classification else LGBMRegressor
    fold_count = len(splits)

    for train_idx, val_idx in splits:
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train[val_idx]
        model = _fit_with_cpu_fallback(
            model_class, params, X_tr, y_tr, X_va, y_va, cat_features
        )
        if is_classification:
            validation_proba = model.predict_proba(X_va)
            test_proba = model.predict_proba(X_test)
            if multiclass:
                columns = np.searchsorted(classes, model.classes_)
                oof_preds[np.ix_(val_idx, columns)] = validation_proba
                test_preds[:, columns] += test_proba / fold_count
            else:
                oof_preds[val_idx] = validation_proba[:, 1]
                test_preds += test_proba[:, 1] / fold_count
        else:
            oof_preds[val_idx] = model.predict(X_va)
            test_preds += model.predict(X_test) / fold_count
        model_list.append(model)

    final_model = _fit_with_cpu_fallback(
        model_class,
        params,
        X_train,
        y_train,
        X_train,
        y_train,
        cat_features,
    )
    if is_classification:
        final_proba = final_model.predict_proba(X_test)
        if multiclass:
            test_preds = np.zeros((len(X_test), len(classes)))
            columns = np.searchsorted(classes, final_model.classes_)
            test_preds[:, columns] = final_proba
        else:
            test_preds = final_proba[:, 1]
    else:
        test_preds = final_model.predict(X_test)
    model_list.append(final_model)
    return oof_preds, test_preds, model_list
