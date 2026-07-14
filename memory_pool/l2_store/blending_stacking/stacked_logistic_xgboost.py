"""Leakage-safe Logistic Regression and XGBoost stacking for binary data."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _prepare_xy(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if target_col not in train_df:
        raise ValueError(f"Target column not found: {target_col!r}")
    feature_columns = [column for column in train_df.columns if column != target_col]
    missing_columns = [column for column in feature_columns if column not in test_df]
    if missing_columns:
        raise ValueError(f"Test data is missing feature columns: {missing_columns}")

    X = train_df[feature_columns].to_numpy(dtype=float)
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    y = train_df[target_col].to_numpy()
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(f"Binary target required; found {len(classes)} classes")
    return X, y, X_test


def fit_predict_stacked(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target",
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """Train a two-model stack and return positive-class test probabilities."""
    X, y, X_test = _prepare_xy(train_df, test_df, target_col)
    min_class_count = int(np.min(np.unique(y, return_counts=True)[1]))
    effective_splits = min(int(n_splits), min_class_count)
    if effective_splits < 2:
        raise ValueError("Each target class must contain at least two rows")

    oof_predictions = np.zeros((len(X), 2), dtype=float)
    test_predictions = np.zeros((len(X_test), 2), dtype=float)
    folds = StratifiedKFold(
        n_splits=effective_splits, shuffle=True, random_state=random_state
    )

    for train_indices, valid_indices in folds.split(X, y):
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train = y[train_indices]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

        logistic = LogisticRegression(max_iter=1000, random_state=random_state)
        logistic.fit(X_train_scaled, y_train)
        oof_predictions[valid_indices, 0] = logistic.predict_proba(X_valid_scaled)[:, 1]
        test_predictions[:, 0] += (
            logistic.predict_proba(X_test_scaled)[:, 1] / effective_splits
        )

        xgboost = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
            tree_method="hist",
        )
        xgboost.fit(X_train, y_train)
        oof_predictions[valid_indices, 1] = xgboost.predict_proba(X_valid)[:, 1]
        test_predictions[:, 1] += (
            xgboost.predict_proba(X_test)[:, 1] / effective_splits
        )

    meta_learner = LogisticRegression(max_iter=1000, random_state=random_state)
    meta_learner.fit(oof_predictions, y)
    return meta_learner.predict_proba(test_predictions)[:, 1]


def fit_predict_stacked_entrypoint(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target",
) -> np.ndarray:
    """Builder-compatible wrapper around :func:`fit_predict_stacked`."""
    return fit_predict_stacked(train_df, test_df, target_col=target_col)
