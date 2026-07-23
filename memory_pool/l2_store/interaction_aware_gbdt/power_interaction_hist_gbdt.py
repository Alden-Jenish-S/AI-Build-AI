"""Reusable power-transformed interaction features with histogram GBDT."""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler


def train_and_predict(
    X_train,
    X_test,
    y_train=None,
    n_top=10,
    max_interactions=45,
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    l2_regularization=0.0,
    random_state=42,
):
    """Fit the node-21 methodology and return predictions aligned to X_test."""
    if y_train is None:
        raise ValueError("y_train is required")
    train = pd.DataFrame(X_train).copy()
    test = pd.DataFrame(X_test).copy()
    test = test.reindex(columns=train.columns)
    target = np.asarray(y_train, dtype=float).reshape(-1)
    if len(train) != len(target):
        raise ValueError("X_train and y_train must have the same length")

    numeric = [
        column
        for column in train.select_dtypes(include=[np.number, "bool"]).columns
        if train[column].notna().any()
    ]
    if not numeric:
        return np.full(len(test), float(np.mean(target)), dtype=float)

    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train[numeric])
    test_values = imputer.transform(test[numeric])

    scaler = StandardScaler()
    train_values = scaler.fit_transform(train_values)
    test_values = scaler.transform(test_values)

    # Yeo-Johnson is unstable on constant columns, so transform only columns
    # whose fold-training variance is meaningful.
    variable = np.std(train_values, axis=0) > 1e-12
    if variable.any():
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        train_values[:, variable] = transformer.fit_transform(
            train_values[:, variable]
        )
        test_values[:, variable] = transformer.transform(
            test_values[:, variable]
        )

    correlations = []
    centered_target = target - np.mean(target)
    target_scale = np.sqrt(np.sum(centered_target**2))
    for index in range(train_values.shape[1]):
        centered = train_values[:, index] - np.mean(train_values[:, index])
        scale = np.sqrt(np.sum(centered**2)) * target_scale
        correlation = (
            abs(float(np.sum(centered * centered_target) / scale))
            if scale > 0
            else 0.0
        )
        correlations.append(correlation)
    chosen = np.argsort(correlations)[::-1][
        : max(0, min(int(n_top), len(numeric)))
    ]

    train_parts = [train_values]
    test_parts = [test_values]
    interactions = 0
    for left_position, left in enumerate(chosen):
        for right in chosen[left_position + 1 :]:
            if interactions >= max(0, int(max_interactions)):
                break
            train_parts.append(
                (train_values[:, left] * train_values[:, right]).reshape(-1, 1)
            )
            test_parts.append(
                (test_values[:, left] * test_values[:, right]).reshape(-1, 1)
            )
            interactions += 1
        if interactions >= max(0, int(max_interactions)):
            break

    train_matrix = np.hstack(train_parts).astype(np.float32, copy=False)
    test_matrix = np.hstack(test_parts).astype(np.float32, copy=False)
    iteration_cap = int(
        os.getenv("AIBUILDAI_MAX_ESTIMATOR_ITERATIONS", str(max_iter))
    )
    model = HistGradientBoostingRegressor(
        max_iter=max(1, min(int(max_iter), iteration_cap)),
        learning_rate=float(learning_rate),
        max_depth=None if max_depth is None else int(max_depth),
        l2_regularization=float(l2_regularization),
        early_stopping=True,
        validation_fraction=0.2,
        random_state=int(random_state),
    )
    model.fit(train_matrix, target)
    return np.asarray(model.predict(test_matrix), dtype=float).reshape(-1)
