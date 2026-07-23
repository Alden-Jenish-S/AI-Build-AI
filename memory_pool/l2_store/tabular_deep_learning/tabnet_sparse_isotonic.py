"""Sparse TabNet regression with leakage-safe OOF isotonic calibration."""

import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def _device_name() -> str:
    preferred = os.getenv("AIBUILDAI_ACCELERATOR", "cpu").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cuda"
        return "cuda"
    # pytorch-tabnet's stable device contract is CPU/CUDA. MPS requests use
    # the explicit safe CPU fallback instead of claiming unsupported execution.
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
    return "cpu"


def _preprocessor(
    frame: pd.DataFrame,
    cat_features: Optional[List[str]],
    numeric_features: Optional[List[str]],
) -> ColumnTransformer:
    categorical = (
        list(cat_features)
        if cat_features is not None
        else frame.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
    )
    numeric = (
        list(numeric_features)
        if numeric_features is not None
        else [
            column
            for column in frame.select_dtypes(
                include=[np.number, "bool"]
            ).columns
            if column not in categorical
        ]
    )
    transformers = []
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                keep_empty_features=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if not transformers:
        raise ValueError("No usable categorical or numeric columns were found")
    return ColumnTransformer(transformers, remainder="drop")


def _new_model(
    n_d: int,
    n_a: int,
    n_steps: int,
    gamma: float,
    lambda_sparse: float,
    learning_rate: float,
    weight_decay: float,
    random_state: int,
) -> TabNetRegressor:
    return TabNetRegressor(
        n_d=int(n_d),
        n_a=int(n_a),
        n_steps=int(n_steps),
        gamma=float(gamma),
        lambda_sparse=float(lambda_sparse),
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={
            "lr": float(learning_rate),
            "weight_decay": float(weight_decay),
        },
        mask_type="entmax",
        seed=int(random_state),
        device_name=_device_name(),
        verbose=0,
    )


def train_and_predict(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train=None,
    cat_features: Optional[List[str]] = None,
    numeric_features: Optional[List[str]] = None,
    n_folds: int = 3,
    n_d: int = 8,
    n_a: int = 8,
    n_steps: int = 3,
    gamma: float = 1.5,
    lambda_sparse: float = 0.0001,
    learning_rate: float = 0.02,
    weight_decay: float = 0.00001,
    batch_size: int = 256,
    random_state: int = 42,
) -> pd.Series:
    """Return TabNet fold-ensemble predictions aligned to X_test.

    Every preprocessor is fitted on fold-training data only. OOF predictions
    train an isotonic calibrator, which is then applied to the fold-averaged
    test predictions.
    """
    if y_train is None:
        raise ValueError("y_train is required")
    train = pd.DataFrame(X_train).copy()
    test = pd.DataFrame(X_test).copy().reindex(columns=train.columns)
    target = np.asarray(y_train, dtype=np.float32).reshape(-1)
    if len(train) != len(target):
        raise ValueError("X_train and y_train must have the same length")
    folds = max(2, min(int(n_folds), len(train)))
    max_epochs = max(1, int(os.getenv("AIBUILDAI_MAX_EPOCHS", "50")))
    patience = max(
        1,
        min(
            int(os.getenv("AIBUILDAI_EARLY_STOPPING_PATIENCE", "10")),
            max_epochs,
        ),
    )
    effective_batch = max(2, min(int(batch_size), len(train)))

    splitter = KFold(
        n_splits=folds, shuffle=True, random_state=int(random_state)
    )
    oof = np.zeros(len(train), dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    for fold, (training_indices, validation_indices) in enumerate(
        splitter.split(train)
    ):
        fold_preprocessor = _preprocessor(
            train.iloc[training_indices],
            cat_features,
            numeric_features,
        )
        fold_train = np.asarray(
            fold_preprocessor.fit_transform(train.iloc[training_indices]),
            dtype=np.float32,
        )
        fold_validation = np.asarray(
            fold_preprocessor.transform(train.iloc[validation_indices]),
            dtype=np.float32,
        )
        fold_test = np.asarray(
            fold_preprocessor.transform(test), dtype=np.float32
        )
        fold_target = target[training_indices].reshape(-1, 1)
        validation_target = target[validation_indices].reshape(-1, 1)
        model = _new_model(
            n_d,
            n_a,
            n_steps,
            gamma,
            lambda_sparse,
            learning_rate,
            weight_decay,
            int(random_state) + fold,
        )
        model.fit(
            fold_train,
            fold_target,
            eval_set=[(fold_validation, validation_target)],
            eval_name=["validation"],
            eval_metric=["rmse"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=min(effective_batch, len(training_indices)),
            virtual_batch_size=max(
                2,
                min(
                    128,
                    effective_batch,
                    len(training_indices),
                ),
            ),
            num_workers=0,
            drop_last=False,
        )
        oof[validation_indices] = model.predict(
            fold_validation
        ).reshape(-1)
        test_predictions += model.predict(fold_test).reshape(-1) / folds

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(oof, target)
    calibrated = calibrator.predict(test_predictions)
    calibrated = np.clip(calibrated, float(target.min()), float(target.max()))
    return pd.Series(calibrated, index=test.index, dtype=float)
