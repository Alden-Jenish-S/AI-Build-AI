"""Mini-batch factorization machine for mixed tabular binary data."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _runtime_limit(name, default):
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return max(1, int(default))


def _encode_binary_target(y):
    values = np.asarray(y).reshape(-1)
    if pd.isna(values).any():
        raise ValueError("binary target contains missing values")
    classes = sorted(pd.unique(values).tolist(), key=lambda value: str(value))
    if len(classes) != 2:
        raise ValueError(f"binary target must contain exactly two classes; got {classes}")
    mapping = {value: index for index, value in enumerate(classes)}
    return np.asarray([mapping[value] for value in values], dtype=np.float32)


class FactorizationMachine(nn.Module):
    def __init__(self, n_features: int, k: int = 10):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)
        self.V = nn.Parameter(torch.empty(n_features, k))
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.V, std=0.01)

    def forward(self, x):
        linear_part = self.linear(x).squeeze(1)
        interaction = 0.5 * torch.sum(
            torch.pow(torch.matmul(x, self.V), 2)
            - torch.matmul(torch.pow(x, 2), torch.pow(self.V, 2)),
            dim=1,
        )
        return linear_part + interaction


def _resolve_device(device=None):
    requested = str(
        device or os.environ.get("AIBUILDAI_ACCELERATOR", "cpu")
    ).lower()
    if requested in {"cuda", "gpu"} and torch.cuda.is_available():
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cuda"
        return "cuda"
    if (
        requested == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "mps"
        return "mps"
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
    return "cpu"


def _build_preprocessor(df):
    usable_cols = [column for column in df.columns if not df[column].isna().all()]
    if not usable_cols:
        raise ValueError("training data has no usable non-empty feature columns")
    num_cols = df[usable_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [column for column in usable_cols if column not in num_cols]
    numeric = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    transformers = []
    if num_cols:
        transformers.append(("num", numeric, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical, cat_cols))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _fit_predict_tensors(
    X_train,
    y_train,
    X_test,
    *,
    k,
    lr,
    weight_decay,
    epochs,
    batch_size,
    patience,
    device,
    random_state,
):
    torch.manual_seed(int(random_state))
    train_tensor = torch.from_numpy(X_train)
    target_tensor = torch.from_numpy(y_train)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_tensor, target_tensor),
        batch_size=batch_size,
        shuffle=True,
    )
    model = FactorizationMachine(X_train.shape[1], k).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses))
        print(f"Epoch {epoch + 1:02d}/{epochs} | loss={mean_loss:.5f}")
        if mean_loss < best_loss - 1e-5:
            best_loss = mean_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, int(patience)):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    prediction_loader = torch.utils.data.DataLoader(
        torch.from_numpy(X_test), batch_size=batch_size, shuffle=False
    )
    predictions = []
    with torch.no_grad():
        for xb in prediction_loader:
            predictions.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(predictions)


def train_and_predict(
    train_df,
    test_df,
    y_train=None,
    target_col="target",
    k=10,
    lr=0.01,
    weight_decay=1e-5,
    epochs=30,
    batch_size=1024,
    patience=5,
    device=None,
    random_state=42,
):
    """Train on a target-bearing frame or a separate ``y_train`` array."""
    if y_train is None:
        if target_col not in train_df:
            raise ValueError("Provide y_train or include target_col in train_df")
        y_train = train_df[target_col].to_numpy()
        X_train = train_df.drop(columns=[target_col])
    else:
        X_train = train_df.drop(columns=[target_col], errors="ignore")
    X_test = test_df[X_train.columns].copy()
    preprocessor = _build_preprocessor(X_train)
    X_train_proc = np.asarray(
        preprocessor.fit_transform(X_train), dtype=np.float32
    )
    X_test_proc = np.asarray(preprocessor.transform(X_test), dtype=np.float32)
    y_array = _encode_binary_target(y_train)
    selected_device = _resolve_device(device)
    bounded_epochs = min(
        max(1, int(epochs)), _runtime_limit("AIBUILDAI_MAX_EPOCHS", epochs)
    )
    bounded_patience = min(
        max(1, int(patience)),
        _runtime_limit("AIBUILDAI_EARLY_STOPPING_PATIENCE", patience),
    )
    kwargs = dict(
        k=int(k),
        lr=float(lr),
        weight_decay=float(weight_decay),
        epochs=bounded_epochs,
        batch_size=max(1, int(batch_size)),
        patience=bounded_patience,
        random_state=int(random_state),
    )
    try:
        return _fit_predict_tensors(
            X_train_proc, y_array, X_test_proc, device=selected_device, **kwargs
        )
    except RuntimeError as exc:
        if selected_device == "cpu" or not any(
            marker in str(exc).lower()
            for marker in ("out of memory", "cuda", "cudnn", "mps")
        ):
            raise
        warnings.warn(
            f"Accelerator training failed ({exc}); retrying on CPU.", RuntimeWarning
        )
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
        return _fit_predict_tensors(
            X_train_proc, y_array, X_test_proc, device="cpu", **kwargs
        )
