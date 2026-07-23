# -*- coding: utf-8 -*-
"""
Tabular Token‑Transformer regression model.

The model stores categorical mappings and numeric statistics learned during
``fit`` so inference uses training-owned preprocessing state.
"""

import os
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------------------------------------------------


def _default_device() -> torch.device:
    """Select device based on the AIBUILDAI_ACCELERATOR env var with safe fallback."""
    preferred = os.getenv("AIBUILDAI_ACCELERATOR", "cpu").lower()
    if preferred == "cuda" and torch.cuda.is_available():
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cuda"
        return torch.device("cuda")
    if preferred == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "mps"
        return torch.device("mps")
    # The manager already selected CUDA > MPS > CPU. If that selected backend
    # is not usable by this installed framework, take the explicit CPU fallback.
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
    return torch.device("cpu")


class TabularDataset(Dataset):
    """Dataset that converts a pandas DataFrame into token indices/values for the transformer."""

    def __init__(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: Optional[str] = None,
        cat_maps: Optional[Dict[str, Dict]] = None,
        num_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.target_col = target_col

        # Build / reuse categorical mappings
        if cat_maps is None:
            self.cat_maps = {
                c: {v: i for i, v in enumerate(df[c].astype("category").cat.categories)}
                for c in cat_cols
            }
        else:
            self.cat_maps = cat_maps

        # Build / reuse numeric statistics (mean, std)
        if num_stats is None:
            self.num_stats = {}
            for column in num_cols:
                values = pd.to_numeric(df[column], errors="coerce")
                mean = float(values.mean()) if values.notna().any() else 0.0
                std = float(values.std()) if values.notna().sum() > 1 else 1.0
                if not np.isfinite(std) or std <= 0:
                    std = 1.0
                self.num_stats[column] = (mean, std)
        else:
            self.num_stats = num_stats

        self.raw = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx):
        row = self.raw.iloc[idx]
        # categorical indices
        cat_idx = torch.tensor(
            [self.cat_maps[c].get(row[c], 0) for c in self.cat_cols], dtype=torch.long
        )
        # numeric values – normalised
        num_vals = torch.tensor(
            [
                (
                    self.num_stats[c][0]
                    if pd.isna(row[c])
                    else float(row[c])
                )
                - self.num_stats[c][0]
                for c in self.num_cols
            ],
            dtype=torch.float,
        )
        if self.num_cols:
            scales = torch.tensor(
                [self.num_stats[c][1] for c in self.num_cols],
                dtype=torch.float,
            )
            num_vals = num_vals / scales
        if self.target_col is not None:
            target = torch.tensor(row[self.target_col], dtype=torch.float)
            return cat_idx, num_vals, target
        return cat_idx, num_vals


class TabularTokenTransformerRegressor(nn.Module):
    """Feature‑Token Transformer for tabular regression."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        mixup_alpha: float = 0.2,
        mixup_calibrate: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or _default_device()
        self.d_model = d_model
        self.mixup_alpha = mixup_alpha
        self.mixup_calibrate = mixup_calibrate

        # Embedding blocks
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(card, d_model) for card in cat_cardinalities]
        )
        self.num_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional / type embeddings
        total_tokens = len(cat_cardinalities) + num_features
        self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, d_model) * 0.02)
        type_bias = torch.cat(
            [
                torch.zeros(len(cat_cardinalities), dtype=torch.long),
                torch.ones(num_features, dtype=torch.long),
            ]
        )
        self.register_buffer("type_bias", type_bias)
        self.type_embed = nn.Embedding(2, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Adaptive attention pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),
            nn.Softmax(dim=1),
        )

        # Regression head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        # placeholders for mappings (filled in ``fit``)
        self.cat_maps: Optional[Dict[str, Dict]] = None
        self.num_stats: Optional[Dict[str, Tuple[float, float]]] = None

        self.to(self.device)

    # ------------------------------------------------------------------
    def _embed(self, cat_idx: torch.Tensor, num_vals: torch.Tensor) -> torch.Tensor:
        """Create token embeddings for a batch."""
        # Categorical embeddings – one per column
        if self.cat_embeds:
            cat_emb = torch.stack(
                [emb(cat_idx[:, i]) for i, emb in enumerate(self.cat_embeds)],
                dim=1,
            )
        else:
            cat_emb = torch.empty(
                (cat_idx.shape[0], 0, self.d_model),
                device=cat_idx.device,
            )

        # Numeric embeddings – project each scalar
        if num_vals.shape[1]:
            num_vals = num_vals.unsqueeze(-1)
            num_emb = self.num_proj(num_vals)
        else:
            num_emb = torch.empty(
                (cat_idx.shape[0], 0, self.d_model),
                device=cat_idx.device,
            )

        # Concatenate and add positional / type information
        tokens = torch.cat([cat_emb, num_emb], dim=1)  # (B, N, d_model)
        tokens = tokens + self.pos_embed  # broadcast (1, N, d_model)
        tokens = tokens + self.type_embed(self.type_bias)  # (N, d_model) broadcast over batch
        return tokens

    # ------------------------------------------------------------------
    def _mixup(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calibrated MixUp for regression on the token space."""
        if self.mixup_alpha <= 0:
            return x, y
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch = x.size(0)

        if self.mixup_calibrate:
            with torch.no_grad():
                y1 = y.unsqueeze(1)  # (B,1)
                y2 = y.unsqueeze(0)  # (1,B)
                dist = torch.abs(y1 - y2)
                sigma = dist.mean().clamp(min=1e-6)
                sim = torch.exp(-dist / sigma)  # (B,B)
                idx = torch.multinomial(sim, 1).squeeze()
                lam = (
                    lam * sim[torch.arange(batch, device=y.device), idx]
                    + (1 - lam)
                    * (
                        1
                        - sim[
                            torch.arange(batch, device=y.device),
                            idx,
                        ]
                    )
                )

        perm = torch.randperm(batch, device=self.device)
        if isinstance(lam, torch.Tensor):
            x_lam = lam.reshape(-1, 1, 1)
            y_lam = lam.reshape(-1)
        else:
            x_lam = y_lam = lam
        x_mix = x_lam * x + (1 - x_lam) * x[perm]
        y_mix = y_lam * y + (1 - y_lam) * y[perm]
        return x_mix, y_mix

    # ------------------------------------------------------------------
    def forward(self, cat_idx: torch.Tensor, num_vals: torch.Tensor) -> torch.Tensor:
        tokens = self._embed(cat_idx, num_vals)          # (B, N, d_model)
        enc = self.transformer(tokens)                   # (B, N, d_model)
        weights = self.attn_pool(enc).squeeze(-1)       # (B, N)
        pooled = (weights.unsqueeze(-1) * enc).sum(dim=1)  # (B, d_model)
        out = self.head(pooled).squeeze(-1)             # (B,)
        return out

    # ------------------------------------------------------------------
    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        target_col: str,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Training loop with early stopping."""
        train_set = TabularDataset(train_df, cat_cols, num_cols, target_col=target_col)
        val_set = TabularDataset(
            val_df,
            cat_cols,
            num_cols,
            target_col=target_col,
            cat_maps=train_set.cat_maps,
            num_stats=train_set.num_stats,
        )

        # store mappings for later inference
        self.cat_maps = train_set.cat_maps
        self.num_stats = train_set.num_stats

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )
        criterion = nn.MSELoss()
        history = {"train_rmse": [], "val_rmse": []}
        best_val = math.inf
        wait = 0
        best_state = None

        for epoch in range(1, epochs + 1):
            self.train()
            train_losses = []
            for cat_idx, num_vals, target in train_loader:
                cat_idx = cat_idx.to(self.device)
                num_vals = num_vals.to(self.device)
                target = target.to(self.device)

                # embed then MixUp on token space
                tokens = self._embed(cat_idx, num_vals)
                tokens, target = self._mixup(tokens, target)

                # forward through transformer & head (reuse forward logic)
                enc = self.transformer(tokens)
                weights = self.attn_pool(enc).squeeze(-1)
                pooled = (weights.unsqueeze(-1) * enc).sum(dim=1)
                pred = self.head(pooled).squeeze(-1)

                loss = criterion(pred, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                for cat_idx, num_vals, target in val_loader:
                    cat_idx = cat_idx.to(self.device)
                    num_vals = num_vals.to(self.device)
                    target = target.to(self.device)
                    pred = self.forward(cat_idx, num_vals)
                    val_losses.append(criterion(pred, target).item())

            train_rmse = math.sqrt(np.mean(train_losses))
            val_rmse = math.sqrt(np.mean(val_losses))
            history["train_rmse"].append(train_rmse)
            history["val_rmse"].append(val_rmse)
            scheduler.step()

            if verbose:
                print(f"Epoch {epoch:02d} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

            if val_rmse < best_val - 1e-4:
                best_val = val_rmse
                wait = 0
                best_state = self.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print("Early stopping triggered.")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return history

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str], batch_size: int = 1024) -> np.ndarray:
        """Point predictions for a DataFrame."""
        self.eval()
        if self.cat_maps is None or self.num_stats is None:
            raise RuntimeError("Model has not been fitted yet.")
        dataset = TabularDataset(df, cat_cols, num_cols, cat_maps=self.cat_maps, num_stats=self.num_stats)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        preds = []
        for cat_idx, num_vals in loader:
            cat_idx = cat_idx.to(self.device)
            num_vals = num_vals.to(self.device)
            out = self.forward(cat_idx, num_vals)
            preds.append(out.cpu().numpy())
        return np.concatenate(preds)

    @torch.no_grad()
    def predict_feature_importance(
        self,
        df: pd.DataFrame,
        cat_cols: List[str],
        num_cols: List[str],
        batch_size: int = 1024,
    ) -> pd.DataFrame:
        """Per‑sample attention weights for each token (feature)."""
        self.eval()
        if self.cat_maps is None or self.num_stats is None:
            raise RuntimeError("Model has not been fitted yet.")
        dataset = TabularDataset(df, cat_cols, num_cols, cat_maps=self.cat_maps, num_stats=self.num_stats)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_weights = []
        for cat_idx, num_vals in loader:
            cat_idx = cat_idx.to(self.device)
            num_vals = num_vals.to(self.device)
            tokens = self._embed(cat_idx, num_vals)
            enc = self.transformer(tokens)
            weights = self.attn_pool(enc).squeeze(-1)  # (B, N)
            all_weights.append(weights.cpu().numpy())
        weights_arr = np.concatenate(all_weights, axis=0)
        feature_names = cat_cols + num_cols
        return pd.DataFrame(weights_arr, columns=feature_names)


def tabular_token_transformer(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    cat_cols: Optional[List[str]] = None,
    num_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    dim_feedforward: int = 128,
    dropout: float = 0.1,
    mixup_alpha: float = 0.2,
    mixup_calibrate: bool = True,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """
    Entry‑point compatible with the platform registry.

    Returns
    -------
    np.ndarray
        Predicted continuous values for ``X_test``.
    """
    if y_train is None:
        raise ValueError("y_train must be provided for regression training.")

    # ------------------------------------------------------------------
    # Infer column types if not supplied
    # ------------------------------------------------------------------
    if cat_cols is None:
        cat_cols = [
            c
            for c in X_train.columns
            if X_train[c].dtype == "object" or str(X_train[c].dtype).startswith("category")
        ]
    if num_cols is None:
        num_cols = [c for c in X_train.columns if c not in cat_cols]

    # ------------------------------------------------------------------
    # Prepare training frame with a target column
    # ------------------------------------------------------------------
    if target_col is None:
        target_col = "__target__"
    X_train = X_train.copy()
    X_train[target_col] = np.asarray(y_train)

    # ------------------------------------------------------------------
    # Determine cardinalities for categorical columns
    # ------------------------------------------------------------------
    cat_cardinalities = [
        max(1, int(X_train[col].nunique(dropna=True))) for col in cat_cols
    ]
    num_features = len(num_cols)

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model = TabularTokenTransformerRegressor(
        cat_cardinalities=cat_cardinalities,
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        mixup_alpha=mixup_alpha,
        mixup_calibrate=mixup_calibrate,
        device=_default_device(),
    )

    # ------------------------------------------------------------------
    # Train / validation split (80/20)
    # ------------------------------------------------------------------
    train_df, val_df = train_test_split(X_train, test_size=0.2, random_state=42)

    epochs = min(
        int(epochs),
        int(os.getenv("AIBUILDAI_MAX_EPOCHS", epochs)),
    )
    patience = min(
        int(patience),
        int(os.getenv("AIBUILDAI_EARLY_STOPPING_PATIENCE", patience)),
    )
    model.fit(
        train_df=train_df,
        val_df=val_df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        target_col=target_col,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=verbose,
    )

    # ------------------------------------------------------------------
    # Predict on the provided test set
    # ------------------------------------------------------------------
    preds = model.predict(X_test, cat_cols=cat_cols, num_cols=num_cols, batch_size=batch_size)
    return preds
