import numpy as np
import pandas as pd
import os
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _resolve_torch_device(device=None):
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
    if requested not in {"cpu", "auto"}:
        warnings.warn(
            f"Requested PyTorch accelerator {requested!r} is unavailable; using CPU.",
            RuntimeWarning,
        )
    os.environ["AIBUILDAI_ACTUAL_ACCELERATOR"] = "cpu"
    return "cpu"

# ------------------------------------------------------------------
# 1. Gradient Reversal Layer (autograd Function)
# ------------------------------------------------------------------
class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversal(nn.Module):
    """Layer that multiplies the incoming gradient by -lambda_."""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return _GradientReversalFunction.apply(x, self.lambda_)

# ------------------------------------------------------------------
# 2. Tabular Dataset wrapper
# ------------------------------------------------------------------
class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y, domain):
        self.X_num = torch.from_numpy(X_num).float()
        self.X_cat = torch.from_numpy(X_cat).long()
        self.y = torch.from_numpy(y).float()
        self.domain = torch.from_numpy(domain).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.X_cat[idx], self.y[idx], self.domain[idx])

# ------------------------------------------------------------------
# 3. Neural net with GRL
# ------------------------------------------------------------------
class _DomainAdaptationNet(nn.Module):
    def __init__(self,
                 num_numeric,
                 cat_cardinalities,
                 embed_dim=16,
                 hidden_dims=(128, 64),
                 dropout=0.2):
        super().__init__()

        # embeddings for each categorical column
        self.embeds = nn.ModuleList([
            nn.Embedding(card, min(embed_dim, (card + 1) // 2))
            for card in cat_cardinalities
        ])
        embed_out = sum(e.embedding_dim for e in self.embeds)

        # shared feature extractor
        layers = []
        in_dim = num_numeric + embed_out
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        # task head (binary classification)
        self.task_head = nn.Linear(in_dim, 1)

        # domain head (2‑class discriminator)
        self.grl = GradientReversal(lambda_=1.0)  # lambda updated each epoch
        self.domain_head = nn.Linear(in_dim, 2)

    def forward(self, x_num, x_cat):
        if len(self.embeds) > 0:
            embed_vecs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeds)]
            x_cat_emb = torch.cat(embed_vecs, dim=1)
            x = torch.cat([x_num, x_cat_emb], dim=1)
        else:
            x = x_num
        feat = self.feature_extractor(x)
        task_logit = self.task_head(feat).squeeze(1)
        rev_feat = self.grl(feat)
        domain_logit = self.domain_head(rev_feat)
        return task_logit, domain_logit

# ------------------------------------------------------------------
# 4. Estimator (scikit‑learn compatible)
# ------------------------------------------------------------------
class GradientReversalDomainAdaptation(BaseEstimator, ClassifierMixin):
    """Adversarial domain‑adaptation classifier for tabular data.

    Parameters are deliberately kept lightweight; most are passed
    directly to the underlying neural network or training loop.
    """

    def __init__(self,
                 numeric_features=None,
                 categorical_features=None,
                 embed_dim=16,
                 hidden_dims=(128, 64),
                 dropout=0.2,
                 batch_size=256,
                 epochs=30,
                 lr=1e-3,
                 lambda_max=1.0,
                 lambda_schedule='linear',
                 device=None,
                 random_state=42,
                 verbose=1):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lambda_max = lambda_max
        self.lambda_schedule = lambda_schedule
        self.device = _resolve_torch_device(device)
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------
    # 4.1 preprocessing helpers
    # ------------------------------------------------------------------
    def _prepare_columns(self, X):
        if self.numeric_features is None:
            self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_features_ = list(self.numeric_features)
        if self.categorical_features is None:
            self.categorical_features_ = X.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            self.categorical_features_ = list(self.categorical_features)
        self.numeric_features_.sort()
        self.categorical_features_.sort()

    def _fit_preprocessors(self, X):
        self.num_scaler_ = StandardScaler()
        self.num_scaler_.fit(X[self.numeric_features_])
        self.cat_encoder_ = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.cat_encoder_.fit(X[self.categorical_features_])
        self.cat_cardinalities_ = [int(self.cat_encoder_.categories_[i].shape[0]) + 1
                                   for i in range(len(self.categorical_features_))]

    def _transform(self, X):
        X_num = self.num_scaler_.transform(X[self.numeric_features_])
        X_cat = self.cat_encoder_.transform(X[self.categorical_features_])
        for i, card in enumerate(self.cat_cardinalities_):
            X_cat[:, i] = np.where(X_cat[:, i] == -1, card - 1, X_cat[:, i])
        return X_num.astype(np.float32), X_cat.astype(np.int64)

    # ------------------------------------------------------------------
    # 4.2 fit
    # ------------------------------------------------------------------
    def fit(self, X, y, aux_X=None, aux_y=None):
        """Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            Main training data (target domain).
        y : array‑like
            Binary labels for ``X``.
        aux_X : pd.DataFrame, optional
            Auxiliary data from a different domain (same columns as ``X``).
        aux_y : array‑like, optional
            Labels for ``aux_X`` – ignored for training but kept for API
            compatibility.
        """
        # column handling & preprocessing
        self._prepare_columns(X)
        combined = pd.concat([X, aux_X]) if aux_X is not None else X
        self._fit_preprocessors(combined)
        X_num, X_cat = self._transform(X)
        y = np.asarray(y).astype(np.float32)

        # build auxiliary tensors if provided
        if aux_X is not None:
            aux_num, aux_cat = self._transform(aux_X)
            aux_y = np.zeros_like(aux_num[:, 0]) if aux_y is None else np.asarray(aux_y).astype(np.float32)
            domain_main = np.ones(len(y), dtype=np.int64)
            domain_aux = np.zeros(len(aux_num), dtype=np.int64)
            X_num_all = np.concatenate([aux_num, X_num], axis=0)
            X_cat_all = np.concatenate([aux_cat, X_cat], axis=0)
            y_all = np.concatenate([aux_y, y], axis=0)
            domain_all = np.concatenate([domain_aux, domain_main], axis=0)
        else:
            X_num_all, X_cat_all, y_all = X_num, X_cat, y
            domain_all = np.ones(len(y), dtype=np.int64)

        dataset = TabularDataset(X_num_all, X_cat_all, y_all, domain_all)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.device == "cuda",
        )

        torch.manual_seed(self.random_state)
        self.model_ = _DomainAdaptationNet(
            num_numeric=len(self.numeric_features_),
            cat_cardinalities=self.cat_cardinalities_,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            epoch_losses = []
            # schedule λ
            if self.lambda_schedule == 'linear':
                lam = self.lambda_max * epoch / self.epochs
            elif self.lambda_schedule == 'exp':
                lam = self.lambda_max * (2.0 ** (epoch - self.epochs) - 1)
            else:
                lam = self.lambda_max
            self.model_.grl.lambda_ = lam

            for xb_num, xb_cat, yb, db in loader:
                xb_num = xb_num.to(self.device, non_blocking=self.device == "cuda")
                xb_cat = xb_cat.to(self.device, non_blocking=self.device == "cuda")
                yb = yb.to(self.device, non_blocking=self.device == "cuda")
                db = db.to(self.device, non_blocking=self.device == "cuda")

                optimizer.zero_grad()
                task_logit, domain_logit = self.model_(xb_num, xb_cat)
                task_loss = F.binary_cross_entropy_with_logits(task_logit, yb)
                domain_loss = F.cross_entropy(domain_logit, db)
                loss = task_loss + domain_loss
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            if self.verbose:
                print(f"Epoch {epoch:02d}/{self.epochs} | λ={lam:.3f} | loss={np.mean(epoch_losses):.4f}")
        return self

    # ------------------------------------------------------------------
    # 4.3 prediction helpers
    # ------------------------------------------------------------------
    def _predict_proba_internal(self, X):
        self.model_.eval()
        X_num, X_cat = self._transform(X)
        tensor_num = torch.from_numpy(X_num).float().to(self.device)
        tensor_cat = torch.from_numpy(X_cat).long().to(self.device)
        with torch.no_grad():
            logits, _ = self.model_(tensor_num, tensor_cat)
            probs = torch.sigmoid(logits).cpu().numpy()
        probs = probs.reshape(-1)
        return np.column_stack([1.0 - probs, probs])

    def predict_proba(self, X):
        """Return probability of class 0 and 1 (shape N×2)."""
        probs_pos = self._predict_proba_internal(X)[:, 1]
        probs = np.vstack([1 - probs_pos, probs_pos]).T
        return probs

    def predict(self, X):
        """Binary predictions (0/1) using 0.5 threshold."""
        probs_pos = self._predict_proba_internal(X)[:, 1]
        return (probs_pos >= 0.5).astype(int)

    # ------------------------------------------------------------------
    # 4.4 optional AUC evaluation helper
    # ------------------------------------------------------------------
    def evaluate_auc(self, X, y, test_size=0.2, random_state=None):
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y,
            random_state=random_state or self.random_state
        )
        self.fit(X_tr, y_tr)
        val_pred = self._predict_proba_internal(X_val)[:, 1]
        return roc_auc_score(y_val, val_pred)

# ------------------------------------------------------------------
# 5. Public entrypoint (wrapper style)
# ------------------------------------------------------------------
def train_predict(X_train, X_test, y_train=None, aux_X=None, **model_kwargs):
    """Fit ``GradientReversalDomainAdaptation`` on the supplied data and return
    probability predictions for the positive class.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (target domain).
    X_test : pd.DataFrame
        Test features for which predictions are required.
    y_train : array‑like, optional
        Binary labels for ``X_train``. Required for training.
    aux_X : pd.DataFrame, optional
        Auxiliary dataset from a different domain (same columns as ``X_train``).
    model_kwargs : dict
        Additional keyword arguments forwarded to the estimator.

    Returns
    -------
    np.ndarray
        1‑D array of positive‑class probabilities for ``X_test``.
    """
    if y_train is None:
        raise ValueError("y_train must be provided for training.")
    model = GradientReversalDomainAdaptation(**model_kwargs)
    model.fit(X_train, y_train, aux_X=aux_X)
    probs = model._predict_proba_internal(X_test)[:, 1]
    return probs
