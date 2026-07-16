import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold, KFold


class CVTargetEncoder(BaseEstimator, TransformerMixin):
    """Cross‑validated, Bayesian‑smoothed target encoder.

    Parameters
    ----------
    cat_cols : List[str]
        Categorical columns to encode.
    target_col : str
        Name of the target column in the training DataFrame.
    n_splits : int, default 5
        Number of folds for out‑of‑fold encoding.
    smoothing : float, default 1.0
        Smoothing effect. Higher = more weight to the global mean.
    random_state : Optional[int], default None
        Random seed for reproducibility.
    stratify : bool, default True
        Whether to use stratified folds (only works for classification).
    dtype : Union[str, np.dtype], default np.float32
        Data type of the produced encoded columns.
    """

    def __init__(
        self,
        cat_cols: List[str],
        target_col: str,
        n_splits: int = 5,
        smoothing: float = 1.0,
        random_state: Optional[int] = None,
        stratify: bool = True,
        dtype: Union[str, np.dtype] = np.float32,
    ):
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state
        self.stratify = stratify
        self.dtype = dtype
        self.global_mean_: float = np.nan
        self.category_stats_: dict = {}
        self._fitted = False

    @staticmethod
    def _smooth(mean: np.ndarray, count: np.ndarray, global_mean: float, smoothing: float) -> np.ndarray:
        """Bayesian smoothing formula.
        """
        return (mean * count + global_mean * smoothing) / (count + smoothing)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CVTargetEncoder":
        """Fit the encoder on the full training set.
        Parameters
        ----------
        X : pd.DataFrame
            Training features **including** the target column.
        y : ignored, kept for scikit‑learn compatibility.
        """
        if self.target_col not in X.columns:
            raise ValueError(f"target_col '{self.target_col}' not found in X")
        self.global_mean_ = X[self.target_col].mean()
        self.category_stats_ = {}
        for col in self.cat_cols:
            stats = (
                X.groupby(col)[self.target_col]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "cat_mean", "count": "cat_count"})
            )
            self.category_stats_[col] = stats
        self._fitted = True
        return self

    def _encode_column(
        self,
        series: pd.Series,
        stats_df: pd.DataFrame,
        global_mean: float,
        smoothing: float,
    ) -> pd.Series:
        """Map each value in ``series`` to its smoothed target encoding.
        Unknown categories receive the global mean.
        """
        merged = series.to_frame(name="cat").merge(
            stats_df,
            left_on="cat",
            right_on=stats_df.columns[0],
            how="left",
        )
        merged["cat_mean"].fillna(global_mean, inplace=True)
        merged["cat_count"].fillna(0, inplace=True)
        enc = self._smooth(
            merged["cat_mean"].values,
            merged["cat_count"].values,
            global_mean,
            smoothing,
        )
        return pd.Series(enc, index=series.index, dtype=self.dtype)

    def transform(self, X: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Encode ``X``.
        If ``is_train=True`` performs OOF encoding; otherwise uses the full‑data mapping.
        """
        if not self._fitted:
            raise RuntimeError("CVTargetEncoder must be fitted before calling transform().")
        X_enc = X.copy()
        if is_train:
            # Out‑of‑fold encoding
            if self.stratify:
                cv = StratifiedKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                splits = cv.split(X_enc, X_enc[self.target_col])
            else:
                cv = KFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.random_state,
                )
                splits = cv.split(X_enc)
            for col in self.cat_cols:
                X_enc[f"{col}_te"] = np.nan
            for train_idx, val_idx in splits:
                train_fold = X_enc.iloc[train_idx]
                val_fold = X_enc.iloc[val_idx]
                for col in self.cat_cols:
                    stats = (
                        train_fold.groupby(col)[self.target_col]
                        .agg(["mean", "count"])
                        .reset_index()
                        .rename(columns={"mean": "cat_mean", "count": "cat_count"})
                    )
                    enc_series = self._encode_column(
                        val_fold[col],
                        stats,
                        global_mean=self.global_mean_,
                        smoothing=self.smoothing,
                    )
                    X_enc.loc[val_idx, f"{col}_te"] = enc_series
            X_enc.drop(columns=self.cat_cols, inplace=True)
        else:
            # Full‑data mapping for test/hold‑out
            for col in self.cat_cols:
                stats = self.category_stats_[col]
                X_enc[f"{col}_te"] = self._encode_column(
                    X_enc[col],
                    stats,
                    global_mean=self.global_mean_,
                    smoothing=self.smoothing,
                )
            X_enc.drop(columns=self.cat_cols, inplace=True)
        # Ensure dtype consistency
        for col in X_enc.columns:
            if X_enc[col].dtype != self.dtype:
                X_enc[col] = X_enc[col].astype(self.dtype, copy=False)
        return X_enc

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        is_train: bool = True,
    ) -> pd.DataFrame:
        """Fit on ``X`` and immediately transform it.
        """
        self.fit(X, y)
        return self.transform(X, is_train=is_train)


def apply_cv_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    cat_cols: List[str],
    n_splits: int = 5,
    smoothing: float = 1.0,
    random_state: Optional[int] = None,
    stratify: bool = True,
    dtype: Union[str, np.dtype] = np.float32,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """High‑level helper that fits a CVTargetEncoder on the training set and
    returns OOF‑encoded training data together with fully‑encoded test data.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Training and test tables (train_df must contain ``target_col``).
    target_col : str
        Name of the label column.
    cat_cols : List[str]
        Categorical columns to encode.
    n_splits, smoothing, random_state, stratify, dtype : see ``CVTargetEncoder``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        ``train_encoded`` – training data with encoded columns (original cat cols removed).
        ``test_encoded`` – test data with the same encoded columns.
    """
    encoder = CVTargetEncoder(
        cat_cols=cat_cols,
        target_col=target_col,
        n_splits=n_splits,
        smoothing=smoothing,
        random_state=random_state,
        stratify=stratify,
        dtype=dtype,
    )
    train_enc = encoder.fit_transform(train_df, is_train=True)
    test_enc = encoder.transform(test_df, is_train=False)
    return train_enc, test_enc