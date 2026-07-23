import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_hist_gradient_boosting  
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from category_encoders import TargetEncoder


def _detect_categorical(df: pd.DataFrame, max_cardinality: int = 50) -> List[str]:
    """Return low‑cardinality object or categorical columns.

    Columns with > ``max_cardinality`` distinct values are ignored to keep the
    target encoder stable.
    """
    cat_cols = [
        c
        for c in df.columns
        if df[c].dtype == "object" or df[c].dtype.name == "category"
    ]
    return [c for c in cat_cols if df[c].nunique(dropna=True) <= max_cardinality]


def _add_interaction_features(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Add product and ratio features for the ``top_n`` most correlated numeric pairs."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return df

    corr = df[numeric_cols].corr().abs()
    np.fill_diagonal(corr.values, 0)
    pairs = (
        corr.unstack()
        .sort_values(ascending=False)
        .drop_duplicates()
        .reset_index()
        .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "corr"})
    )
    top_pairs = pairs.head(top_n)

    out = df.copy()
    for _, row in top_pairs.iterrows():
        a, b = row["col_a"], row["col_b"]
        out[f"{a}_x_{b}"] = out[a] * out[b]
        out[f"{a}_div_{b}"] = out[a] / (out[b] + 1e-9)
    return out


class ScreenHistGradientBoostingRegressor:
    """Encapsulates the full screen‑fidelity pipeline."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.preprocessor_: Optional[ColumnTransformer] = None
        self.model_: Optional[HistGradientBoostingRegressor] = None
        self.numeric_features_: List[str] = []
        self.oof_predictions_: Optional[np.ndarray] = None
        self.fold_rmse_: List[float] = []
        self._target_encoders: Dict[str, TargetEncoder] = {}
        self._cat_cols: List[str] = []

    # --------------------------------------------------------------------- #
    # Target encoding helpers
    # --------------------------------------------------------------------- #
    def _fit_target_encoders(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit a TargetEncoder for each categorical column and return an encoded copy."""
        X_enc = X.copy()
        for col in self._cat_cols:
            encoder = TargetEncoder(cols=[col], smoothing=1.0, min_samples_leaf=1)
            encoder.fit(X[[col]], y)
            self._target_encoders[col] = encoder
            X_enc[f"{col}_te"] = encoder.transform(X[[col]])[col]
        X_enc.drop(columns=self._cat_cols, inplace=True, errors="ignore")
        return X_enc

    def _apply_target_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply previously fitted encoders to a new DataFrame."""
        X_enc = X.copy()
        for col, encoder in self._target_encoders.items():
            if col not in X_enc.columns:
                raise KeyError(f"Expected column '{col}' for target encoding not found.")
            X_enc[f"{col}_te"] = encoder.transform(X[[col]])[col]
        X_enc.drop(columns=self._cat_cols, inplace=True, errors="ignore")
        return X_enc

    # --------------------------------------------------------------------- #
    # Core API
    # --------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the pipeline using a 2‑fold CV on the provided data."""
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index, name="target")

        # 1. Detect low‑cardinality categoricals and fill missing values.
        self._cat_cols = _detect_categorical(X)
        X = X.copy()
        if self._cat_cols:
            X[self._cat_cols] = X[self._cat_cols].fillna("missing")

        # 2. Target‑encode categoricals.
        X_enc = self._fit_target_encoders(X, y)

        # 3. Add interaction features.
        X_int = _add_interaction_features(X_enc, top_n=5)

        # 4. Identify numeric columns for scaling.
        self.numeric_features_ = X_int.select_dtypes(include=[np.number]).columns.tolist()

        # 5. Build preprocessing pipeline (median imputer + quantile transformer).
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "scaler",
                    QuantileTransformer(
                        output_distribution="normal", random_state=self.random_state
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, self.numeric_features_)],
            remainder="passthrough",
        )

        # 6. 2‑fold CV to collect OOF predictions.
        oof = np.zeros(len(X_int))
        indices = np.arange(len(X_int))
        for fold in [0, 1]:
            val_mask = indices % 2 == fold
            train_mask = ~val_mask
            train_pos = np.where(train_mask)[0]
            val_pos = np.where(val_mask)[0]

            X_tr = X_int.iloc[train_pos]
            X_val = X_int.iloc[val_pos]
            y_tr = y.iloc[train_pos]
            y_val = y.iloc[val_pos]

            preproc = preprocessor.fit(X_tr)
            X_tr_proc = preproc.transform(X_tr)
            X_val_proc = preproc.transform(X_val)

            model = HistGradientBoostingRegressor(
                max_depth=10,
                learning_rate=0.05,
                max_iter=500,
                max_bins=255,
                early_stopping=True,
                random_state=self.random_state,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )
            model.fit(X_tr_proc, y_tr)
            preds = model.predict(X_val_proc)
            oof[val_pos] = preds
            # Compute RMSE manually for compatibility with older sklearn versions.
            self.fold_rmse_.append(np.sqrt(mean_squared_error(y_val, preds)))

        self.oof_predictions_ = oof

        # 7. Fit preprocessing on the full data and retrain final model.
        self.preprocessor_ = preprocessor.fit(X_int)
        X_full = self.preprocessor_.transform(X_int)
        self.model_ = HistGradientBoostingRegressor(
            max_depth=10,
            learning_rate=0.05,
            max_iter=500,
            max_bins=255,
            early_stopping=True,
            random_state=self.random_state,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
        self.model_.fit(X_full, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for ``X``."""
        if self.preprocessor_ is None or self.model_ is None:
            raise RuntimeError("The model has not been fitted yet.")

        X = X.copy()

        # 1. Fill missing categoricals (same logic as during fit).
        if self._cat_cols:
            X[self._cat_cols] = X[self._cat_cols].fillna("missing")

        # 2. Apply stored target encoders.
        if self._cat_cols:
            X = self._apply_target_encoders(X)

        # 3. Add interaction features (same top_n as during training).
        X = _add_interaction_features(X, top_n=5)

        # 4. Transform with the fitted preprocessor and predict.
        X_proc = self.preprocessor_.transform(X)
        preds = self.model_.predict(X_proc)
        return np.clip(preds, a_min=0.0, a_max=None)


def run_screen_hist_gb(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    target_column: str = "loss",
) -> np.ndarray:
    """Fit the screen‑fidelity pipeline and return predictions for ``X_test``."""
    if y_train is None:
        if target_column not in X_train.columns:
            raise ValueError(f"Target column '{target_column}' not found in X_train.")
        y = X_train[target_column]
        X = X_train.drop(columns=[target_column])
    else:
        if len(y_train) != len(X_train):
            raise ValueError("Length of y_train does not match X_train.")
        y = y_train
        X = X_train

    model = ScreenHistGradientBoostingRegressor(random_state=42)
    model.fit(X, y)
    return model.predict(X_test)
