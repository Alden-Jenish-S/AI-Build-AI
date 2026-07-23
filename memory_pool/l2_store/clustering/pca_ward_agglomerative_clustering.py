import pathlib
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


def _detect_categorical(df: pd.DataFrame, max_cardinality: int = 10) -> List[str]:
    """Return column names that are categorical (object dtype or low‑cardinality integer)."""
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            cat_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col].dtype):
            if df[col].nunique() <= max_cardinality:
                cat_cols.append(col)
    return cat_cols


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """One‑hot encode categoricals, standard‑scale, then apply PCA."""

    def __init__(self, pca_variance: float = 0.95, max_pca_components: int = 30):
        self.pca_variance = pca_variance
        self.max_pca_components = max_pca_components
        self.categorical_: List[str] = []
        self.scaler_: StandardScaler = StandardScaler()
        self.pca_: PCA = None
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        # Detect categoricals
        self.categorical_ = _detect_categorical(X)

        # One‑hot encode
        X_enc = pd.get_dummies(
            X,
            columns=self.categorical_,
            drop_first=False,
            dtype=np.float32,
        )

        # Fill any missing numeric values with 0 for stability
        X_enc = X_enc.fillna(0.0)

        # Scale
        self.scaler_.fit(X_enc.values)

        # PCA – fit with the maximum allowed components; later the
        # transformer will keep the same number of components.
        X_scaled = self.scaler_.transform(X_enc.values)
        n_components = min(self.max_pca_components, X_scaled.shape[1])
        self.pca_ = PCA(
            n_components=n_components,
            svd_solver="full",
            random_state=42,
        )
        self.pca_.fit(X_scaled)

        # Store column order for deterministic transforms
        self.feature_names_ = list(X_enc.columns)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        # One‑hot using the same columns discovered during fit
        X_enc = pd.get_dummies(
            X,
            columns=self.categorical_,
            drop_first=False,
            dtype=np.float32,
        )
        # Align to training layout: add missing columns, drop unseen ones,
        # and fill any NaNs with 0.
        X_enc = X_enc.reindex(columns=self.feature_names_, fill_value=0.0)
        X_enc = X_enc.fillna(0.0)

        # Scale
        X_scaled = self.scaler_.transform(X_enc.values)

        # PCA projection
        return self.pca_.transform(X_scaled)


class PCAWardClustering:
    """Full unsupervised pipeline: preprocessing → silhouette‑based k search → Ward clustering."""

    def __init__(self, k_range: Tuple[int, int] = (2, 15)):
        self.k_range = k_range
        self.preprocessor = TabularPreprocessor()
        self.best_k_: int = None
        self.final_model_: AgglomerativeClustering = None

    def fit(self, train_df: pd.DataFrame):
        X_train = train_df.copy()
        # Preprocess
        self.preprocessor.fit(X_train)
        X_emb = self.preprocessor.transform(X_train)

        # Silhouette search
        best_score = -np.inf
        best_k = None
        for k in range(self.k_range[0], self.k_range[1] + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
            )
            labels = model.fit_predict(X_emb)
            if len(set(labels)) > 1:
                score = silhouette_score(X_emb, labels)
                if score > best_score:
                    best_score = score
                    best_k = k
        self.best_k_ = best_k if best_k is not None else self.k_range[0]
        return self

    def predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
        X_combined = self.preprocessor.transform(combined)

        self.final_model_ = AgglomerativeClustering(
            n_clusters=self.best_k_,
            linkage="ward",
        )
        combined_labels = self.final_model_.fit_predict(X_combined)

        # Relabel by descending cluster size for deterministic ordering
        unique, counts = np.unique(combined_labels, return_counts=True)
        order = np.argsort(-counts)
        mapping = {unique[idx]: new_id for new_id, idx in enumerate(order)}
        relabeled = np.vectorize(mapping.get)(combined_labels)

        return relabeled[-len(test_df) :]

    def _sanitize_id_column(self, id_column: Union[str, List[str]]) -> str:
        """Ensure the identifier column name is a hashable string."""
        if isinstance(id_column, (list, tuple, np.ndarray)):
            if len(id_column) == 0:
                raise ValueError("id_column list/tuple cannot be empty.")
            return str(id_column[0])
        return str(id_column)

    def run_and_save(
        self,
        train_path: Union[str, pathlib.Path, pd.DataFrame],
        test_path: Union[str, pathlib.Path, pd.DataFrame],
        submission_path: Union[str, pathlib.Path, None] = None,
        id_column: Union[str, List[str]] = "Id",
    ) -> pd.DataFrame:
        """
        Execute the pipeline and return a submission DataFrame.

        Parameters
        ----------
        train_path, test_path :
            Either file paths (str/Path) to CSV files or already‑loaded
            ``pd.DataFrame`` objects.
        submission_path :
            Optional path where the caller would like to persist the CSV.
            If ``None`` (default) the DataFrame is **not** written to disk.
        id_column :
            Name of the identifier column present in the test set.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ``[id_column, "Predicted"]`` ready for
            submission.
        """
        # Normalise id_column to a hashable string
        id_column = self._sanitize_id_column(id_column)

        # Load or forward the inputs
        if isinstance(train_path, pd.DataFrame):
            train_df = train_path
        else:
            train_df = pd.read_csv(train_path)

        if isinstance(test_path, pd.DataFrame):
            test_df = test_path
        else:
            test_df = pd.read_csv(test_path)

        if id_column not in test_df.columns:
            raise KeyError(f"Identifier column '{id_column}' not found in test data.")

        # Fit on training data and predict on test data
        self.fit(train_df)
        test_pred = self.predict(train_df, test_df)

        submission = pd.DataFrame(
            {
                id_column: test_df[id_column].values,
                "Predicted": test_pred.astype(int),
            }
        )

        # Optional persistence – guarded against side‑effects in sandboxed env.
        if submission_path is not None:
            pathlib.Path(submission_path).parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(submission_path, index=False)

        return submission


def run_pca_ward_clustering(
    train_csv: Union[str, pd.DataFrame],
    test_csv: Union[str, pd.DataFrame],
    submission_csv: Union[str, None] = None,
    id_column: Union[str, List[str]] = "Id",
) -> pd.DataFrame:
    """
    Public entry‑point that executes the full clustering pipeline.

    Parameters
    ----------
    train_csv, test_csv :
        Paths to CSV files **or** ``pd.DataFrame`` objects containing the data.
    submission_csv :
        Optional destination path for the ``Id,Predicted`` CSV.  If ``None``,
        the function simply returns the DataFrame.
    id_column :
        Name of the identifier column (default ``"Id"``).

    Returns
    -------
    pd.DataFrame
        The submission DataFrame.
    """
    pipeline = PCAWardClustering(k_range=(2, 15))
    return pipeline.run_and_save(
        train_path=train_csv,
        test_path=test_csv,
        submission_path=submission_csv,
        id_column=id_column,
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run PCA + Ward Agglomerative clustering on mixed‑type tabular data."
    )
    parser.add_argument("train_csv", help="Path to training CSV")
    parser.add_argument("test_csv", help="Path to test CSV")
    parser.add_argument(
        "submission_csv",
        nargs="?",
        default=None,
        help="Optional path where the submission CSV will be saved",
    )
    parser.add_argument(
        "--id_column",
        default="Id",
        help="Name of the identifier column (default: Id)",
    )
    args = parser.parse_args()
    try:
        run_pca_ward_clustering(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            submission_csv=args.submission_csv,
            id_column=args.id_column,
        )
    except Exception as exc:
        print(f"❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)
