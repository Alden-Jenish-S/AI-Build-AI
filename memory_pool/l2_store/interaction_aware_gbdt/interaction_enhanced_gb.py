from __future__ import annotations

import numpy as np
from typing import List, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class InteractionEnhancedGB(BaseEstimator, ClassifierMixin):
    """
    Gradient Boosting model augmented with automatically selected
    pairwise interaction features.

    Parameters
    ----------
    top_k_interactions : int, default=20
        Number of feature pairs to turn into interaction columns.
    **hgb_kwargs :
        Additional keyword arguments passed to
        `HistGradientBoostingClassifier`.
    """

    def __init__(
        self,
        top_k_interactions: int = 20,
        **hgb_kwargs,
    ):
        self.top_k_interactions = top_k_interactions
        self.hgb_kwargs = hgb_kwargs
        self._hgb: HistGradientBoostingClassifier | None = None
        self._interaction_pairs: List[Tuple[int, int]] | None = None
        self._n_features_in_: int | None = None

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _compute_mi_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Mutual information of each feature with the target."""
        return mutual_info_classif(X, y, discrete_features="auto", random_state=0)

    def _select_top_pairs(self, mi_scores: np.ndarray) -> List[Tuple[int, int]]:
        """
        Rank all unordered feature pairs by the sum of their MI scores
        and return the top K pairs.
        """
        n_features = mi_scores.shape[0]
        if n_features < 2 or self.top_k_interactions <= 0:
            return []
        rows, cols = np.triu_indices(n_features, k=1)
        scores = mi_scores[rows] + mi_scores[cols]
        k = min(self.top_k_interactions, len(scores))
        top_indices = np.argsort(scores, kind="stable")[-k:][::-1]
        return [(int(rows[index]), int(cols[index])) for index in top_indices]

    def _build_interaction_matrix(self, X: np.ndarray) -> np.ndarray:
        """Append selected interaction columns to X."""
        if self._interaction_pairs is None:
            return X
        interaction_cols = [
            X[:, i] * X[:, j] for (i, j) in self._interaction_pairs
        ]
        if interaction_cols:
            interaction_mat = np.column_stack(interaction_cols)
            return np.hstack([X, interaction_mat])
        return X

    # ------------------------------------------------------------------ #
    # Scikit‑learn API
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        """
        Fit the interaction‑enhanced gradient boosting model.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training data.
        y : array‑like of shape (n_samples,)
            Binary target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64, accept_sparse=False)
        self._n_features_in_ = X.shape[1]

        # 1. MI screening
        mi_scores = self._compute_mi_scores(X, y)

        # 2. Select top‑K interaction pairs
        self._interaction_pairs = self._select_top_pairs(mi_scores)

        # 3. Build augmented training matrix
        X_aug = self._build_interaction_matrix(X)

        # 4. Fit HistGradientBoostingClassifier
        self._hgb = HistGradientBoostingClassifier(**self.hgb_kwargs)
        self._hgb.fit(X_aug, y)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)

        Returns
        -------
        probas : ndarray of shape (n_samples, 2)
            Probability of each class. The second column corresponds to
            the positive class (target == 1).
        """
        check_is_fitted(self, ["_hgb", "_interaction_pairs", "_n_features_in_"])
        X = check_array(X, dtype=np.float64, accept_sparse=False)
        if X.shape[1] != self._n_features_in_:
            raise ValueError(
                f"Number of features {X.shape[1]} does not match "
                f"the number seen during fit ({self._n_features_in_})."
            )
        X_aug = self._build_interaction_matrix(X)
        return self._hgb.predict_proba(X_aug)

    def predict(self, X):
        """Predict class labels (0 or 1)."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # Optional: expose feature names for debugging
    def get_feature_names_out(self, input_features=None):
        """Return feature names after interaction augmentation."""
        check_is_fitted(self, ["_n_features_in_", "_interaction_pairs"])
        if input_features is None:
            input_features = [f"f{i}" for i in range(self._n_features_in_)]
        else:
            if len(input_features) != self._n_features_in_:
                raise ValueError(
                    f"Length of input_features ({len(input_features)}) "
                    f"does not match n_features ({self._n_features_in_})."
                )
        names = list(input_features)
        if self._interaction_pairs:
            for i, j in self._interaction_pairs:
                names.append(f"{input_features[i]}*{input_features[j]}")
        return np.array(names)


# ---------------------------------------------------------------------- #
# Entrypoint for the builder
# ---------------------------------------------------------------------- #
def build_model(X_train, X_test, y_train=None, **kwargs):
    """
    Build and fit an InteractionEnhancedGB model.

    Parameters
    ----------
    X_train : array-like
        Training features.
    X_test : array-like
        Test features (used to generate predictions if y_train is provided).
    y_train : array-like, optional
        Training labels. If None, returns an unfitted model instance.
    **kwargs :
        Passed to InteractionEnhancedGB constructor (e.g., top_k_interactions,
        learning_rate, max_iter, etc.).

    Returns
    -------
    If y_train is provided: ndarray of shape (n_test_samples,) containing
        predicted probabilities for the positive class.
    Else: unfitted InteractionEnhancedGB instance.
    """
    model = InteractionEnhancedGB(**kwargs)
    if y_train is not None:
        model.fit(X_train, y_train)
        # Return probability of the positive class for the test set
        return model.predict_proba(X_test)[:, 1]
    else:
        return model
