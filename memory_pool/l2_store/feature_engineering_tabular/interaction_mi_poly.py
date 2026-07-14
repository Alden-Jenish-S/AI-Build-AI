from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class InteractionMIPoly(BaseEstimator, ClassifierMixin):
    '''
    Create interaction features based on Mutual Information and fit a
    HistGradientBoostingClassifier.

    Parameters
    ----------
    top_k : int, default=30
        Number of highest‑MI feature pairs to turn into interaction terms.
    random_state : int | None, default=None
        Seed for reproducibility (passed to the underlying classifier).
    **hgb_kwargs :
        Additional keyword arguments forwarded to HistGradientBoostingClassifier.
    '''

    def __init__(
        self,
        top_k: int = 30,
        random_state: Optional[int] = None,
        **hgb_kwargs,
    ):
        self.top_k = top_k
        self.random_state = random_state
        self.hgb_kwargs = hgb_kwargs

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _compute_mi_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        '''MI between each feature and the target.'''
        return mutual_info_classif(X, y, discrete_features='auto', random_state=self.random_state)

    def _select_top_pairs(self, X: np.ndarray, mi_scores: np.ndarray) -> List[Tuple[int, int]]:
        '''
        Choose the top_k feature pairs with the highest combined MI score.
        Pair score = mi_i + mi_j (simple additive heuristic).
        '''
        n_features = X.shape[1]
        if n_features < 2 or self.top_k <= 0:
            return []
        rows, cols = np.triu_indices(n_features, k=1)
        scores = mi_scores[rows] + mi_scores[cols]
        k = min(self.top_k, len(scores))
        top_indices = np.argsort(scores, kind="stable")[-k:][::-1]
        return [(int(rows[index]), int(cols[index])) for index in top_indices]

    def _build_interaction_matrix(self, X: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
        '''Append product interaction columns to X.'''
        if not pairs:
            return X
        interaction_cols = []
        for i, j in pairs:
            interaction_cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if interaction_cols:
            return np.hstack([X] + interaction_cols)
        return X

    # ------------------------------------------------------------------ #
    # Scikit‑learn API
    # ------------------------------------------------------------------ #
    def fit(self, X, y):
        '''
        Fit the interaction generator and the downstream classifier.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)
            Training data.
        y : array‑like of shape (n_samples,)
            Binary target.

        Returns
        -------
        self
        '''
        input_feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        # Validate input
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        self._n_features_in = X.shape[1]

        # 1️⃣ Compute MI scores
        mi_scores = self._compute_mi_scores(X, y)

        # 2️⃣ Select top‑k pairs
        self.top_pairs_ = self._select_top_pairs(X, mi_scores)

        # 3️⃣ Build interaction‑augmented training matrix
        X_aug = self._build_interaction_matrix(X, self.top_pairs_)

        # 4️⃣ Fit classifier
        self.clf_ = HistGradientBoostingClassifier(
            random_state=self.random_state, **self.hgb_kwargs
        )
        self.clf_.fit(X_aug, y)

        # Store feature names for convenience (if X is a DataFrame)
        if input_feature_names is not None:
            base_names = input_feature_names
            inter_names = [f'{base_names[i]}*{base_names[j]}' for i, j in self.top_pairs_]
            self.feature_names_in_ = base_names + inter_names
        else:
            self.feature_names_in_ = None

        return self

    def predict_proba(self, X):
        '''
        Predict class probabilities.

        Parameters
        ----------
        X : array‑like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        '''
        check_is_fitted(self, ['clf_', 'top_pairs_'])
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        X_aug = self._build_interaction_matrix(X, self.top_pairs_)
        return self.clf_.predict_proba(X_aug)

    def predict(self, X):
        '''Predict class labels.'''
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def get_feature_names_out(self, input_features=None):
        '''Output feature names after interaction augmentation.'''
        if self.feature_names_in_ is not None:
            return np.array(self.feature_names_in_, dtype=object)
        # fallback: generic names
        n_base = self.n_features_in_
        n_inter = len(self.top_pairs_)
        base = [f'x{i}' for i in range(n_base)]
        inter = [f'x{i}*x{j}' for (i, j) in self.top_pairs_]
        return np.array(base + inter, dtype=object)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def n_features_in_(self):
        '''Number of original features.'''
        return getattr(self, '_n_features_in', None)

    def _more_tags(self):
        return {'binary_only': True, 'requires_y': True}


# ---------------------------------------------------------------------- #
# Convenience factory function (interface entrypoint)
# ---------------------------------------------------------------------- #
def train_interaction_mi_poly(
    X_train,
    y_train,
    top_k: int = 30,
    random_state: Optional[int] = None,
    **hgb_kwargs,
):
    '''
    Train an InteractionMIPoly model and return the fitted estimator.

    Parameters
    ----------
    X_train : array‑like or DataFrame
        Training features.
    y_train : array‑like
        Binary target.
    top_k : int, default=30
        Number of interaction pairs to generate.
    random_state : int | None, default=None
        Random seed.
    **hgb_kwargs :
        Passed to HistGradientBoostingClassifier (e.g., max_iter, learning_rate).

    Returns
    -------
    model : InteractionMIPoly
        Fitted model ready for ``predict_proba``.
    '''
    model = InteractionMIPoly(top_k=top_k, random_state=random_state, **hgb_kwargs)
    model.fit(X_train, y_train)
    return model
