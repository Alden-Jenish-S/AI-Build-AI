from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class InteractionFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, k_interactions: int = 50, random_state: Optional[int] = None):
        self.k_interactions = k_interactions
        self.random_state = random_state
        self._poly: Optional[PolynomialFeatures] = None
        self._idx_selected: Optional[np.ndarray] = None
        self._n_features_in: Optional[int] = None

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self._n_features_in = X.shape[1]

        self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_inter = self._poly.fit_transform(X)

        mi = mutual_info_classif(X_inter, y, discrete_features=False, random_state=self.random_state)
        interaction_mi = mi[self._n_features_in :]
        n_interactions = interaction_mi.shape[0]
        k = min(self.k_interactions, n_interactions)

        top_k_rel = np.argsort(interaction_mi)[::-1][:k]
        self._idx_selected = np.concatenate(
            [np.arange(self._n_features_in), self._n_features_in + top_k_rel]
        )
        return self

    def transform(self, X):
        check_is_fitted(self, ["_poly", "_idx_selected", "_n_features_in"])
        X = check_array(X, dtype=np.float64)
        X_inter = self._poly.transform(X)
        return X_inter[:, self._idx_selected]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ["_poly", "_idx_selected", "_n_features_in"])
        if input_features is None:
            input_features = [f"x{i}" for i in range(self._n_features_in)]
        all_names = self._poly.get_feature_names_out(input_features)
        return all_names[self._idx_selected]


def fit_predict_interaction_gbm(
    X_train,
    y_train,
    X_test,
    *,
    k_interactions: int = 50,
    gbm_random_state: int = 42,
) -> Tuple[np.ndarray, InteractionFeatureEngineer]:
    engineer = InteractionFeatureEngineer(k_interactions=k_interactions, random_state=gbm_random_state)
    engineer.fit(X_train, y_train)
    X_train_eng = engineer.transform(X_train)
    X_test_eng = engineer.transform(X_test)

    gbm = HistGradientBoostingClassifier(random_state=gbm_random_state)
    gbm.fit(X_train_eng, y_train)

    prob_test = gbm.predict_proba(X_test_eng)[:, 1]
    return prob_test, engineer