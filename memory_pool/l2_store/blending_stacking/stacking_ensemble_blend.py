import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

def train_and_predict(X_train, y_train, X_test):
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    # define base models
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    base_models = [
        ("catboost", CatBoostClassifier(verbose=False, random_seed=42)),
        ("lgbm", LGBMClassifier(verbose=-1, random_state=42)),
        ("lr", LogisticRegression(max_iter=1000, random_state=42))
    ]
    n_folds = 5
    oof = np.zeros((X_train.shape[0], len(base_models)))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for i, (_, model) in enumerate(base_models):
        for train_idx, hold_idx in skf.split(X_train, y_train):
            X_tr, X_ho = X_train[train_idx], X_train[hold_idx]
            y_tr = y_train[train_idx]
            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            oof[hold_idx, i] = fold_model.predict_proba(X_ho)[:, 1]
        model.fit(X_train, y_train)
    meta = LogisticRegression(max_iter=1000, random_state=42)
    meta.fit(oof, y_train)
    base_probs = np.column_stack([m.predict_proba(X_test)[:, 1] for _, m in base_models])
    meta_prob = meta.predict_proba(base_probs)[:, 1]
    final = 0.5 * meta_prob + 0.5 * base_probs.mean(axis=1)
    return final
