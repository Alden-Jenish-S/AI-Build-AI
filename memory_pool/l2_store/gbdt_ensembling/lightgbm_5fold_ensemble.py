import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold

def fit_predict(X_train, y_train, X_test, cat_features=None, n_folds=5, is_classification=True):
    """
    Train a 5-fold LightGBM model on tabular features.
    Returns OOF predictions, test predictions, and a list of trained model objects.
    """
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    model_list = []
    
    if is_classification:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train.iloc[val_idx]
        y_va = y_train[val_idx]
        
        if is_classification:
            model = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=cat_features)
            oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds
        else:
            model = LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], categorical_feature=cat_features)
            oof_preds[val_idx] = model.predict(X_va)
            test_preds += model.predict(X_test) / n_folds
            
        model_list.append(model)
        
    return oof_preds, test_preds, model_list
