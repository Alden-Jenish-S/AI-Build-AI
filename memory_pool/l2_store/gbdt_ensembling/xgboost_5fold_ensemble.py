import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import KFold, StratifiedKFold

def fit_predict(X_train, y_train, X_test, cat_features=None, n_folds=5, is_classification=True):
    """
    Train a 5-fold XGBoost model on tabular features.
    Returns OOF predictions, test predictions, and a list of trained model objects.
    """
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    model_list = []
    
    # For XGBoost categorical support, categorical features should be of pandas category type.
    X_train_xgb = X_train.copy()
    X_test_xgb = X_test.copy()
    if cat_features is not None:
        for col in cat_features:
            X_train_xgb[col] = X_train_xgb[col].astype('category')
            X_test_xgb[col] = X_test_xgb[col].astype('category')
            
    if is_classification:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_xgb, y_train)):
        X_tr = X_train_xgb.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_va = X_train_xgb.iloc[val_idx]
        y_va = y_train[val_idx]
        
        if is_classification:
            model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, enable_categorical=True, verbosity=0)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
            test_preds += model.predict_proba(X_test_xgb)[:, 1] / n_folds
        else:
            model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, enable_categorical=True, verbosity=0)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof_preds[val_idx] = model.predict(X_va)
            test_preds += model.predict(X_test_xgb) / n_folds
            
        model_list.append(model)
        
    return oof_preds, test_preds, model_list
