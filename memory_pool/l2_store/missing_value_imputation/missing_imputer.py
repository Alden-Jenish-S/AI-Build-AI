import numpy as np
import pandas as pd

def transform(X_train, X_test, cont_cols, cat_cols):
    """
    Impute missing values:
    - continuous columns: replace with training median
    - categorical columns: replace with training mode (or most frequent value)
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    
    for col in cont_cols:
        median = X_tr[col].median()
        if pd.isna(median):
            median = 0.0
        X_tr[col] = X_tr[col].fillna(median)
        X_te[col] = X_te[col].fillna(median)
        
    for col in cat_cols:
        mode = X_tr[col].mode()
        mode_val = mode.iloc[0] if len(mode) > 0 else "missing"
        X_tr[col] = X_tr[col].fillna(mode_val)
        X_te[col] = X_te[col].fillna(mode_val)
        
    return X_tr, X_te
