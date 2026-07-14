import numpy as np
import pandas as pd

def transform(X_train, X_test, cat_features):
    """
    Perform frequency encoding on categorical columns.
    Each value is replaced by its normalized count in the training dataset.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    
    for col in cat_features:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[col] = X_tr[col].map(freq).fillna(0)
        X_te[col] = X_te[col].map(freq).fillna(0)
        
    return X_tr, X_te
