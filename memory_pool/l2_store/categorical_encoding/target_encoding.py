import numpy as np
import pandas as pd
from category_encoders import TargetEncoder

def transform(X_train, y_train, X_test, cat_features):
    """
    Perform Target Encoding on high-cardinality categorical features.
    Returns transformed train and test dataframes.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    
    encoder = TargetEncoder(cols=cat_features)
    X_tr[cat_features] = encoder.fit_transform(X_tr[cat_features], y_train)
    X_te[cat_features] = encoder.transform(X_te[cat_features])
    
    return X_tr, X_te
