import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

def transform(X_train, X_test, columns, degree=2, interaction_only=True):
    """
    Generate polynomial and interaction features for continuous columns.
    Returns training and test dataframes with generated features.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_cols_tr = poly.fit_transform(X_tr[columns])
    poly_cols_te = poly.transform(X_te[columns])
    
    # Generate output column names
    new_cols = [f"poly_{i}" for i in range(poly_cols_tr.shape[1])]
    
    df_poly_tr = pd.DataFrame(poly_cols_tr, columns=new_cols, index=X_tr.index)
    df_poly_te = pd.DataFrame(poly_cols_te, columns=new_cols, index=X_te.index)
    
    # Drop original columns and concatenate
    X_tr = X_tr.drop(columns, axis=1)
    X_te = X_te.drop(columns, axis=1)
    
    X_tr = pd.concat([X_tr, df_poly_tr], axis=1)
    X_te = pd.concat([X_te, df_poly_te], axis=1)
    
    return X_tr, X_te
