import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

def transform(X_train, X_test, columns):
    """
    Scale continuous features robust to outliers using scikit-learn RobustScaler.
    Returns scaled train and test dataframes.
    """
    X_tr = X_train.copy()
    X_te = X_test.copy()
    
    scaler = RobustScaler()
    X_tr[columns] = scaler.fit_transform(X_tr[columns])
    X_te[columns] = scaler.transform(X_te[columns])
    
    return X_tr, X_te
