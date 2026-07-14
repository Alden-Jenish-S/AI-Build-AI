import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_splits(X, y, n_splits=5, shuffle=True, random_state=42):
    """
    Generate train/val index splits for Stratified K-Fold.
    Returns a list of tuples containing (train_idx, val_idx) as numpy arrays.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    # Convert generators/lists of numpy arrays
    splits = []
    for train_idx, val_idx in skf.split(X, y):
        splits.append((train_idx.astype(np.int64), val_idx.astype(np.int64)))
    return splits
