import numpy as np
import pandas as pd
from scipy.stats import rankdata

def blend(preds_list, weights=None):
    """
    Perform rank blending on a list of predictions.
    Each set of predictions is converted to rank percentiles, and then averaged (optionally weighted).
    preds_list: list of 1D numpy arrays of same size.
    """
    if weights is None:
        weights = [1.0 / len(preds_list)] * len(preds_list)
    else:
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        
    blended = np.zeros(len(preds_list[0]))
    for preds, w in zip(preds_list, weights):
        ranks = rankdata(preds)
        ranks_norm = (ranks - 1) / (len(preds) - 1) if len(preds) > 1 else ranks
        blended += ranks_norm * w
        
    return blended
