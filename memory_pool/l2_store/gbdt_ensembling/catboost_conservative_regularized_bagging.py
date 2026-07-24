import catboost as cb
import numpy as np


def get_refined_catboost_params(
    n_iterations: int = 2000,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 10.0,
    bagging_temperature: float = 1.0,
    random_seed: int = 42,
    **overrides
):
    """
    Returns a refined CatBoost parameter dictionary with conservative learning rate,
    higher iterations, L2 regularization, and bagging for better generalization.
    
    Args:
        n_iterations: Number of boosting iterations (default 2000)
        learning_rate: Conservative learning rate (default 0.03)
        l2_leaf_reg: L2 regularization strength (default 10.0)
        bagging_temperature: Bayesian bootstrap temperature (default 1.0)
        random_seed: Random seed for reproducibility
        **overrides: Additional parameters to override defaults
    
    Returns:
        Dictionary of CatBoost parameters
    """
    params = {
        # Core boosting parameters
        'iterations': n_iterations,
        'learning_rate': learning_rate,
        'depth': 6,
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        
        # Regularization
        'l2_leaf_reg': l2_leaf_reg,
        'random_strength': 1.0,
        'bagging_temperature': bagging_temperature,
        
        # Sampling for generalization
        'subsample': 0.8,
        'colsample_bylevel': 0.8,
        
        # Categorical handling
        'cat_features': None,  # Set externally
        'one_hot_max_size': 10,
        
        # Training control
        'early_stopping_rounds': 100,
        'od_type': 'Iter',
        'od_wait': 100,
        
        # Reproducibility & performance
        'random_seed': random_seed,
        'thread_count': -1,
        'verbose': False,
        
        # Probability calibration
        'posterior_sampling': False,
    }
    
    # Apply any overrides
    params.update(overrides)
    return params


class RefinedCatBoostModel:
    """
    Wrapper for CatBoost with refined hyperparameters.
    Preserves compatibility with existing CV and calibration pipelines.
    """
    
    def __init__(
        self,
        params=None,
        cat_features=None,
        **kwargs
    ):
        """
        Initialize with refined parameters.
        
        Args:
            params: Optional parameter dict (uses refined defaults if None)
            cat_features: List of categorical feature indices/names
            **kwargs: Passed to get_refined_catboost_params if params is None
        """
        if params is None:
            params = get_refined_catboost_params(**kwargs)
        
        self.params = params.copy()
        if cat_features is not None:
            self.params['cat_features'] = cat_features
        
        self.model = cb.CatBoostClassifier(**self.params)
        self.cat_features = cat_features
    
    def fit(self, X, y, eval_set=None, **fit_kwargs):
        """Fit the model with optional validation set."""
        fit_params = {
            'eval_set': eval_set,
            'use_best_model': True,
            'verbose': False
        }
        fit_params.update(fit_kwargs)
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict class labels."""
        return self.model.predict(X)
    
    def get_feature_importance(self, **kwargs):
        """Get feature importance scores."""
        return self.model.get_feature_importance(**kwargs)
    
    def save_model(self, path: str):
        """Save model to disk."""
        self.model.save_model(path)
    
    @classmethod
    def load_model(cls, path: str, cat_features=None):
        """Load model from disk."""
        model = cb.CatBoostClassifier()
        model.load_model(path)
        instance = cls.__new__(cls)
        instance.model = model
        instance.cat_features = cat_features
        instance.params = model.get_params()
        return instance


def create_refined_catboost(
    cat_features=None,
    **param_overrides
):
    """
    Factory function to create a RefinedCatBoostModel with default refined parameters.
    
    Args:
        cat_features: List of categorical feature indices or names
        **param_overrides: Any parameter overrides for the refined config
    
    Returns:
        Configured RefinedCatBoostModel instance
    """
    return RefinedCatBoostModel(cat_features=cat_features, **param_overrides)