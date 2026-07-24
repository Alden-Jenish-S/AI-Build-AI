import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class GMMMixedTypeClustering(BaseEstimator, TransformerMixin):
    """
    Clustering pipeline for mixed-type tabular data.
    
    Steps:
    1. Identify numeric and categorical columns (excluding 'Id' if present).
    2. Preprocess: RobustScaler for numeric, OneHotEncoder for categorical.
    3. Dimensionality reduction: PCA to 30 components.
    4. GaussianMixture with full covariance, select n_components by BIC in range 5..20.
    5. Predict cluster labels (max posterior probability) and relabel to 0..K-1.
    """
    def __init__(self, n_components_range=(5, 20), pca_components=30, random_state=42):
        self.n_components_range = n_components_range
        self.pca_components = pca_components
        self.random_state = random_state
        self.preprocessor_ = None
        self.pca_ = None
        self.gmm_ = None
        self.best_n_components_ = None
        self.label_mapping_ = None
        self.feature_names_in_ = None
        self.numeric_features_ = None
        self.categorical_features_ = None

    def _identify_features(self, X):
        """Separate numeric and categorical columns, drop 'Id' if present."""
        cols = X.columns.tolist()
        if 'Id' in cols:
            cols.remove('Id')
        numeric = X[cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical = [c for c in cols if c not in numeric]
        return numeric, categorical

    def fit(self, X, y=None):
        """Fit the full pipeline."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.feature_names_in_ = X.columns.tolist()
        self.numeric_features_, self.categorical_features_ = self._identify_features(X)
        
        # Preprocessing
        transformers = []
        if self.numeric_features_:
            transformers.append(('num', RobustScaler(), self.numeric_features_))
        if self.categorical_features_:
            transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features_))
        
        self.preprocessor_ = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
        X_pre = self.preprocessor_.fit_transform(X)
        
        # PCA
        n_comp = min(self.pca_components, X_pre.shape[1], X_pre.shape[0])
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_pre)
        
        # GaussianMixture model selection via BIC
        best_bic = np.inf
        best_gmm = None
        best_n = None
        for n in range(self.n_components_range[0], self.n_components_range[1] + 1):
            if n > X_pca.shape[0]:
                break
            gmm = GaussianMixture(n_components=n, covariance_type='full',
                                  random_state=self.random_state, n_init=5, max_iter=200)
            gmm.fit(X_pca)
            bic = gmm.bic(X_pca)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_n = n
        
        self.gmm_ = best_gmm
        self.best_n_components_ = best_n
        
        # Create label mapping to consecutive integers based on component order
        self.label_mapping_ = {i: i for i in range(best_n)}
        return self

    def predict(self, X):
        """Predict cluster labels for each row."""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        # Ensure same columns as fit
        missing = set(self.feature_names_in_) - set(X.columns)
        for col in missing:
            X[col] = np.nan
        X = X[self.feature_names_in_]
        
        X_pre = self.preprocessor_.transform(X)
        X_pca = self.pca_.transform(X_pre)
        probs = self.gmm_.predict_proba(X_pca)
        labels = np.argmax(probs, axis=1)
        # Relabel to consecutive integers (already consecutive, but safe)
        unique_labels = np.unique(labels)
        mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([mapping[l] for l in labels])
        return labels

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

def gmm_mixed_type_bic_pca30(X, n_components_range=(5, 20), pca_components=30, random_state=42):
    """
    Fit a Gaussian Mixture Model on mixed-type tabular data with automatic component selection via BIC.
    
    Parameters
    ----------
    X : array-like or pandas DataFrame, shape (n_samples, n_features)
        Input data with mixed numeric and categorical columns. Column 'Id' is ignored if present.
    n_components_range : tuple of (int, int), default=(5, 20)
        Range of n_components to search (inclusive).
    pca_components : int, default=30
        Number of PCA components after preprocessing.
    random_state : int, default=42
        Random seed for reproducibility.
    
    Returns
    -------
    labels : numpy.ndarray, shape (n_samples,)
        Cluster labels as consecutive integers 0..K-1.
    """
    clusterer = GMMMixedTypeClustering(
        n_components_range=n_components_range,
        pca_components=pca_components,
        random_state=random_state
    )
    return clusterer.fit_predict(X)