from sklearn.decomposition import PCA
import numpy as np 

def get_reduced_features(X: np.ndarray, ncp: int) -> np.ndarray: 
    """Description. 
    Apply Principal Components Analysis and return principal axes.
    
    Attributes: 
        - X: (T, m, n) numpy array
        - ncp: number of principal components to keep
        
    Returns: (T, m, ncp) numpy array."""

    T, m, n = X.shape
    X_ = X.reshape(-1, n)

    pca = PCA(n_components=ncp)
    X_pca = pca.fit_transform(X_)

    return X_pca.reshape(T, m, ncp)
