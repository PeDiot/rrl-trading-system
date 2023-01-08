from sklearn.decomposition import PCA
import numpy as np 
from typing import Optional, Tuple 

def get_pca_features(X: np.ndarray, ncp: int, pca: Optional[PCA]=None) -> Tuple: 
    """Description. 
    Apply Principal Components Analysis and return principal axes.
    
    Attributes: 
        - X: (T, m, n) numpy array
        - ncp: number of principal components to keep
        
    Returns: 
        - (T, m, ncp) numpy array
        - PCA object."""

    T, m, n = X.shape
    X_ = X.reshape(-1, n)

    if pca == None: 
        pca = PCA(n_components=ncp)
        pca.fit(X_)
    
    X_pca = pca.transform(X_)

    return X_pca.reshape(T, m, ncp), pca
