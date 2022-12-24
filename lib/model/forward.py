import numpy as np 

def linear_transform(x: np.ndarray, theta: np.ndarray) -> np.ndarray: 
    """Description. Apply element-wise product and sum elements.

    Attributes: 
        - x: (m, n+2) matrix of feature matrix
        - theta: (m, n+2) matrix of network weights"""

    if theta.shape != x.shape:
        raise ValueError(f"theta and x must have the same shape.")
    ones_ = np.ones(shape=(x.shape[1], 1))
    return np.dot((x * theta), ones_)

def activate(y: np.ndarray) -> np.ndarray: 
    """Description. Activate neurons with tanh function."""

    return np.tanh(y)

def get_positions(x: np.ndarray) -> np.ndarray: 
    """Description. Apply softmax activation to get asset positions."""

    s = np.sum(np.exp(x))
    positions = np.exp(x) / s
    return positions.T.reshape(-1, 1)