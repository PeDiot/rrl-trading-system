"""Description.

Implement an automated trading system based on recurrent reinforcement learning and technical indicators."""

import numpy as np 

from .forward import (
    linear_transform, 
    activate, 
    get_positions,
)

class RRL: 
    """Description. Recurent reinforcement learning trading system.
    
    Attributes: 
        - n_assets: number of assets in the portfolio
        - n_features: number of technical indicators
        - rho: learning rate
        - l2: regularization parameter
        - n_epochs: number of iterations for gradient ascent 
        - tol: thresold for optimality check
        - delta: transaction costs """
    
    def __init__(
        self, 
        n_assets: int, 
        n_features: int, 
        rho: float = .1, 
        l2: float=.01,
        n_epochs: int=100, 
        tol: float=.0001, 
        delta: float=0
    ):
        self.n_assets, self.n_features = n_assets, n_features
        self.delta = delta 
        self.rho, self.l2 = rho, l2
        self.n_epochs = n_epochs
        self.tol = tol

        np.random.seed(42)

        self.theta = np.random.normal(size=(self.n_assets, self.n_features+2))
        self.F0 = np.zeros(shape=(self.n_assets, ))

    def __repr__(self) -> str:
        return f"RRL({self.n_assets=})"
        
    def forward(self, X: np.ndarray, F_prev: np.ndarray): 
        """Description. Forward propagation.
        
        Attributes: 
            - X: (m, n+2) matrix of feature matrix
            - F_prev: (m, 1) matrix of positions at time t-1
            
        Returns: positions at time t."""

        if F_prev.shape != (self.n_assets, 1): 
            raise ValueError(f"F_prev must have {(self.n_assets, 1)} shape.")
        X = np.concatenate(
            (np.ones(shape=(self.n_assets, 1)), X, F_prev), 
            axis=1
        )
        y = linear_transform(X, self.theta)
        z = activate(y)
        return get_positions(z)
