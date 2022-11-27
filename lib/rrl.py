"""Description.

Implement an automated trading system based on recurrent reinforcement learning and technical indicators."""

import numpy as np 
from tqdm import tqdm

from .forward import (
    linear_transform, 
    activate, 
    get_positions,
)

from .backpropagation import calc_sharpe_derivative

from .metrics import (
    calc_portfolio_returns, 
    calc_cumulative_profits, 
    calc_sharpe_ratio, 
)

class RRL: 
    """Description. Recurent reinforcement learning trading system.
    
    Attributes: 
        - n_assets: number of assets in the portfolio
        - n_features: number of technical indicators
        - rho: learning rate for gradient ascent
        - l2: regularization parameter for gradient ascent
        - initial_invest: initial investment
        - delta: transaction costs """
    
    def __init__(
        self, 
        n_assets: int, 
        n_features: int, 
        delta: float=0, 
        rho: float = .1, 
        l2: float=.01
    ):
        """Description. RRL network initialization. 
        
        Returns: 
            - theta: initialized network parameter matrix
            - F: initilized positions
            - dF: initialized derivative matrix of positions wrt theta
            - dS: initialized derivative matrix of Sharpe ratio wrt theta
            - portfolio_returns: array to store portfolio returns"""

        self.n_assets, self.n_features = n_assets, n_features
        self.rho, self.l2 = rho, l2
        self.delta = delta 

        np.random.seed(42)

        self.theta = np.random.normal(size=(self.n_assets, self.n_features+2))
        self.F = np.zeros(shape=(self.n_assets, 1))

        self.dS = np.zeros(shape=(self.n_assets, self.n_features+2))
        self.dF = np.zeros(shape=(1, self.n_assets, self.n_features+2))

        self.portfolio_returns = np.ones(shape=(1, 1))

    def __repr__(self) -> str:
        return f"RRL(n_assets={self.n_assets}, n_features={self.n_features}, delta={self.delta}, rho={self.rho}, l2={self.l2})"
        
    def forward(self, X: np.ndarray, returns: np.ndarray): 
        """Description. Forward propagation.
        
        Attributes: 
            - X: (m, n+2) feature matrix
            - returns: (m, 1) array of asset returns
            
        Returns: positions at time t."""

        F_prev = self.F[:, -1].reshape(self.n_assets, -1)
        if F_prev.shape != (self.n_assets, 1): 
            raise ValueError(f"F_prev must have {(self.n_assets, 1)} shape.")
        self._X = np.concatenate(
            (np.ones(shape=(self.n_assets, 1)), X, F_prev), 
            axis=1
        )
        self._y = linear_transform(self._X, self.theta)
        self._z = activate(self._y)
        self.F = np.concatenate(
            (self.F, get_positions(self._z)), 
            axis=1
        )

        self.portfolio_returns = np.concatenate((
            self.portfolio_returns, 
            calc_portfolio_returns(
                self.F[:, -1].reshape(self.n_assets, -1), 
                F_prev, 
                returns, 
                self.delta)
        ))
        
    def backward(self, returns: np.ndarray): 
        """Description. Run backpropagation.
        
        Attributes: 
            - returns: (m, 1) array of returns at time t."""

        dS, dF = calc_sharpe_derivative(
            self.portfolio_returns, 
            returns, 
            self.delta, 
            self.F[:, -1].reshape(self.n_assets, -1),
            self.F[:, -2].reshape(self.n_assets, -1),
            self.dF[-1], 
            self._X, 
            self.theta, 
            self._y
        )

        self.dS = self.dS + dS
        self.dF = np.concatenate((
            self.dF, dF.reshape(1, self.n_assets, self.n_features+2)
        ))

    def update_weights(self): 
        """Description. Update network weights using gradient ascent rule."""
        self.theta = (1 - self.rho * self.l2) * self.theta + self.rho * self.dS

def train(
    model: RRL, 
    X: np.ndarray, 
    returns: np.ndarray, 
    n_epochs: int, 
    tol: float
): 
    """Description. 
    Train the recurrent reinforcement learning model on data with m assets and n indicators. 
    
    Attributes: 
        - model: RRL type model
        - X: (m, n) feature matrix with indicators
        - returns: (m, 1) array of returns
        - n_epochs: number of epochs to train the RRL
        - tol: iteration stopping thresold
        
    Returns: updated RRL model."""
    ...