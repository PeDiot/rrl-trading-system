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

from lib.metrics.metrics import (
    calc_portfolio_returns, 
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

    def init_weights(self): 
        """Description. Initialize RRL parameters."""
        np.random.seed(42)
        self.theta = np.random.normal(size=(self.n_assets, self.n_features+2))

    def _init_portfolio(self): 
        """Description. Initialize portfolio positions and returns."""
        self.F = np.zeros(shape=(self.n_assets, 1))
        self.portfolio_returns = np.ones(shape=(1, 1))

    def _init_gradients(self): 
        """Description. Initialize RRL gradients."""
        self.dS = np.zeros(shape=(self.n_assets, self.n_features+2))
        self._dF = np.zeros(shape=(1, self.n_assets, self.n_features+2))

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
            self._dF[-1], 
            self._X, 
            self.theta, 
            self._y
        )

        self.dS = self.dS + dS
        self._dF = np.concatenate((
            self._dF, dF.reshape(1, self.n_assets, self.n_features+2)
        ))

    def update_weights(self): 
        """Description. Update network weights using gradient ascent rule."""
        self.theta = (1 - self.rho * self.l2) * self.theta + self.rho * self.dS

def train(
    model: RRL, 
    X: np.ndarray, 
    returns: np.ndarray, 
    n_epochs: int=100, 
    tol: float=0
): 
    """Description. 
    Train the recurrent reinforcement learning model on data with m assets and n indicators. 
    
    Attributes: 
        - model: RRL type model
        - X: (T, m, n) feature matrix with indicators over T periods
        - returns: (T, m, 1) array of returns over T periods
        - n_epochs: number of epochs to train the RRL
        - tol: iteration stopping thresold"""

    sharpe_ratios = []

    epochs = tqdm(range(n_epochs))
    epochs.set_description("Training in progress...")

    for i in epochs: 
        model._init_gradients()
        model._init_portfolio()

        for t in range(X.shape[0]): 
            Xt = X[t]
            rt = returns[t]

            model.forward(Xt, rt)
            model.backward(rt)

        S = calc_sharpe_ratio(model.portfolio_returns)
        epochs.set_postfix(sharpe_ratio=S)

        if i >= 1 and np.abs(S - sharpe_ratios[-1]) <= tol: 
            break
        else: 
            model.update_weights()
            sharpe_ratios.append(S)

def validation(model: RRL, X: np.ndarray, returns: np.ndarray): 
    """Description. 
    Implement the strategy on unseen data.
    
    Attributes: 
        - model: RRL type model
        - X: (T, m, n) feature matrix with indicators over T periods
        - returns: (T, m, 1) array of returns over T periods
    
    Returns: Sharpe ratio."""
    model._init_portfolio()

    print("Validation in progress...")

    for t in range(X.shape[0]): 
        Xt = X[t]
        rt = returns[t]
        model.forward(Xt, rt)

    sharpe_ratio = calc_sharpe_ratio(model.portfolio_returns)
    print(f"{sharpe_ratio=}")

    return sharpe_ratio
