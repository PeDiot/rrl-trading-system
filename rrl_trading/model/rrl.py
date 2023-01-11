import numpy as np 

from .forward import (
    linear_transform, 
    activate, 
    get_positions,
)

from .backpropagation import calc_sharpe_derivative

from rrl_trading.metrics.metrics import calc_portfolio_returns

class RRL: 
    """Description. Recurent reinforcement learning trading system.
    
    Attributes: 
        - n_assets: number of assets in the portfolio
        - n_features: number of technical indicators
        - rho: learning rate for gradient ascent
        - l2: regularization parameter for gradient ascent
        - initial_invest: initial investment
        - delta: transaction costs"""
    
    def __init__(
        self, 
        n_assets: int, 
        n_features: int, 
        delta: float=0, 
        rho: float = .1, 
        l2: float=.01
    ):
        """Description. RRL network initialization."""

        self.n_assets, self.n_features = n_assets, n_features
        self.rho, self.l2 = rho, l2
        self.delta = delta 

    def init_weights(self): 
        """Description. Initialize RRL parameters."""
        np.random.seed(42)
        self.theta = np.random.normal(size=(self.n_assets, self.n_features+2))

    def _init_portfolio(self): 
        """Description. Initialize portfolio positions and returns."""
        self.positions = np.zeros(shape=(self.n_assets, 1))
        self.portfolio_returns = np.array([])

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

        self.positions_prev = self.positions[:, -1].reshape(self.n_assets, -1)
        if self.positions_prev.shape != (self.n_assets, 1): 
            raise ValueError(f"positions_prev must have {(self.n_assets, 1)} shape.")

        self._X = np.concatenate(
            (np.ones(shape=(self.n_assets, 1)), X, self.positions_prev), 
            axis=1)

        self._y = linear_transform(self._X, self.theta)
        self._z = activate(self._y)

        self.positions = np.concatenate(
            (self.positions, get_positions(self._z)), 
            axis=1)

    def update_portfolio_returns(self, returns: np.ndarray): 
        """Description. 
        Compute portfolio returns for positions at time t using returns at time t+2."""

        portfolio_returns = calc_portfolio_returns(
            self.positions[:, -1].reshape(self.n_assets, -1), 
            self.positions_prev, 
            returns, 
            self.delta)

        if self.portfolio_returns.shape == (0,): 
            self.portfolio_returns = portfolio_returns
        else: 
            self.portfolio_returns = np.concatenate((self.portfolio_returns, portfolio_returns)) 
        
    def backward(self, returns: np.ndarray): 
        """Description. Run backpropagation.
        
        Attributes: 
            - returns: (m, 1) array of returns at time t."""

        dS, dF = calc_sharpe_derivative(
            self.portfolio_returns, 
            returns, 
            self.delta, 
            self.positions[:, -1].reshape(self.n_assets, -1),
            self.positions[:, -2].reshape(self.n_assets, -1),
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