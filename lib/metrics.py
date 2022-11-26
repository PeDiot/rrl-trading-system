"""Description. 

Financial metrics."""

import numpy as np

def calc_portfolio_returns(
    F: np.ndarray, 
    Fprev: np.ndarray, 
    rt: np.ndarray, 
    delta: float
) -> float: 
    """Description. Calculate portfolio returns at time t.
    
    Attributes: 
        - F: (m, 1) matrix of portfolio positions at time t
        - F_prev: (m, 1) matrix of portfolio positions at time t-1
        - rt: (m, 1) matrix of asset returns at time t
        - delta: transaction costs
    
    Returns: portfolio returns """

    if Fprev.shape != F.shape: 
        raise ValueError("Fprev and F must have the same shape.")
    elif Fprev.shape[0] != rt.shape[0]:
        raise ValueError("Fprev and rt must have the same second dimension.")

    a = 1 + np.dot(Fprev.T, rt)
    b = 1 - delta * np.sum(np.abs(F - Fprev))
    rt = a * b
    return rt.flatten()[0] - 1

def calc_cumulative_profits(returns: np.ndarray, initial: float) -> float: 
    """Description. Calculate cumulative profits since initial investment.

    Attributes: 
        - returns: portfolio returns
        - initial: initial invested amount

    Returns: cumulative returns"""

    return initial * (np.cumprod(1 + returns) - 1)

def calc_sharpe_ratio(returns: np.ndarray) -> float: 
    """Return Sharpe ratio from an array of returns."""
    return np.mean(returns) / np.std(returns)
