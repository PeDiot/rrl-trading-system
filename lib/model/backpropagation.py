"""Description. 

Methods to calculate derivatives for backpropagation."""

import numpy as np 
from typing import Tuple

from lib.metrics.metrics import calc_sharpe_ratio

def calc_portfolio_returns_derivatives(
    F: np.ndarray, 
    Fprev: np.ndarray, 
    rt: np.ndarray, 
    delta: float
) -> Tuple:
    """Description. Derive portfolio returns wtr to positions at time t and time t-1.
    
    Attributes: 
        - F: (m, 1) matrix ofportfolio positions at time t
        - F_prev: (m, 1) matrix of portfolio positions at time t-1
        - rt: (m, 1) matrix of asset returns at time t
        - delta: transaction costs

    Returns: 
        - dRdF: (m, 1) matrix derivative of portfolio returns wrt positions at time t
        - dRdFprev: (m, 1) matrix derivative of portfolio returns wrt positions at time t-1
    """

    if F.shape != Fprev.shape: 
        raise ValueError("F and Fprev must have the same shape.")
    elif Fprev.shape[0] != rt.shape[0]:
        raise ValueError("Fprev and rt must have the same second dimension.")
        
    dRdF = -delta * (1 + np.dot(Fprev.T, rt)) * np.sign(F - Fprev)
    dRdFprev = (1 -delta * np.sum(np.abs(F - Fprev))) * rt - dRdF

    return dRdF, dRdFprev

def calc_positions_derivative(
    F: np.ndarray, 
    X: np.ndarray, 
    theta: np.ndarray, 
    y: np.ndarray, 
    dFprev: np.ndarray
) -> np.ndarray: 
    """Description. Return derivative of portfolio positions wrt to network parameters.
    
    Attributes: 
        - F: (m, 1) matrix of portfolio positions at time t
        - X: (m, n+2) feature matrix
        - theta: (m, n+2) matrix of network parameters
        - y: (m, 1) matrix of linear transformations
        - dFprev: (m, n+2) derivative matrix of portfolio positions at time t-1
        
    Returns: matrix with derivatives of positions wrt theta matrix."""

    if F.shape != y.shape: 
        raise ValueError("F and y must have the same shapes.")

    dFdf = -np.dot(F, F.T) 
    diag = F * (1 - F)
    np.fill_diagonal(dFdf, diag)

    dFdy = np.zeros(shape=(y.shape[0], y.shape[0]))
    diag = 1 - np.tanh(y)**2
    np.fill_diagonal(dFdy, diag)

    theta_Fprev = theta[:, -1].T

    a = np.dot(dFdf, dFdy)
    b = X + np.dot(np.diag(theta_Fprev), dFprev)

    return np.dot(a, b)

def calc_sharpe_derivative(
    portfolio_returns: np.ndarray, 
    returns: np.ndarray, 
    delta: float, 
    F: np.ndarray, 
    Fprev: np.ndarray, 
    dFprev: np.ndarray, 
    X: np.ndarray, 
    theta: np.ndarray, 
    y: np.ndarray
) -> Tuple: 
    """Description. Return matrix of sharpe ratio derivatives wrt network parameters. 
    
    Attributes: 
        - portfolio_returns: (t, 1) array of returns up to time t
        - returns: (m, 1) matrix of asset returns at time t
        - delta: transaction costs
        - F: (m, 1) matrix of portfolio positions at time t
        - F_prev: (m, 1) matrix of portfolio positions at time t-1
        - dFprev: (m, n+2) matrix derivative of portfolio positions at time t-1
        - X: (m, n+2) feature matrix
        - theta: (m, n+2) matrix of network parameters
        - y: (m, 1) matrix of linear transformations
    
    Returns: 
        - dS: (m, m) array of Sharpe ratio derivative wrt parameter matrix theta
        - dF: (m, m) array of positions derivative wrt parameter matrix theta"""

    T = portfolio_returns.shape[0]
    A = np.mean(portfolio_returns) 
    S = calc_sharpe_ratio(portfolio_returns, window_size=T)

    dSdA = S * (1 + S**2) / A
    dSdB = - S**3 / (2 * A**2)
    dAdR = 1 / T
    dBdR = 2 * portfolio_returns[-1, 0]
    a = dSdA * dAdR + dSdB * dBdR

    dRdF, dRdFprev = calc_portfolio_returns_derivatives(F, Fprev, returns, delta)
    dF = calc_positions_derivative(F, X, theta, y, dFprev)
    b = np.dot(np.diag(dRdF[:, 0]), dF) + np.dot(np.diag(dRdFprev[:, 0]), dFprev) 

    dS = a * b

    return dS, dF
