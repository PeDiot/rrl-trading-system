"""Description. 

Methods to calculate derivatives for backpropagation."""

import numpy as np 
from typing import Tuple

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
    """

    if F.shape != Fprev.shape: 
        raise ValueError("F and Fprev must have the same shape.")
    elif Fprev.shape[0] != rt.shape[0]:
        raise ValueError("Fprev and rt must have the same second dimension.")
        
    dRtdF = -delta * (1 + np.dot(Fprev.T, rt)) * np.sign(F - Fprev)
    dRtdFprev = (1 -delta * np.sum(np.abs(F - Fprev))) * rt - dRtdF

    return dRtdF, dRtdFprev

def cal_positions_derivative(
    F: np.ndarray, 
    X: np.ndarray, 
    theta: np.ndarray, 
    y: np.ndarray, 
    dFdtheta_prev: np.ndarray
) -> Tuple: 
    """Description. Return derivative of portfolio positions wrt to network parameters.
    
    Attributes: 
        - F: (m, 1) matrix of portfolio positions at time t
        - y: (m, 1) matrix of linear transformations
        
    Returns: 
        - dFdtheta: matrix with derivatives of positions wrt theta matrix."""

    if F.shape != y.shape: 
        raise ValueError("F and y must have the same shapes.")

    dFdf = -np.dot(F, F.T) 
    diag = F * (1 - F)
    np.fill_diagonal(dFdf, diag)

    dFdy = np.zeros(shape=(y.shape[0], y.shape[0]))
    diag = 1 - np.tanh(y)**2
    np.fill_diagonal(dFdy, diag)

    thetaF_prev = theta[:, -1].T

    return dFdf * dFdy * (np.sum(X, axis=1) + np.diag(thetaF_prev) * dFdtheta_prev)