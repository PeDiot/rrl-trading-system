from .rrl import RRL
from lib.metrics.metrics import (
    calc_sharpe_ratio, 
    calc_cumulative_profits, 
    calc_cumulative_returns, 
)

import numpy as np 
from tqdm import tqdm 

from typing import Tuple

def train(
    model: RRL, 
    X: np.ndarray, 
    returns: np.ndarray, 
    n_epochs: int=100, 
    tol: float=1e-5
): 
    """Description. 
    Train the recurrent reinforcement learning model on data with m assets and n indicators. 
    
    Attributes: 
        - model: RRL type model
        - X: (T, m, n) feature matrix with indicators over T periods
        - returns: (T, m, 1) array of returns over T periods
        - n_epochs: number of epochs to train the RRL
        - tol: iteration stopping thresold"""

    epochs = tqdm(range(n_epochs))
    epochs.set_description("Training in progress...")

    for i in epochs: 
        model._init_gradients()
        model._init_portfolio()
        window_size = X.shape[0]

        for t in range(window_size): 
            Xt = X[t]
            rt = returns[t]

            model.forward(Xt, rt)

            if t > 0: model.backward(rt)

        S = calc_sharpe_ratio(model.portfolio_returns)
        epochs.set_postfix(sharpe_ratio=S)

        if i >= 1 and np.abs(S - S_prev) <= tol: 
            break
        else: 
            model.update_weights()
            S_prev = S

def validation(
    model: RRL, 
    X: np.ndarray, 
    returns: np.ndarray, 
    invest: float=100.
) -> Tuple: 
    """Description. 
    Implement the strategy on unseen data.
    
    Attributes: 
        - model: RRL type model
        - X: (T, m, n) feature matrix with indicators over T periods
        - returns: (T, m, 1) array of returns over T periods
        - invest: initial investment
    
    Returns: Sharpe ratio and cumulative profits"""
    model._init_portfolio()
    window_size = X.shape[0]

    for t in range(window_size): 
        Xt = X[t]
        rt = returns[t]
        model.forward(Xt, rt)

    sharpe_ratio = calc_sharpe_ratio(model.portfolio_returns, window_size=252)
    cum_returns = calc_cumulative_returns(model.portfolio_returns)
    cum_profits = calc_cumulative_profits(cum_returns, invest)

    return sharpe_ratio, cum_returns, cum_profits
