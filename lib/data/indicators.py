from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import List

import numpy as np

from ta.momentum import rsi
from ta.trend import macd
from ta.volume import (
    on_balance_volume, 
    money_flow_index, 
    chaikin_money_flow, 
)
from ta.volatility import average_true_range

def momentum_indicator(close: Series, window: int=14) -> Series:
    """Description. Calculate momentum indicator."""
    close_shifted = close.shift(periods=window)

    return 100 * close / close_shifted

def normalized_average_true_range(close: Series, atr: Series) -> Series: 
    """Description. Calculate normalized average true range."""
    
    return 100 * atr / close

INDICATORS = {
    "MOM": lambda df, asset: momentum_indicator(df["Close_"+asset]), 
    "MACD": lambda df, asset: macd(df["Close_"+asset]), 
    "MFI": lambda df, asset: money_flow_index(df["High_"+asset], df["Low_"+asset], df["Close_"+asset], df["Volume_"+asset]), 
    "RSI": lambda df, asset: rsi(df["Close_"+asset]), 
    "ATR": lambda df, asset: average_true_range(df["High_"+asset], df["Low_"+asset], df["Close_"+asset]), 
    "NATR": lambda df, asset: normalized_average_true_range(df["Close_"+asset], df["ATR_"+asset]), 
    "CO": lambda df, asset: chaikin_money_flow(df["High_"+asset], df["Low_"+asset], df["Close_"+asset], df["Volume_"+asset]), 
    "OBV": lambda df, asset: on_balance_volume(close=df["Close_"+asset], volume=df["Volume_"+asset])
}

def add_indicators(
    df: DataFrame, 
    assets: List, 
    indicators: List
) -> DataFrame: 
    """Description. Add technical indicators.
    
    Attributes: 
        - df: financial data set to transform
        - assets: asset names
        - indicators: indicator names
    
    Returns: transformed financial data set."""

    for asset in assets: 
        for indicator in indicators: 

            if indicator not in INDICATORS.keys(): 
                raise ValueError(f"Select indicators in {list(INDICATORS.keys())}")

            else: 
                col_name = f"{indicator}_{asset}"
                df[col_name] = INDICATORS[indicator](df, asset)

    return df

def get_buy_and_hold_returns(data: DataFrame) -> DataFrame: 
    """Description. 
    Return cumulative returns for 'Buy and hold' strategy assuming equal weights for each asset."""

    returns_df = data.loc[:, data.columns.str.contains("Close")].pct_change().dropna()
    returns_df = returns_df.apply(lambda x: np.cumprod(1+x))
    returns_df["cumulative_returns"] = returns_df.sum(axis=1) / returns_df.shape[1]

    return returns_df