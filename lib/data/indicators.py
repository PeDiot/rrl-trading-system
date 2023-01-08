from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import List

from ta.momentum import rsi
from ta.trend import macd
from ta.volume import (
    on_balance_volume, 
    money_flow_index, 
    chaikin_money_flow, 
)
from ta.volatility import average_true_range

from talib.abstract import (
    HT_DCPHASE,
    HT_SINE, 
    HT_TRENDMODE
)

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
    "OBV": lambda df, asset: on_balance_volume(close=df["Close_"+asset], volume=df["Volume_"+asset]), 
    "HTDCP": lambda df, asset: HT_DCPHASE(df["Close_"+asset]), 
    "HTS": lambda df, asset: HT_SINE(df["Close_"+asset]), 
    "HTTMMM": lambda df, asset: HT_TRENDMODE(df["Close_"+asset])
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
                if indicator == "HTS": 
                    sine, leadsine = INDICATORS[indicator](df, asset)
                    df.loc[:, f"SINE_{asset}"] = sine 
                    df.loc[:, f"LEADSINE_{asset}"] = leadsine
                else: 
                    col_name = f"{indicator}_{asset}"
                    df.loc[:, col_name] = INDICATORS[indicator](df, asset)

    return df