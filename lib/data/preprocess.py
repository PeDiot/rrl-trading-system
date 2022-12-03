from pandas.core.frame import DataFrame
from typing import List, Tuple
import numpy as np 

def extract_columns(df: DataFrame, assets: List, feature_names: List) -> DataFrame: 
    """Description. Return data set with closing prices as features."""

    df = df.loc[:, feature_names]
    df.columns = [
        "_".join(col).strip().replace(" ", "_")
        for col in df.columns.values
    ]
    return df

def add_returns(df: DataFrame, assets: List) -> DataFrame: 
    """Description. Add daily returns based on closing prices."""
    cols = [col for col in df.columns if "Close" in col]
    df[["Returns_"+asset for asset in assets]] = df[cols].pct_change()
    
    return df

def reverse_colnames(columns: List) -> List:
    """Description. Reverse column names."""
    return [
        "_".join(list(reversed(col.split("_"))))
        for col in columns
    ]

def sort_by_colnames(df: DataFrame) -> DataFrame:
    """Description. Sort columns alphabetically."""

    return df.reindex(sorted(df.columns), axis=1)

def get_returns_matrix(df: DataFrame) -> np.ndarray: 
    """Description. Extract returns matrix for m assets over T periods."""

    cols = [col for col in df.columns if "Returns" in col]
    return df[cols].values.reshape(df.shape[0], len(cols), 1)

def get_feature_matrix(df: DataFrame, n_assets: int, n_features: int) -> np.ndarray: 
    """Description. Normalize indicators and convert DataFrame into (T, m, n) array."""
    to_remove = ["close", "low", "high", "returns", "volume"]
    df = df.loc[:, ~df.columns.str.contains("|".join(to_remove), case=False)]
    df_std = df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    return df_std.values.reshape(df.shape[0], n_assets, n_features)

def to_batches(features: np.ndarray, returns: np.ndarray, window_size: int) -> Tuple: 
    """Description. 
    Divide feature and return matrices into multiple batches.
    
    Attributes: 
        - features: (T, m, n) array with indicators
        - returns: (T, m, 1) array of returns
        - window_size: time length of each batches

    Returns: batches of features, batches of returns
    """

    assert features.shape[0] == returns.shape[0]
    n_batches = int(features.shape[0] / window_size)

    return np.array_split(features, n_batches), np.array_split(returns, n_batches)