import yfinance as yf 

from dataclasses import dataclass, field
from pandas.core.frame import DataFrame
from typing import Tuple

from .preprocess import (
    extract_columns, 
    add_returns, 
    reverse_colnames, 
    sort_by_colnames, 
    get_returns_matrix, 
    get_feature_matrix, 
    to_batches,
)

from .indicators import add_indicators

@dataclass
class Data:
    """Description. 
    Financial data set.

    Attributes: 
        - start_date: training/trading start date
        - end_date: training/trading end date
        - assets: list of asset names 
        - indicators: list of technical indicators
        - batch_size: length of training/trading windows
    """ 
    start_date: str 
    end_date: str 
    assets: list=field(default_factory=list) 
    indicators: list=field(default_factory=list) 
    window_size:int = 100


    def __post_init__(self): 
        """Description. Apply data preprocessing based on class inputs.
        """
        self.n_assets, self.n_features = len(self.assets), len(self.indicators)

        self.df = yf.download(self.assets, start=self.start_date, end=self.end_date)
        self.df = extract_columns(self.df, self.assets, ["Close", "Volume", "High", "Low"])

        self.batches = to_batches(self.df, self.window_size)

    def preprocess_batch(self, batch: DataFrame) -> Tuple: 
        """Description. Compute returns and add indicators for a given batch.
        
        Attributes: 
            - batch: initial data set
        
        Returns: feature matrix, returns matrix."""

        batch = add_returns(batch, self.assets)
        batch = add_indicators(batch, self.assets, self.indicators)

        batch.columns = reverse_colnames(batch.columns)
        batch =  sort_by_colnames(batch)
        batch = batch.dropna(axis=0)

        features = get_feature_matrix(batch, self.n_assets, self.n_features)
        returns = get_returns_matrix(batch)

        return features, returns