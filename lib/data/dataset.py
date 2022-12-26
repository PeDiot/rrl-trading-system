import yfinance as yf 

from dataclasses import dataclass, field
from pandas.core.frame import DataFrame
from typing import Tuple, Optional

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

from .pca import get_reduced_features

@dataclass
class Data:
    """Description. 
    Financial data set.

    Attributes: 
        - start_date: training/trading start date
        - end_date: training/trading end date
        - window_size: length of training/trading windows
        - assets: list of asset names 
        - indicators: technical indicators to compute
        - pca_ncp: optional number of principal components for PCA
    """ 
    start_date: str 
    end_date: str 
    window_size:int

    assets: list=field(default_factory=list) 
    indicators: list=field(default_factory=list)

    pca_ncp: Optional[int]=None

    def __post_init__(self): 
        """Description. Apply data preprocessing based on class inputs."""

        self.n_assets = len(self.assets)

        self.n_features = len(self.indicators)
        if "HTS" in self.indicators: 
            self.n_features += 1            

        self.df = yf.download(self.assets, start=self.start_date, end=self.end_date)
        self.df = extract_columns(self.df, self.assets, ["Close", "Volume", "High", "Low"])

        self.batches = to_batches(self.df, self.window_size)


    def preprocess_batch(self, batch: DataFrame) -> DataFrame: 
        """Description. Compute returns and add indicators for a given batch.
        
        Attributes: 
            - batch: initial data set
        
        Returns: batch dataframe."""

        batch = add_returns(batch, self.assets)
        batch = add_indicators(batch, self.assets, self.indicators)

        batch.columns = reverse_colnames(batch.columns)
        batch = sort_by_colnames(batch)
        batch = batch.dropna(axis=0)

        return batch 

    def split(self, batch: DataFrame) -> Tuple: 
        """Description. Extract feature matrix and returns from batch dataframe."""

        features = get_feature_matrix(batch, self.n_assets, self.n_features)
        
        if self.pca_ncp != None: 
            features = get_reduced_features(features, self.pca_ncp)

        returns = get_returns_matrix(batch)

        return features, returns