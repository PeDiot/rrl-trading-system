import yfinance as yf 

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from serde import serialize, deserialize

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
from .pca import get_pca_features
from .dwt import DiscreteWavelet

from rrl_trading.metrics import calc_cumulative_returns

def get_buy_and_hold_df(data: DataFrame) -> DataFrame: 
    """Description. 
    Return cumulative returns for 'Buy and hold' strategy assuming equal weights for each asset."""

    df = data.loc[:, data.columns.str.contains("Close")].pct_change().dropna()
    df.loc[:, "returns"] = df.sum(axis=1) / df.shape[1]
    df.loc[:, "cumulative_returns"] = calc_cumulative_returns(df["returns"].values)

    return df.loc[:, ["returns", "cumulative_returns"]]

@serialize
@deserialize
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
        - discrete_wavelet: boolean indicating whether to denoise features using DiscretWavelet.
    """ 
    start_date: str 
    end_date: str 
    window_size:int

    assets: list=field(default_factory=list) 
    indicators: list=field(default_factory=list)

    pca_ncp: Optional[int]=None
    discrete_wavelet: bool=False

    def __post_init__(self): 
        """Description. Apply data preprocessing based on class inputs."""

        self.n_assets = len(self.assets)

        self.n_features = len(self.indicators)
        if "HTS" in self.indicators: 
            self.n_features += 1            

        self.df = yf.download(self.assets, start=self.start_date, end=self.end_date)
        self.df = extract_columns(self.df, self.assets, ["Close", "Volume", "High", "Low"])

        self.batches = to_batches(self.df, self.window_size)

        if self.discrete_wavelet: 
            self._dwt = DiscreteWavelet()

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

    def split(
        self, 
        batch: DataFrame, 
        scaler: Optional[StandardScaler]=None, 
        pca:Optional[PCA]=None
    ) -> Tuple: 
        """Description. 
        Extract feature matrix and returns from batch dataframe.
        
        Attributes: 
            - batch: preprocessed batch data frame
            - scaler: optional scaler used to apply normalization
            - pca: optional PCA object
            
        Returns: 
            - feature matrix 
            - returns matrix 
            - standard scaler
            - optional PCA object.
            
        Details: 
            - when PCA is used, fit is only applied on the train data set
            - when discret wavelet is used, features are denoised."""

        features, scaler = get_feature_matrix(batch, self.n_assets, self.n_features, scaler)
        window_size = features.shape[0]

        if self.discrete_wavelet: 
            features = np.apply_along_axis(
                func1d=lambda x: self._dwt.denoise(x)[:window_size], 
                axis=0, 
                arr=features)

        returns = get_returns_matrix(batch)
        
        if self.pca_ncp != None: 
            features, pca = get_pca_features(features, self.pca_ncp, pca)

            if self.discrete_wavelet: 
                features = np.apply_along_axis(
                    func1d=lambda x: self._dwt.denoise(x)[:window_size], 
                    axis=0, 
                    arr=features)
            
            return features, returns, scaler, pca
        
        return features, returns, scaler