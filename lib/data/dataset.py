import yfinance as yf 

from dataclasses import dataclass, field

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
        self.df = add_returns(self.df, self.assets)
        self.df = add_indicators(self.df, self.assets, self.indicators)

        self.df.columns = reverse_colnames(self.df.columns)
        self.df = sort_by_colnames(self.df)

        self.df = self.df.dropna()

        self._features = get_feature_matrix(self.df, self.n_assets, self.n_features)
        self._returns = get_returns_matrix(self.df)

        self.batch_features, self.batch_returns = to_batches(self._features, self._returns, self.window_size)