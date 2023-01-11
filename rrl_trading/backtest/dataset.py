from typing import List, Dict 
from pandas.core.frame import DataFrame

import numpy as np 
import pandas as pd
from itertools import chain

def create_backtest_dataset(results: Dict) -> DataFrame: 
    """Description. 
    Return a data set with batch names, days, dates, cumulative returns and cumulative profits."""

    window_sizes = [len(window) for window in results["trading_dates"]]

    batch_names = list(
        chain(*[
            [results["batch_names"][ix] for _ in range(size)]
            for ix, size in enumerate(window_sizes) 
        ])
    )
    days = list(
        chain(*[
            [t for t in range(size)]
            for size in window_sizes
        ])
    )
    sharpe_ratios = list(
        chain(*[
            [results["sharpe_ratios"][ix] for _ in range(size)]
            for ix, size in enumerate(window_sizes) 

        ])
    )

    df = pd.DataFrame({
        "Batch": batch_names, 
        "Days": days, 
        "Dates": list(chain(*results["trading_dates"])), 
        "Cumulative profits": list(chain(*results["cumulative_profits"][1:])),
        "Cumulative returns": list(chain(*results["cumulative_returns"])), 
        "Sharpe ratio": sharpe_ratios
    })

    df["Cumulative returns"] = df["Cumulative returns"].apply(lambda x: (x-1) * 100)

    return df

def generate_df_barplot(df: DataFrame) -> DataFrame: 
    """Description. Get last cumulative returns for each trading batch.""" 

    df_barplot = df.loc[ 
        df["Days"] == df["Days"].max(), 
        ["Batch", "Cumulative returns"]
    ]
    df_barplot["Cumulative returns"] = df_barplot["Cumulative returns"].round(2)

    return df_barplot.rename(columns={"Batch": "Trading window"}) 

def _make_df_pie(results: Dict) -> DataFrame: 
    """Description. 
    Aggregate startegy results to make pie chart dataframe."""

    assets = results["assets"]
    positions = results["positions"]

    avg_positions_per_window = np.array([p.mean(axis=1) for p in positions]) 
    avg_positions = avg_positions_per_window.mean(axis=0)


    df_pie = pd.DataFrame({
        "Asset": assets, 
        "% Share": 100 * avg_positions
    })

    return df_pie