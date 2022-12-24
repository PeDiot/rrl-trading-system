from plotly.graph_objs._figure import Figure
from plotly.graph_objs._scatter import Scatter
from plotly.graph_objs._layout import Layout

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from typing import Dict, List, Optional
from pandas.core.frame import DataFrame

def generate_layout(
    fig_height: int, 
    fig_width: int, 
    fig_title: str, 
    background_color: str="white", 
    x_axis_args: Optional[Dict]=None, 
    y_axis_args: Optional[Dict]=None 
) -> Layout: 
    """Description. Generate figure's overall layout."""

    d = {
        "height": fig_height,
        "plot_bgcolor": background_color,
        "title": {"text": (fig_title)},
        "width": fig_width,
        "xaxis": x_axis_args,
        "yaxis": y_axis_args
    }

    return Layout(d)

def generate_nrows(n_batches: int, ncols: int) -> int:
    """Description. Return number of rows based on number of trading batches."""

    if n_batches // ncols == 0: 
        nrows = n_batches // ncols
    else: 
        nrows = n_batches // ncols + 1

    return nrows 

def generate_subplot_titles(bacth_names: List) -> List: 
    """Description. Return subplot titles based on batch names."""
    
    return [f"Trading window {i+1}: {batch}" for i, batch in enumerate(bacth_names)]

def init_subplots(nrows: int, ncols: int, titles: List) -> Figure: 
    """Description. Return plotly subplots."""

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=titles)

    return fig

def __generate_no_returns_trace(n :int) -> Scatter:
    """Description. Generate horizontal dotted line for no returns.""" 
    return go.Scatter(
        x=[0, n], 
        y=[0, 0], 
        line=dict(color="red", dash="dot", width=.5), 
        name="No returns", 
        mode="lines", 
        showlegend=False)

def populate_subplots(
    fig: Figure, 
    df_backtest: DataFrame, 
    batch_names: List, 
    subplot_titles: List, 
    ncols: int, 
    x_varname: str="Days", 
    y_varname: str="Cumulative returns", 
    initial_invest: Optional[float]=None
) -> Figure: 
    """Description. Add traces to plotly Figure.
    
    Attributes: 
        - fig: initialized plotly Figure
        - df_backtest: backtest data set
        - batch_names: names of trading windows
        - subplot_titles: list of subplot titles
        - ncols: number of columns
        - x_varname: variable for the x axis ('Days', 'Trading dates') 
        - y_varname: variable of interest ('Cumulative returns', 'Cumulative profits') 
        - initial_invest: optional argument to add if y_varname='Cumulative profits'
    
    Returns: plotly Figure."""

    for ix, batch in enumerate(batch_names): 
        df_ = df_backtest.loc[df_backtest.Batch == batch, :]

        fig.add_trace(
            trace=go.Scatter(
                x=df_[x_varname].values, 
                y=df_[y_varname].values,
                name=subplot_titles[ix].split("-")[-1],  
                showlegend=False), 
            row=1 + ix // ncols, 
            col=1 + ix % ncols
        )

        if y_varname == "Cumulative returns": 
            trace = __generate_no_returns_trace(n=df_.shape[0])
            fig.add_trace(trace=trace, row=1 + ix // ncols, col=1 + ix % ncols)

        fig["layout"]["annotations"][ix]["font"] = dict(size=10) 

    for ix in range(len(subplot_titles)): 
        fig["layout"][f"xaxis{ix+1}"].update({
            "gridcolor": "lightgrey", 
            "title": {"text": "Days", "font": {"size": 10}}
        })
        fig["layout"][f"yaxis{ix+1}"].update({
            "gridcolor": "lightgrey", 
            "showticklabels": True, 
            "title": {"text": "%", "font": {"size": 10}}
        })

    return fig 

def plot_cumulative_profits(
    backtest_results: Dict, 
    buy_and_hold_df: DataFrame, 
    initial_invest: float, 
    layout: Layout
) -> Figure: 
    """Description. Visualize cumulative profits for different strategies.
    
    Attributes: 
        - backtest_results: dictionnary with backtest results for multiple strategies
        - buy_and_hold_df: data set with 'Buy and hold' results
        - initial_invest: initial amount invested
        - layout: plotly figure layout
    
    Returns: plotly Figure with multiple traces."""

    fig = go.Figure(layout=layout)

    for key, df in backtest_results.items(): 

        sr = df["Sharpe ratio"].mean()
        trace_name = f"RRL | {key} | sr={round(sr, 2)}"

        fig.add_trace(trace=go.Scatter(
            x=df["Dates"].values, 
            y=df["Cumulative profits"].values,
            mode="lines", 
            name=trace_name
        ))

    fig.add_trace(trace=go.Scatter(
        x=buy_and_hold_df.index.values, 
        y=initial_invest*buy_and_hold_df["cumulative_returns"].values,
        mode="lines", 
        name="Buy & Hold"
    ))

    title = "Cumulative profits from {} to {} | Initial investment={}".format(
        df["Dates"].min(), 
        df["Dates"].max(),
        initial_invest)

    fig.update_layout(title=title) 

    return fig

def make_cumrets_barplot(df: DataFrame, layout: Layout) -> Figure: 
    """Description. 
    Return a barplot representing total cumulative returns for each trading period."""

    fig = px.bar(
        df, 
        x="Trading window", 
        y="Cumulative returns", 
        color="Cumulative returns", 
        color_continuous_scale="RdYlGn") 

    fig.update_xaxes(showticklabels=False)
    fig.update_layout(layout)
    fig.update_coloraxes(showscale=False) 

    return fig