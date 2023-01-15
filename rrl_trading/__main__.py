"""Description. 

Run RRL strategy from command line. 

Inputs: 
    - data config yaml file name
    - initial investment
    - transaction fees 
    - number of epochs
    - version
    
Save: 
    - trained RRL model 
    - visualisation of cumulative profits 
    - visualisation of total cumulative returns per trading window
    - portfolio allocation pie chart
    
Example: 
> python -m rrl_trading -config_path "./config/data.yaml" -initial_invest 100 -fees no_fees -n_epochs 100 -version 3"""

from .utils import read_yaml
from rrl_trading.data import Data, get_buy_and_hold_df
from rrl_trading.model import RRL
from rrl_trading.backtest import (
    run_rrl_strategy, 
    create_backtest_dataset, 
    generate_layout, 
    plot_cumulative_profits, 
    generate_df_barplot, 
    make_cumrets_barplot,
    plot_avg_portfolio_allocation, 
)
from rrl_trading.metrics import calc_sharpe_ratio

import sys 
from rich import print
from pickle import dump
import numpy as np 

def extract_info(flag: str):
    """Description. Extract information from command line."""
    i = sys.argv.index(flag) + 1
    return sys.argv[i]

if "-config_path" not in sys.argv:
    print ("You must provide config_path using the -config_path flag")
    sys.exit(1)

if "-initial_invest" not in sys.argv:
    print ("You must provide initial_invest using the -initial_invest flag")
    sys.exit(1)

if "-fees" not in sys.argv:
    print ("You must provide fees using the -fees flag")
    sys.exit(1)

if "-n_epochs" not in sys.argv:
    print ("You must provide n_epochs using the -n_epochs flag")
    sys.exit(1)

if "-version" not in sys.argv:
    print ("You must provide version using the -version flag")
    sys.exit(1)

config_path = extract_info("-config_path")
if config_path.split(".")[-1] != "yaml": 
    raise ValueError("config_path must be a .yaml file.")

data = read_yaml(obj_type=Data, path=config_path)

initial_invest = float(extract_info("-initial_invest"))

fees = extract_info("-fees")
if fees == "no_fees": 
    delta = 0.
else: 
    delta = float(''.join(filter(str.isdigit, fees)))
    delta = delta / 10000

n_epochs = int(extract_info("-n_epochs")) 

version = extract_info("-version")

if data.pca_ncp == None: 
    n_features = data.n_features 
else:
    n_features = data.pca_ncp

rrl = RRL(n_assets=data.n_assets, n_features=n_features, delta=delta) 
print(rrl) 

results = run_rrl_strategy(rrl, data, n_epochs, initial_invest)

def generate_file_name(): 
    file_name = "rrl_"

    if data.pca_ncp != None: 
        file_name += "pca_"

    if data.discrete_wavelet: 
        file_name += "dwt_"

    file_name += f"{fees}_{version}"

    return file_name 

model_name = generate_file_name()
setting = " ".join(model_name.split("_"))

path = f"./backup/{model_name}.pkl"
with open(path, "wb") as f: 
    dump(results, f)

buy_and_hold = get_buy_and_hold_df(data.df)
buy_and_hold_sr = calc_sharpe_ratio(returns=buy_and_hold.returns, window_size=252)

df_backtest = {setting: create_backtest_dataset(results)}

layout = generate_layout(
    fig_height=500, 
    fig_width=1100, 
    fig_title="", 
    x_axis_args={"gridcolor": "lightgrey"}, 
    y_axis_args={"gridcolor": "lightgrey", "title": {"text": "$"}}
)
fig = plot_cumulative_profits(
    df_backtest, 
    buy_and_hold, 
    buy_and_hold_sr, 
    initial_invest, 
    layout)
fig.write_image(f"./imgs/cum_profits_{model_name}.png")

df = df_backtest[setting]
df_barplot = generate_df_barplot(df)

title = f"% Cumulative returns at the end of each trading window | {setting}"

layout = generate_layout(
    fig_height=500, 
    fig_width=900, 
    fig_title=title,
    x_axis_args={"title": {"text": "Trading window"}}, 
    y_axis_args={"gridcolor": "lightgrey", "title": {"text": "%"}}
)
fig = make_cumrets_barplot(df_barplot, layout)
fig.write_image(f"./imgs/cum_rets_{model_name}.png")

layout = generate_layout(
    fig_height=500, 
    fig_width=700, 
    fig_title="Average portfolio allocation over trading windows"
)

fig = plot_avg_portfolio_allocation(results, layout)
fig.write_image(f"./imgs/allocation_{model_name}.png")
