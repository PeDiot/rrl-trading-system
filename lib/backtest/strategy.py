from typing import Dict 

from lib.model import train, validation 
from lib.model import RRL 
from lib.data import Data, get_batch_window

def run_rrl_strategy(
    rrl: RRL, 
    data: Data, 
    n_epochs: int=20, 
    initial_invest: float=100.
) -> Dict:
    """Description. Run the recurrent reinforcement learning (RRL) strategy. 

    Attributes: 
        - rrl: recurrent reinforcement learning model
        - data: financial data 
        - n_epochs: number of training epochs
        - initial_invest: initial invested amount

    Returns: dictionnary with 
        - batch names
        - trading dates
        - positions
        - portfolio returns
        - sharpe ratios
        - cumulative returns
        - cumulative profits
    """ 

    results = {
        "batch_names": [], 
        "trading_dates": [], 
        "positions": [], 
        "portfolio_returns": [], 
        "cumulative_returns": [], 
        "cumulative_profits": [[initial_invest]], 
        "sharpe_ratios": [], 
        "assets": data.assets, 
        "indicators": data.indicators, 
        "pca_ncp": data.pca_ncp
    }

    batches = list(data.batches)

    for ix in range(1, data.batches.ngroups):

        (_, batch_train), (_, batch_val) = batches[ix-1], batches[ix]
        
        batch_train = data.preprocess_batch(batch_train)
        X_tr, r_tr = data.split(batch_train)
        
        if ix == 1: rrl.init_weights()

        print(f"Training window: {get_batch_window(batch_train)}")
        train(rrl, X_tr, r_tr, n_epochs=n_epochs)
        
        batch_val = batch_val.copy()
        batch_val = data.preprocess_batch(batch_val)

        if batch_val.shape[0] > 30: 
            window_val = get_batch_window(batch_val)
            print(f"Validation window: {window_val}")

            results["trading_dates"].append(batch_val.index.strftime("%Y-%m-%d"))

            X_val, r_val = data.split(batch_val) 

            sharpe, cum_rets, cum_profits = validation(
                model=rrl, 
                X=X_val, 
                returns=r_val, 
                invest=results["cumulative_profits"][-1][-1])
            
            results["positions"].append(rrl.positions)

            results["portfolio_returns"].append(rrl.portfolio_returns)
            results["sharpe_ratios"].append(sharpe)
            results["cumulative_returns"].append(cum_rets)
            results["cumulative_profits"].append(cum_profits)

            print(f"Sharpe ratio on validation set: {sharpe}")
            print(f"Cumulative profits: {cum_profits[-1]}")

            batch_name = f"{window_val} - sr={round(sharpe, 2)}"
            results["batch_names"].append(batch_name)

        else: 
            print("Less than 30 observations in the validation set after preprocessing.")

    return results 