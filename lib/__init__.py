from .rrl import RRL

from .forward import (
    linear_transform, 
    activate, 
    get_positions,
)

from .backpropagation import (
    calc_portfolio_returns_derivatives, 
    calc_positions_derivative, 
    calc_sharpe_derivative,
)

from .metrics import (
    calc_portfolio_returns, 
    calc_cumulative_profits, 
    calc_sharpe_ratio, 
)