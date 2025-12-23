"""
This module implements the `plot_pl` function, which displays the profit/loss diagram 
of an options trading strategy.
"""

from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy import zeros, full

from optionlab.models import Outputs


def plot_pl(outputs: Outputs) -> None:
    """
    Displays the strategy's profit/loss diagram.

    ### Parameters

    `outputs`: output data from a strategy calculation with `optionlab.engine.run_strategy`.

    ### Returns

    `None`.
    """

    st = outputs.data
    inputs = outputs.inputs

    if len(st.strategy_profit) == 0:
        raise RuntimeError(
            "Before plotting the profit/loss profile diagram, you must run a calculation!"
        )

    rcParams.update({"figure.autolayout": True})

    zero_line = zeros(st.stock_price_array.shape[0])
    strike_call_buy = []
    strike_put_buy = []
    zero_call_buy = []
    zero_put_buy = []
    strike_call_sell = []
    strike_put_sell = []
    zero_call_sell = []
    zero_put_sell = []
    comment = "Profit/Loss diagram:\n--------------------\n"
    comment += "The vertical green dashed line corresponds to the position "
    comment += "of the stock's spot price. The right and left arrow "
    comment += "markers indicate the strike prices of calls and puts, "
    comment += "respectively, with blue representing long and red representing "
    comment += "short positions."

    plt.axvline(inputs.stock_price, ls="--", color="green")
    plt.xlabel("Stock price")
    plt.ylabel("Profit/Loss")
    plt.xlim(st.stock_price_array.min(), st.stock_price_array.max())

    for i, strike in enumerate(st.strike):
        if strike == 0.0:
            continue

        if st.type[i] == "call":
            if st.action[i] == "buy":
                strike_call_buy.append(strike)
                zero_call_buy.append(0.0)
            elif st.action[i] == "sell":
                strike_call_sell.append(strike)
                zero_call_sell.append(0.0)
        elif st.type[i] == "put":
            if st.action[i] == "buy":
                strike_put_buy.append(strike)
                zero_put_buy.append(0.0)
            elif st.action[i] == "sell":
                strike_put_sell.append(strike)
                zero_put_sell.append(0.0)

    target_line = None
    if inputs.profit_target is not None:
        comment += " The blue dashed line represents the profit target level."
        target_line = full(st.stock_price_array.shape[0], inputs.profit_target)

    loss_line = None
    if inputs.loss_limit is not None:
        comment += " The red dashed line represents the loss limit level."
        loss_line = full(st.stock_price_array.shape[0], inputs.loss_limit)

    print(comment)

    if loss_line is not None and target_line is not None:
        plt.plot(
            st.stock_price_array,
            zero_line,
            "m--",
            st.stock_price_array,
            loss_line,
            "r--",
            st.stock_price_array,
            target_line,
            "b--",
            st.stock_price_array,
            st.strategy_profit,
            "k-",
            strike_call_buy,
            zero_call_buy,
            "b>",
            strike_put_buy,
            zero_put_buy,
            "b<",
            strike_call_sell,
            zero_call_sell,
            "r>",
            strike_put_sell,
            zero_put_sell,
            "r<",
            markersize=10,
        )
    elif loss_line is not None:
        plt.plot(
            st.stock_price_array,
            zero_line,
            "m--",
            st.stock_price_array,
            loss_line,
            "r--",
            st.stock_price_array,
            st.strategy_profit,
            "k-",
            strike_call_buy,
            zero_call_buy,
            "b>",
            strike_put_buy,
            zero_put_buy,
            "b<",
            strike_call_sell,
            zero_call_sell,
            "r>",
            strike_put_sell,
            zero_put_sell,
            "r<",
            markersize=10,
        )
    elif target_line is not None:
        plt.plot(
            st.stock_price_array,
            zero_line,
            "m--",
            st.stock_price_array,
            target_line,
            "b--",
            st.stock_price_array,
            st.strategy_profit,
            "k-",
            strike_call_buy,
            zero_call_buy,
            "b>",
            strike_put_buy,
            zero_put_buy,
            "b<",
            strike_call_sell,
            zero_call_sell,
            "r>",
            strike_put_sell,
            zero_put_sell,
            "r<",
            markersize=10,
        )
    else:
        plt.plot(
            st.stock_price_array,
            zero_line,
            "m--",
            st.stock_price_array,
            st.strategy_profit,
            "k-",
            strike_call_buy,
            zero_call_buy,
            "b>",
            strike_put_buy,
            zero_put_buy,
            "b<",
            strike_call_sell,
            zero_call_sell,
            "r>",
            strike_put_sell,
            zero_put_sell,
            "r<",
            markersize=10,
        )
