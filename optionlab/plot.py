from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy import zeros, full

from optionlab import StrategyEngine


def plot_pl(st: StrategyEngine):
    """
    plot_pl -> displays the strategy's profit/loss profile diagram.

    Returns
    -------
    None.
    """
    if len(st.strategy_profit) == 0:
        raise RuntimeError(
            "Before plotting the profit/loss profile diagram, you must run a calculation!"
        )

    rcParams.update({"figure.autolayout": True})

    zero_line = zeros(st.s.shape[0])
    strike_call_buy = []
    strike_put_buy = []
    zero_call_buy = []
    zero_put_buy = []
    strike_call_sell = []
    strike_put_sell = []
    zero_call_sell = []
    zero_put_sell = []
    comment = "P/L profile diagram:\n--------------------\n"
    comment += "The vertical green dashed line corresponds to the position "
    comment += "of the stock's spot price. The right and left arrow "
    comment += "markers indicate the strike prices of calls and puts, "
    comment += "respectively, with blue representing long and red representing "
    comment += "short positions."

    plt.axvline(st._stock_price, ls="--", color="green")
    plt.xlabel("Stock price")
    plt.ylabel("Profit/Loss")
    plt.xlim(st.s.min(), st.s.max())

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
    if st._profit_target is not None:
        comment += " The blue dashed line represents the profit target level."
        target_line = full(st.s.shape[0], st._profit_target)

    loss_line = None
    if st._loss_limit is not None:
        comment += " The red dashed line represents the loss limit level."
        loss_line = full(st.s.shape[0], st._loss_limit)

    print(comment)

    if loss_line is not None and target_line is not None:
        plt.plot(
            st.s,
            zero_line,
            "m--",
            st.s,
            loss_line,
            "r--",
            st.s,
            target_line,
            "b--",
            st.s,
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
            st.s,
            zero_line,
            "m--",
            st.s,
            loss_line,
            "r--",
            st.s,
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
            st.s,
            zero_line,
            "m--",
            st.s,
            target_line,
            "b--",
            st.s,
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
            st.s,
            zero_line,
            "m--",
            st.s,
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
