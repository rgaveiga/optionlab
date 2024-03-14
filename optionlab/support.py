from __future__ import division

import numpy as np
from numpy import asarray, ndarray, exp, abs, round, diff, flatnonzero, arange, inf
from numpy.lib.scimath import log, sqrt
from numpy.random import normal, laplace
from scipy import stats

from optionlab.black_scholes import get_d1_d2, get_option_price
from optionlab.models import OptionType, Action, Distribution


def get_pl_profile(
    op_type: OptionType,
    action: Action,
    x: float,
    val: float,
    n: int,
    s: np.ndarray,
    commission: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    get_pl_profile(op_type, action, x, val, n, s, commission) -> returns the profit/loss
    profile and cost of an option trade at expiration.

    Arguments:
    ----------
    op_type: option type ('call' or 'put').
    action: either 'buy' or 'sell' the option.
    x: strike price.
    val: option price.
    n: number of options.
    s: a numpy array of stock prices.
    commission: commission charged by the broker (default is zero).
    """

    if action == "buy":
        cost = -val
    elif action == "sell":
        cost = val
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    if op_type in ("call", "put"):
        return (
            n * _get_pl_option(op_type, val, action, s, x) - commission,
            n * cost - commission,
        )
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def get_pl_profile_stock(
    s0: float, action: Action, n: int, s: np.ndarray, commission: float = 0.0
) -> tuple[np.ndarray, float]:
    """
    get_pl_profile_stock(s0, action, n, s, commission) -> returns the profit/loss
    profile and cost of a stock position.

    Arguments:
    ----------
    s0: initial stock price.
    action: either 'buy' or 'sell' the shares.
    n: number of shares.
    s: a numpy array of stock prices.
    comission: commission charged by the broker (default is zero).
    """

    if action == "buy":
        cost = -s0
    elif action == "sell":
        cost = s0
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    return n * _get_pl_stock(s0, action, s) - commission, n * cost - commission


def get_pl_profile_bs(
    op_type: OptionType,
    action: Action,
    x: float,
    val: float,
    r: float,
    target_to_maturity: float,
    volatility: float,
    n: int,
    s: np.ndarray,
    y: float = 0.0,
    commission: float = 0.0,
):
    """
    get_pl_profile_bs(op_type,action,x,val,r,target_to_maturity,volatility,n,s,y,
    commission) -> returns the profit/loss profile and cost of an option trade
    on a target date before maturity using the Black-Scholes model for option
    pricing.

    Arguments:
    ----------
    op_type: option type (either 'call' or 'put').
    action: either 'buy' or 'sell' the option.
    x: strike.
    val: option price when the trade was open.
    r: risk-free interest rate.
    target_to_maturity: time remaining to maturity from the target date.
    volatility: annualized volatility of the underlying asset.
    n: number of options.
    s: a numpy array of stock prices.
    y: annualized dividend yield (default is zero)
    commission: commission charged by the broker (default is zero).
    """

    if action == "buy":
        cost = -val
        fac = 1
    elif action == "sell":
        cost = val
        fac = -1
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    d1, d2 = get_d1_d2(s, x, r, volatility, target_to_maturity, y)
    calcprice = get_option_price(op_type, s, x, r, target_to_maturity, d1, d2, y)

    return fac * n * (calcprice - val) - commission, n * cost - commission


def create_price_seq(min_price: float, max_price: float) -> np.ndarray:
    """
    create_price_seq(minprice, maxprice) -> generates a sequence of stock prices
    from 'minprice' to 'maxprice' with increment $0.01.

    Arguments:
    ----------
    minprice: minimum stock price in the range.
    maxprice: maximum stock price in the range.
    """
    if max_price > min_price:
        return round(
            (arange(int(max_price - min_price) * 100 + 1) * 0.01 + min_price), 2
        )
    else:
        raise ValueError("Maximum price cannot be less than minimum price!")


def create_price_samples(
    s0: float,
    volatility: float,
    time_to_maturity: int,
    r: float = 0.01,
    distribution: Distribution = "black-scholes",
    y: float = 0.0,
    n: int = 100_000,
) -> float:
    """
    create_price_samples(s0, volatility, time_to_maturity, r, distribution, y, n) -> generates
    random stock prices at maturity according to a statistical distribution.

    Arguments:
    ----------
    s0: spot price of the stock.
    volatility: annualized volatility.
    time_to_maturity: time left to maturity in units of year.
    r: annualized risk-free interest rate (default is 0.01). Used only if
       distribution is 'black-scholes'.
    distribution: statistical distribution used to generate random stock prices
                  at maturity. It can be 'black-scholes' (default), 'normal' or
                  'laplace'.
    y: annualized dividend yield (default is zero).
    n: number of randomly generated terminal prices.
    """
    if distribution == "normal":
        return exp(normal(log(s0), volatility * sqrt(time_to_maturity), n))
    elif distribution == "black-scholes":
        drift = (r - y - 0.5 * volatility * volatility) * time_to_maturity

        return exp(normal((log(s0) + drift), volatility * sqrt(time_to_maturity), n))
    elif distribution == "laplace":
        return exp(
            laplace(log(s0), (volatility * sqrt(time_to_maturity)) / sqrt(2.0), n)
        )
    else:
        raise ValueError("Distribution not implemented yet!")


def get_profit_range(
    s: np.ndarray, profit: np.ndarray, target: float = 0.01
) -> list[list[float]]:
    """
    get_profit_range(s, profit, target) -> returns pairs of stock prices, as a list,
    for which an option trade is expected to get the desired profit in between.

    Arguments:
    ----------
    s: a numpy array of stock prices.
    profit: a numpy array containing the profit (or loss) of the trade for each
            stock price in the stock price array.
    target: profit target (0.01 is the default).
    """

    profitrange: list[list[float]] = []

    t = s[profit >= target]

    if t.shape[0] == 0:
        return profitrange

    mask1 = diff(t) <= target + 0.001
    mask2 = diff(t) > target + 0.001
    maxi = flatnonzero(mask1[:-1] & mask2[1:]) + 1

    for i in range(maxi.shape[0] + 1):
        profitrange.append([])

        if i == 0:
            if t[0] == s[0]:
                profitrange[0].append(0.0)
            else:
                profitrange[0].append(t[0])
        else:
            profitrange[i].append(t[maxi[i - 1] + 1])

        if i == maxi.shape[0]:
            if t[t.shape[0] - 1] == s[s.shape[0] - 1]:
                profitrange[maxi.shape[0]].append(inf)
            else:
                profitrange[maxi.shape[0]].append(t[t.shape[0] - 1])
        else:
            profitrange[i].append(t[maxi[i]])

    return profitrange


def get_pop(
    profitranges: list[list[float]], source: Distribution = "black-scholes", **kwargs
) -> float:
    """
    get_pop(profitranges, source, kwargs) -> estimates the probability of profit
    (PoP) of an option trade.

    Arguments:
    ----------
    profitranges: a Python list containing the stock price ranges, as given by
                  'getprofitrange()', for which a trade results in profit.
    source: a string. It determines how the probability of profit is estimated
            (see next).
    **kwargs: a Python dictionary. The input that has to be provided depends on
              the value of the 'source' argument:

              * For 'source="normal"' or 'source="laplace"': the probability of
              profit is calculated assuming either a (log)normal or a (log)Laplace
              distribution of terminal stock prices at maturity.
              The keywords 'stockprice', 'volatility' and 'time_to_maturity' must be
              set.

              * For 'source="black-scholes"' (default): the probability of profit
              is calculated assuming a (log)normal distribution with risk neutrality
              as implemented in the Black-Scholes model.
              The keywords 'stockprice', 'volatility', 'interestrate' and
              'time_to_maturity' must be set. The keyword 'dividendyield' is optional.

              * For 'source="array"': the probability of profit is calculated
              from a 1D numpy array of stock prices typically at maturity generated
              by a Monte Carlo simulation (or another user-defined data generation
              process); this numpy array must be assigned to the 'array' keyword.
    """
    if not bool(kwargs):
        raise ValueError("'kwargs' is empty, nothing to do!")

    pop = 0.0
    drift = 0.0

    if len(profitranges) == 0:
        return pop

    if source in ("normal", "laplace", "black-scholes"):
        if "stockprice" in kwargs.keys():
            stockprice = float(kwargs["stockprice"])

            if stockprice <= 0.0:
                raise ValueError("Stock price must be greater than zero!")
        else:
            raise ValueError("Stock price must be provided!")

        if "volatility" in kwargs.keys():
            volatility = float(kwargs["volatility"])

            if volatility <= 0.0:
                raise ValueError("Volatility must be greater than zero!")
        else:
            raise ValueError("Volatility must be provided!")

        if "time_to_maturity" in kwargs.keys():
            time_to_maturity = float(kwargs["time_to_maturity"])

            if time_to_maturity < 0.0:
                raise ValueError("Time left to expiration must be a positive number!")
        else:
            raise ValueError("Time left to expiration must be provided!")

        if source == "black-scholes":
            if "interestrate" in kwargs.keys():
                r = float(kwargs["interestrate"])

                if r < 0.0:
                    raise ValueError(
                        "Risk-free interest rate must be a positive number!"
                    )
            else:
                raise ValueError("Risk-free interest rate must be provided!")

            if "dividendyield" in kwargs.keys():
                y = float(kwargs["dividendyield"])

                if y < 0.0:
                    raise ValueError("Dividend yield must be a positive number!")
            else:
                y = 0.0

            drift = (r - y - 0.5 * volatility * volatility) * time_to_maturity

        sigma = volatility * sqrt(time_to_maturity)

        if sigma == 0.0:
            sigma = 1e-10

        if source == "laplace":
            beta = sigma / sqrt(2.0)

        for p_range in profitranges:
            lval = p_range[0]
            hval = p_range[1]

            if lval <= 0.0:
                lval = 1e-10

            if source in ["normal", "black-scholes"]:
                pop += stats.norm.cdf(
                    (log(hval / stockprice) - drift) / sigma
                ) - stats.norm.cdf((log(lval / stockprice) - drift) / sigma)
            else:
                pop += stats.laplace.cdf(
                    log(hval / stockprice) / beta
                ) - stats.laplace.cdf(log(lval / stockprice) / beta)

    elif source == "array":
        if "array" in kwargs.keys():
            stocks = asarray(kwargs["array"])

            if stocks.shape[0] > 0:
                for i, p_range in enumerate(profitranges):
                    lval, hval = p_range
                    tmp1 = stocks[stocks >= lval]
                    tmp2 = tmp1[tmp1 <= hval]
                    pop += tmp2.shape[0]

                pop = pop / stocks.shape[0]
            else:
                raise ValueError("The array of stock prices is empty!")
        else:
            raise ValueError("An array of stock prices must be provided!")
    else:
        raise ValueError("Source not supported yet!")

    return pop


def _get_pl_option(
    op_type: OptionType, opvalue: float, action: Action, s: np.ndarray, x: float
) -> np.ndarray:
    """
    getPLoption(op_type,opvalue,action,s,x) -> returns the profit (P) or loss
    (L) per option of an option trade at expiration.

    Arguments:
    ----------
    op_type: option type (either 'call' or 'put').
    opvalue: option price.
    action: either 'buy' or 'sell' the option.
    s: a numpy array of stock prices.
    x: strike price.
    """
    if not isinstance(s, ndarray):
        raise TypeError("'s' must be a numpy array!")

    if action == "sell":
        return opvalue - _get_payoff(op_type, s, x)
    elif action == "buy":
        return _get_payoff(op_type, s, x) - opvalue
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")


def _get_payoff(op_type: OptionType, s: np.ndarray, x: float) -> np.ndarray:
    """
    get_payoff(op_type, s, x) -> returns the payoff of an option trade at expiration.

    Arguments:
    ----------
    op_type: option type (either 'call' or 'put').
    s: a numpy array of stock prices.
    x: strike price.
    """

    if op_type == "call":
        return (s - x + abs(s - x)) / 2.0
    elif op_type == "put":
        return (x - s + abs(x - s)) / 2.0
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def _get_pl_stock(s0: float, action: Action, s: np.ndarray) -> np.ndarray:
    """
    get_pl_stock(s0,action,s) -> returns the profit (P) or loss (L) of a stock
    position.

    Arguments:
    ----------
    s0: initial stock price.
    action: either 'buy' or 'sell' the stock.
    s: a numpy array of stock prices.
    """

    if action == "sell":
        return s0 - s
    elif action == "buy":
        return s - s0
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")
