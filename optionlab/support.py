from __future__ import division

from functools import lru_cache

import numpy as np
from numpy import exp, abs, round, diff, flatnonzero, arange, inf
from numpy.lib.scimath import log, sqrt
from numpy.random import seed as np_seed_number, normal, laplace
from scipy import stats

from optionlab.black_scholes import get_d1, get_d2, get_option_price
from optionlab.models import (
    OptionType,
    Action,
    Distribution,
    ProbabilityOfProfitInputs,
    ProbabilityOfProfitArrayInputs,
    Range,
)


def get_pl_profile(
    option_type: OptionType,
    action: Action,
    x: float,
    val: float,
    n: int,
    s: np.ndarray,
    commission: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Returns the profit/loss profile and cost of an options trade at expiration.

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either 'call' or 'put'.
    action : str
        `Action` literal value, which must be either 'buy' or 'sell'.
    x : float
        Strike price.
    val : float
        Option price.
    n : int
        Number of options.
    s : numpy.ndarray
        Array of stock prices.
    commission : float, optional
        Brokerage commission. The default is 0.0.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Profit/loss profile and cost of an option trade at expiration.
    """

    if action == "buy":
        cost = -val
    elif action == "sell":
        cost = val
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    if option_type in ("call", "put"):
        return (
            n * _get_pl_option(option_type, val, action, s, x) - commission,
            n * cost - commission,
        )
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def get_pl_profile_stock(
    s0: float, action: Action, n: int, s: np.ndarray, commission: float = 0.0
) -> tuple[np.ndarray, float]:
    """
    Returns the profit/loss profile and cost of a stock position.

    Parameters
    ----------
    s0 : float
        Initial stock price.
    action : str
        `Action` literal value, which must be either 'buy' or 'sell'.
    n : int
        Number of shares.
    s : numpy.ndarray
        Array of stock prices.
    commission : float, optional
        Brokerage commission. The default is 0.0.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Profit/loss profile and cost of a stock position.
    """

    if action == "buy":
        cost = -s0
    elif action == "sell":
        cost = s0
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    return n * _get_pl_stock(s0, action, s) - commission, n * cost - commission


def get_pl_profile_bs(
    option_type: OptionType,
    action: Action,
    x: float,
    val: float,
    r: float,
    target_to_maturity_years: float,
    volatility: float,
    n: int,
    s: np.ndarray,
    y: float = 0.0,
    commission: float = 0.0,
) -> tuple[np.ndarray, float]:
    """
    Returns the profit/loss profile and cost of an options trade on a target date
    before expiration using the Black-Scholes model for option pricing.

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either 'call' or 'put'.
    action : str
        `Action` literal value, which must be either 'buy' or 'sell'.
    x : float
        Strike price.
    val : float
        Initial option price.
    r : float
        Annualized risk-free interest rate.
    target_to_maturity_years : float
        Time remaining to maturity from the target date, in years.
    volatility : float
        Annualized volatility of the underlying asset.
    n : int
        Number of options.
    s : numpy.ndarray
        Array of stock prices.
    y : float, optional
        Annualized dividend yield. The default is 0.0.
    commission : float, optional
        Brokerage commission. The default is 0.0.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Profit/loss profile and cost of an option trade before expiration.
    """

    if action == "buy":
        cost = -val
        fac = 1
    elif action == "sell":
        cost = val
        fac = -1
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    d1 = get_d1(s, x, r, volatility, target_to_maturity_years, y)
    d2 = get_d2(s, x, r, volatility, target_to_maturity_years, y)
    calcprice = get_option_price(
        option_type, s, x, r, target_to_maturity_years, d1, d2, y
    )

    return fac * n * (calcprice - val) - commission, n * cost - commission


@lru_cache
def create_price_seq(min_price: float, max_price: float) -> np.ndarray:
    """
    Generates a sequence of stock prices from a minimum to a maximum price with
    increment $0.01.

    Parameters
    ----------
    min_price : float
        Minimum stock price in the range.
    max_price : float
        Maximum stock price in the range.

    Returns
    -------
    numpy.ndarray
        Array of sequential stock prices.
    """

    if max_price > min_price:
        return round(
            (arange(int(max_price - min_price) * 100 + 1) * 0.01 + min_price), 2
        )
    else:
        raise ValueError("Maximum price cannot be less than minimum price!")


# TODO: Add a distribution, 'external', to allow customization.
# TODO: Remove or change Laplace
@lru_cache
def create_price_samples(
    s0: float,
    volatility: float,
    r: float,
    years_to_maturity: float,
    distribution: Distribution = "black-scholes",
    n: int = 100_000,
    y: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generates terminal stock prices assuming a statistical distribution.

    Parameters
    ----------
    s0 : float
        Spot price of the stock.
    volatility : float
        Annualized volatility of the underlying asset.
    r : float
        Annualized risk-free interest rate.
    years_to_maturity : float
        Time remaining to maturity, in years.
    distribution : str, optional
        `Distribution` literal value, which can be 'black-scholes' (the same as
        'normal') or 'laplace'. The default is 'black-scholes'.
    n : int, optional
        Number of terminal prices. The default is 100,000.
    y : float, optional
        Annualized dividend yield. The default is 0.0.
    seed : int | None, optional
        Seed for random number generation. The default is None.

    Returns
    -------
    numpy.ndarray
        Array of terminal prices.
    """

    np_seed_number(seed)

    drift = (r - y - 0.5 * volatility * volatility) * years_to_maturity

    if distribution in ("black-scholes", "normal"):
        array = exp(normal((log(s0) + drift), volatility * sqrt(years_to_maturity), n))
    elif distribution == "laplace":
        array = exp(
            laplace(
                (log(s0) + drift), (volatility * sqrt(years_to_maturity)) / sqrt(2.0), n
            )
        )
    else:
        np_seed_number(None)

        raise ValueError("Distribution not implemented yet!")

    np_seed_number(None)

    return array


def get_profit_range(
    s: np.ndarray, profit: np.ndarray, target: float = 0.01
) -> list[Range]:
    """
    Returns a list of stock price pairs, where each pair represents the lower and
    upper bounds within which an options trade is expected to make the desired profit.

    Parameters
    ----------
    s : numpy.ndarray
        Array of stock prices.
    profit : numpy.ndarray
        Array of profits and losses.
    target : float, optional
        Profit target. The default is 0.01.

    Returns
    -------
    list[Range]
        List of stock price pairs.
    """

    t = s[profit >= target]

    if t.shape[0] == 0:
        return []

    profit_range: list[list[float]] = []

    mask1 = diff(t) <= target + 0.001
    mask2 = diff(t) > target + 0.001
    maxi = flatnonzero(mask1[:-1] & mask2[1:]) + 1

    for i in range(maxi.shape[0] + 1):
        profit_range.append([])

        if i == 0:
            if t[0] == s[0]:
                profit_range[0].append(0.0)
            else:
                profit_range[0].append(t[0])
        else:
            profit_range[i].append(t[maxi[i - 1] + 1])

        if i == maxi.shape[0]:
            if t[t.shape[0] - 1] == s[s.shape[0] - 1]:
                profit_range[maxi.shape[0]].append(inf)
            else:
                profit_range[maxi.shape[0]].append(t[t.shape[0] - 1])
        else:
            profit_range[i].append(t[maxi[i]])

    return [(r[0], r[1]) for r in profit_range]


# TODO: Improve the description of the inputs.
def get_pop(
    profit_ranges: list[Range],
    inputs: ProbabilityOfProfitInputs | ProbabilityOfProfitArrayInputs,
) -> float:
    """
    Estimates the probability of profit (PoP) of an options trade.

    Parameters
    ----------
    profit_ranges : list[Range]
        List of stock price pairs, where each pair represents the lower and upper
        bounds within which the options trade makes a profit.
    inputs : ProbabilityOfProfitInputs | ProbabilityOfProfitArrayInputs
        Inputs for the probability of profit calculation. See the documentation
        for `ProbabilityOfProfitInputs` and `ProbabilityOfProfitArrayInputs` for
        more details.

    Returns
    -------
    float
        Probability of profit.
    """

    pop = 0.0

    if len(profit_ranges) == 0:
        return pop

    if isinstance(inputs, ProbabilityOfProfitInputs):
        stock_price = inputs.stock_price
        volatility = inputs.volatility
        years_to_maturity = inputs.years_to_maturity
        r = (
            inputs.interest_rate or 0.0
        )  # 'or' just for typing purposes, as `interest_rate` must be non-zero
        y = inputs.dividend_yield
        drift = (r - y - 0.5 * volatility * volatility) * years_to_maturity
        sigma = volatility * sqrt(years_to_maturity)

        if sigma == 0.0:
            sigma = 1e-10

        beta = sigma / sqrt(2.0)

        for p_range in profit_ranges:
            lval = p_range[0]
            hval = p_range[1]

            if lval <= 0.0:
                lval = 1e-10

            if inputs.source in ("normal", "black-scholes"):
                pop += stats.norm.cdf(
                    (log(hval / stock_price) - drift) / sigma
                ) - stats.norm.cdf((log(lval / stock_price) - drift) / sigma)
            else:
                pop += stats.laplace.cdf(
                    (log(hval / stock_price) - drift) / beta
                ) - stats.laplace.cdf((log(lval / stock_price) - drift) / beta)

    elif isinstance(inputs, ProbabilityOfProfitArrayInputs):
        stocks = inputs.array

        if stocks.shape[0] == 0:
            raise ValueError("The array of terminal stock prices is empty!")

        for i, p_range in enumerate(profit_ranges):
            lval, hval = p_range
            tmp1 = stocks[stocks >= lval]
            tmp2 = tmp1[tmp1 <= hval]
            pop += tmp2.shape[0]

        pop = pop / stocks.shape[0]

    else:
        raise ValueError("Source not supported yet!")

    return pop


def _get_pl_option(
    option_type: OptionType, opvalue: float, action: Action, s: np.ndarray, x: float
) -> np.ndarray:
    """
    Returns the profit or loss profile of an option leg at expiration.

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either 'call' or 'put'.
    opvalue : float
        Option price.
    action : str
        `Action` literal value, which must be either 'buy' or 'sell'.
    s : numpy.ndarray
        Array of stock prices.
    x : float
        Strike price.

    Returns
    -------
    numpy.ndarray
        Profit or loss profile of an option leg at expiration.
    """

    if action == "sell":
        return opvalue - _get_payoff(option_type, s, x)
    elif action == "buy":
        return _get_payoff(option_type, s, x) - opvalue
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")


def _get_payoff(option_type: OptionType, s: np.ndarray, x: float) -> np.ndarray:
    """
    Returns the payoff of an option leg at expiration.

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either 'call' or 'put'.
    s : numpy.ndarray
        Array of stock prices.
    x : float
        Strike price.

    Returns
    -------
    numpy.ndarray
        Payoff of an option leg at expiration.
    """

    if option_type == "call":
        return (s - x + abs(s - x)) / 2.0
    elif option_type == "put":
        return (x - s + abs(x - s)) / 2.0
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def _get_pl_stock(s0: float, action: Action, s: np.ndarray) -> np.ndarray:
    """
    Returns the profit or loss profile of a stock position.

    Parameters
    ----------
    s0 : float
        Spot price of the underlying asset.
    action : str
        `Action` literal value, which must be either 'buy' or 'sell'.
    s : numpy.ndarray
        Array of stock prices.

    Returns
    -------
    numpy.ndarray
        Profit or loss profile of a stock position.
    """

    if action == "sell":
        return s0 - s
    elif action == "buy":
        return s - s0
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")
