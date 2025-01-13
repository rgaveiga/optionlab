from __future__ import division

from functools import lru_cache

import numpy as np
from numpy import abs, round, arange, inf, sum, exp
from numpy.lib.scimath import log, sqrt
from scipy import stats

from optionlab.black_scholes import get_d1, get_d2, get_option_price
from optionlab.models import (
    OptionType,
    Action,
    BlackScholesModelInputs,
    ArrayInputs,
    Range,
    PoPOutputs,
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
        return round((arange((max_price - min_price) * 100 + 1) * 0.01 + min_price), 2)
    else:
        raise ValueError("Maximum price cannot be less than minimum price!")


def get_profit_range(
    s: np.ndarray, profit: np.ndarray, target: float = 0.01
) -> (list[Range], list[Range]):
    """
    Returns a tuple of lists of stock price ranges: one representing the ranges
    where the options trade returns are equal to or greater than the target, and
    the other representing the ranges where they fall short.

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
    tuple(list[Range], list[Range])
        Tuple of lists of stock price pairs.
    """

    profit_range = []
    loss_range = []

    crossings = _get_sign_changes(profit, target)
    n_crossings = len(crossings)

    if n_crossings == 0:
        if profit[0] >= target:
            return [(0, inf)], [(None, None)]
        else:
            return [(None, None)], [(0, inf)]

    lb_profit = hb_profit = None
    lb_loss = hb_loss = None

    for i, index in enumerate(crossings):
        if i == 0:
            if profit[index] < profit[index - 1]:
                lb_profit = 0.0
                hb_profit = s[index - 1]
                lb_loss = s[index]
            else:
                lb_profit = s[index]
                lb_loss = 0.0
                hb_loss = s[index - 1]
        elif i == n_crossings - 1:
            if profit[index] > profit[index - 1]:
                lb_profit = s[index]
                hb_profit = inf
                hb_loss = s[index - 1]
            else:
                hb_profit = s[index - 1]
                lb_loss = s[index]
                hb_loss = inf
        else:
            if profit[index] > profit[index - 1]:
                lb_profit = s[index]
                hb_loss = s[index - 1]
            else:
                hb_profit = s[index - 1]
                lb_loss = s[index]

        if lb_profit is not None and hb_profit is not None:
            profit_range.append((lb_profit, hb_profit))

            lb_profit = hb_profit = None

        if lb_loss is not None and hb_loss is not None:
            loss_range.append((lb_loss, hb_loss))

            lb_loss = hb_loss = None

    return profit_range, loss_range


def get_pop(
    s: np.ndarray,
    profit: np.ndarray,
    inputs_data: BlackScholesModelInputs | ArrayInputs | dict,
    target: float = 0.01,
) -> PoPOutputs:
    """
    Estimates the probability of profit (PoP) of an options trading strategy.

    Parameters
    ----------
    s : np.ndarray
        Array of stock prices.
    profit : np.ndarray
        Array of profits and losses.
    inputs_data : BlackScholesModelInputs | ArrayInputs | dict
        Input data used to estimate the probability of profit. See the documentation
        for `BlackScholesModelInputs` and `ArrayInputs` for more details.
    target : float, optional
        Target profit. The default is 0.01.

    Returns
    -------
    PoPOutputs
        Outputs. See the documentation for `PoPOutputs` for more details.
    """

    probability_of_reaching_target = 0.0
    probability_of_missing_target = 0.0

    t_ranges = get_profit_range(s, profit, target)

    if isinstance(inputs_data, dict):
        input_type = inputs_data["model"]

        if input_type in ("black-scholes", "normal"):
            inputs = BlackScholesModelInputs.model_validate(inputs_data)
        elif input_type == "array":
            inputs = ArrayInputs.model_validate(inputs_data)
        else:
            raise ValueError("Inputs are not valid!")
    else:
        inputs = inputs_data

        if isinstance(inputs, BlackScholesModelInputs):
            input_type = "black-scholes"
        elif isinstance(inputs, ArrayInputs):
            input_type = "array"
        else:
            raise ValueError("Inputs are not valid!")

    if input_type in ("black-scholes", "normal"):
        sigma = (
            inputs.volatility * sqrt(inputs.years_to_target_date)
            if inputs.volatility > 0.0
            else 1e-10
        )

        for i, t in enumerate(t_ranges):
            prob = []
            exp_profit = []

            for p_range in t:
                lval = log(p_range[0]) if p_range[0] > 0.0 else -inf
                hval = log(p_range[1])
                drift = (
                    inputs.interest_rate
                    - inputs.dividend_yield
                    - 0.5 * inputs.volatility * inputs.volatility
                ) * inputs.years_to_target_date
                m = log(inputs.stock_price) + drift
                w = stats.norm.cdf((hval - m) / sigma) - stats.norm.cdf(
                    (lval - m) / sigma
                )

                if w > 0.0:
                    v = stats.norm.pdf((hval - m) / sigma) - stats.norm.pdf(
                        (lval - m) / sigma
                    )
                    exp_stock = round(
                        exp(m - sigma * v / w), 2
                    )  # Using inverse Mills ratio

                    if exp_stock > 0.0 and exp_stock <= s.max():
                        exp_stock_index = np.where(s == exp_stock)[0][0]
                        exp_profit.append(profit[exp_stock_index])
                        prob.append(w)

            if len(t) > 0:
                prob = np.array(prob)
                exp_profit = np.array(exp_profit)

                if i == 0:
                    probability_of_reaching_target = sum(prob)
                    expected_return_above_target = round(
                        sum(exp_profit * prob) / probability_of_reaching_target, 2
                    )
                else:
                    probability_of_missing_target = sum(prob)
                    expected_return_below_target = round(
                        sum(exp_profit * prob) / probability_of_missing_target, 2
                    )

    elif input_type == "array":
        if inputs.array.shape[0] == 0:
            raise ValueError("The array is empty!")

        tmp1 = inputs.array[inputs.array >= target]
        tmp2 = inputs.array[inputs.array < target]

        probability_of_reaching_target = tmp1.shape[0] / inputs.array.shape[0]
        probability_of_missing_target = 1.0 - probability_of_reaching_target

        expected_return_above_target = tmp1.mean() if tmp1.shape[0] > 0 else None
        expected_return_below_target = tmp2.mean() if tmp2.shape[0] > 0 else None

    return PoPOutputs(
        probability_of_reaching_target=probability_of_reaching_target,
        probability_of_missing_target=probability_of_missing_target,
        reaching_target_range=t_ranges[0],
        missing_target_range=t_ranges[1],
        expected_return_above_target=expected_return_above_target,
        expected_return_below_target=expected_return_below_target,
    )


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


def _get_sign_changes(profit: np.ndarray, target: float) -> list:
    """
    Returns a list of the indices in the array of profits where the sign changes.

    Parameters
    ----------
    profit : np.ndarray
        Array of profits and losses.
    target : float
        Profit target.

    Returns
    -------
    list
        List of indices.
    """

    p_temp = profit - target + 1e-10

    sign_changes = (np.sign(p_temp[:-1]) * np.sign(p_temp[1:])) < 0

    return (np.where(sign_changes)[0] + 1).tolist()
