"""
This module implements a number of helper functions that are not intended to be 
called directly by users, but rather support functionalities within the 
`optionlab.engine.run_strategy` function.
"""

from __future__ import division

from functools import lru_cache

from typing import Optional, cast

import numpy as np
from numpy import abs, round, arange
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
    FloatOrNdarray,
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

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `action`: either *'buy'* or *'sell'*.

    `x`: strike price.

    `val`: option price.

    `n`: number of options.

    `s`: array of stock prices.

    `commission`: brokerage commission.

    ### Returns

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

    ### Parameters

    `s0`: initial stock price.

    `action`: either *'buy'* or *'sell'*.

    `n`: number of shares.

    `s`: array of stock prices.

    `commission`: brokerage commission.

    ### Returns

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
) -> tuple[FloatOrNdarray, float]:
    """
    Returns the profit/loss profile and cost of an options trade on a target date
    before expiration using the Black-Scholes model for option pricing.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `action`: either *'buy'* or *'sell'*.

    `x`: strike price.

    `val`: initial option price.

    `r`: annualized risk-free interest rate.

    `target_to_maturity_years`: time remaining to maturity from the target date,
    in years.

    `volatility`: annualized volatility of the underlying asset.

    `n`: number of options.

    `s`: array of stock prices.

    `y`: annualized dividend yield.

    `commission`: brokerage commission.

    ### Returns

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

    d1: FloatOrNdarray = get_d1(s, x, r, volatility, target_to_maturity_years, y)
    d2: FloatOrNdarray = get_d2(s, x, r, volatility, target_to_maturity_years, y)
    calcprice: FloatOrNdarray = get_option_price(
        option_type, s, x, r, target_to_maturity_years, d1, d2, y
    )
    profile: FloatOrNdarray = fac * n * (calcprice - val) - commission

    return profile, n * cost - commission


@lru_cache
def create_price_seq(min_price: float, max_price: float) -> np.ndarray:
    """
    Generates a sequence of stock prices from a minimum to a maximum price with
    increment $0.01.

    ### Parameters

    `min_price`: minimum stock price in the range.

    `max_price`: maximum stock price in the range.

    ### Returns

    Array of sequential stock prices.
    """

    if max_price > min_price:
        return round((arange((max_price - min_price) * 100 + 1) * 0.01 + min_price), 2)
    else:
        raise ValueError("Maximum price cannot be less than minimum price!")


def get_pop(
    s: np.ndarray,
    profit: np.ndarray,
    inputs_data: BlackScholesModelInputs | ArrayInputs,
    target: float = 0.01,
) -> PoPOutputs:
    """
    Estimates the probability of profit (PoP) of an options trading strategy.

    ### Parameters

    `s`: array of stock prices.

    `profit`: array of profits (and losses).

    `inputs_data`: input data used to estimate the probability of profit.

    `target`: target return.

    ### Returns

    Outputs of a probability of profit (PoP) calculation.
    """

    probability_of_reaching_target: float
    probability_of_missing_target: float

    expected_return_above_target: Optional[float] = None
    expected_return_below_target: Optional[float] = None

    t_ranges = _get_profit_range(s, profit, target)

    reaching_target_range = t_ranges[0] if t_ranges[0] != [(0.0, 0.0)] else []
    missing_target_range = t_ranges[1] if t_ranges[1] != [(0.0, 0.0)] else []

    if isinstance(inputs_data, BlackScholesModelInputs):
        probability_of_reaching_target, probability_of_missing_target = _get_pop_bs(
            s, profit, inputs_data, t_ranges
        )
        expected_return_above_target, expected_return_below_target = (
            _compute_expected_returns_bs(s, profit, inputs_data, target)
        )
    elif isinstance(inputs_data, ArrayInputs):
        (
            probability_of_reaching_target,
            expected_return_above_target,
            probability_of_missing_target,
            expected_return_below_target,
        ) = _get_pop_array(inputs_data, target)

    return PoPOutputs(
        probability_of_reaching_target=probability_of_reaching_target,
        probability_of_missing_target=probability_of_missing_target,
        reaching_target_range=reaching_target_range,
        missing_target_range=missing_target_range,
        expected_return_above_target=expected_return_above_target,
        expected_return_below_target=expected_return_below_target,
    )


def _get_pl_option(
    option_type: OptionType, opvalue: float, action: Action, s: np.ndarray, x: float
) -> np.ndarray:
    """
    Returns the profit or loss profile of an option leg at expiration.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `opvalue`: option price.

    `action`: either *'buy'* or *'sell'*.

    `s`: array of stock prices.

    `x`: strike price.

    ### Returns

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

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `s`: array of stock prices.

    `x`: strike price.

    ### Returns

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

    ### Parameters

    `s0`: spot price of the underlying asset.

    `action`: either *'buy'* or *'sell'*.

    `s`: array of stock prices.

    ### Returns

    Profit or loss profile of a stock position.
    """

    if action == "sell":
        return s0 - s
    elif action == "buy":
        return s - s0
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")


def _get_pop_bs(
    s: np.ndarray,
    profit: np.ndarray,
    inputs: BlackScholesModelInputs,
    profit_range: tuple[list[Range], list[Range]],
) -> tuple[float, float]:
    """
    Estimates the probability of profit (PoP) of an options trading strategy using
    the Black-Scholes model.

    ### Parameters

    `s`: array of stock prices.

    `profit`: array of profits and losses.

    `inputs`: input data used to estimate the probability of profit.

    `profit_range`: lists of stock price pairs defining the profit and loss
    ranges.

    ### Returns

    Probability of reaching the return target and probability of missing the
    return target.
    """

    sigma = (
        inputs.volatility * sqrt(inputs.years_to_target_date)
        if inputs.volatility > 0.0
        else 1e-10
    )
    drift = (
        inputs.interest_rate
        - inputs.dividend_yield
        - 0.5 * inputs.volatility * inputs.volatility
    ) * inputs.years_to_target_date
    m = log(inputs.stock_price) + drift

    for i, t in enumerate(profit_range):
        prob = 0.0

        if t != [(0.0, 0.0)]:
            for p_range in t:
                lval = log(p_range[0]) if p_range[0] > 0.0 else -float("inf")
                hval = log(p_range[1])
                prob += stats.norm.cdf((hval - m) / sigma) - stats.norm.cdf(
                    (lval - m) / sigma
                )

        if i == 0:
            probability_of_reaching_target = prob
        else:
            probability_of_missing_target = prob

    return probability_of_reaching_target, probability_of_missing_target


def _compute_expected_returns_bs(
    s: np.ndarray,
    profit: np.ndarray,
    inputs: BlackScholesModelInputs,
    target: float = 0.01,
) -> tuple[float, float]:
    """
    Computes the expected returns above or below the target assuming the Black-Scholes
    model.

    ### Parameters

    `s`: array of stock prices.

    `profit`: array of profits and losses.

    `inputs`: input data used to estimate the probability of profit.

    `target`: target return.

    ### Returns

    Expected value above the target and expected value below the target.
    """

    expected_return_above_target: float
    expected_return_below_target: float

    sigma = (
        inputs.volatility * sqrt(inputs.years_to_target_date)
        if inputs.volatility > 0.0
        else 1e-10
    )
    drift = (
        inputs.interest_rate
        - inputs.dividend_yield
        - 0.5 * inputs.volatility * inputs.volatility
    ) * inputs.years_to_target_date
    m = log(inputs.stock_price) + drift

    lower_prices = []
    upper_prices = []
    slopes = []
    intercepts = []
    is_above_target = []

    def add_interval(
        lower_price: float,
        upper_price: float,
        slope: float,
        intercept: float,
        is_above: bool,
    ) -> None:
        if upper_price <= lower_price:
            return

        lower_prices.append(lower_price)
        upper_prices.append(upper_price)
        slopes.append(slope)
        intercepts.append(intercept)
        is_above_target.append(is_above)

    def accumulate_segment(
        lower_price: float,
        upper_price: float,
        slope: float,
        intercept: float,
    ) -> None:
        if upper_price <= lower_price:
            return

        lower_profit = slope * lower_price + intercept
        upper_profit = (
            slope * upper_price + intercept if np.isfinite(upper_price) else None
        )

        if slope == 0.0:
            intervals = [(lower_price, upper_price, lower_profit >= target)]
        elif np.isinf(upper_price):
            if slope > 0.0:
                if lower_profit >= target:
                    intervals = [(lower_price, upper_price, True)]
                else:
                    cross_price = (target - intercept) / slope
                    intervals = [
                        (lower_price, cross_price, False),
                        (cross_price, upper_price, True),
                    ]
            elif lower_profit < target:
                intervals = [(lower_price, upper_price, False)]
            else:
                cross_price = (target - intercept) / slope
                intervals = [
                    (lower_price, cross_price, True),
                    (cross_price, upper_price, False),
                ]
        else:
            assert upper_profit is not None
            if (lower_profit >= target and upper_profit >= target) or (
                lower_profit < target and upper_profit < target
            ):
                intervals = [(lower_price, upper_price, lower_profit >= target)]
            else:
                cross_price = (target - intercept) / slope
                if lower_profit < target:
                    intervals = [
                        (lower_price, cross_price, False),
                        (cross_price, upper_price, True),
                    ]
                else:
                    intervals = [
                        (lower_price, cross_price, True),
                        (cross_price, upper_price, False),
                    ]

        for interval_lower, interval_upper, is_above in intervals:
            add_interval(interval_lower, interval_upper, slope, intercept, is_above)

    if s[0] > 0.0 and s.shape[0] > 1:
        left_slope = (profit[1] - profit[0]) / (s[1] - s[0])
        accumulate_segment(
            0.0,
            s[0],
            left_slope,
            profit[0] - left_slope * s[0],
        )

    for lower_price, upper_price, lower_profit, upper_profit in zip(
        s[:-1], s[1:], profit[:-1], profit[1:]
    ):
        slope = (upper_profit - lower_profit) / (upper_price - lower_price)
        accumulate_segment(
            lower_price,
            upper_price,
            slope,
            lower_profit - slope * lower_price,
        )

    if s.shape[0] > 1:
        right_slope = (profit[-1] - profit[-2]) / (s[-1] - s[-2])
    else:
        right_slope = 0.0

    accumulate_segment(
        s[-1],
        float("inf"),
        right_slope,
        profit[-1] - right_slope * s[-1],
    )

    interval_prob, interval_profit = _integrate_linear_profit_bs(
        np.asarray(lower_prices),
        np.asarray(upper_prices),
        np.asarray(slopes),
        np.asarray(intercepts),
        m,
        sigma,
    )
    is_above_target_array = np.asarray(is_above_target)

    sum_prob_above_target = interval_prob[is_above_target_array].sum()
    sum_prob_below_target = interval_prob[~is_above_target_array].sum()
    weighted_profit_above_target = interval_profit[is_above_target_array].sum()
    weighted_profit_below_target = interval_profit[~is_above_target_array].sum()

    expected_return_above_target = (
        round(weighted_profit_above_target / sum_prob_above_target, 2)
        if sum_prob_above_target > 0.0
        else 0.0
    )
    expected_return_below_target = (
        round(weighted_profit_below_target / sum_prob_below_target, 2)
        if sum_prob_below_target > 0.0
        else 0.0
    )

    return expected_return_above_target, expected_return_below_target


def _integrate_linear_profit_bs(
    lower_price: np.ndarray,
    upper_price: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    log_mean: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrates a linear profit function over a Black-Scholes stock-price interval.
    """

    lower_z = cast(np.ndarray, _lognormal_z(lower_price, log_mean, sigma))
    upper_z = cast(np.ndarray, _lognormal_z(upper_price, log_mean, sigma))
    probability = stats.norm.cdf(upper_z) - stats.norm.cdf(lower_z)

    lower_moment_z = cast(
        np.ndarray, _lognormal_z(lower_price, log_mean + sigma * sigma, sigma)
    )
    upper_moment_z = cast(
        np.ndarray, _lognormal_z(upper_price, log_mean + sigma * sigma, sigma)
    )
    first_moment = np.exp(log_mean + 0.5 * sigma * sigma) * (
        stats.norm.cdf(upper_moment_z) - stats.norm.cdf(lower_moment_z)
    )

    return probability, slope * first_moment + intercept * probability


def _lognormal_z(
    price: FloatOrNdarray, log_mean: float, sigma: float
) -> FloatOrNdarray:
    """Returns the normal z-score for a lognormal stock price boundary."""

    price_array = np.asarray(price)
    z = np.empty(price_array.shape)
    z[price_array <= 0.0] = -float("inf")
    z[np.isinf(price_array)] = float("inf")

    finite_positive = (price_array > 0.0) & np.isfinite(price_array)
    z[finite_positive] = (log(price_array[finite_positive]) - log_mean) / sigma

    return z.item() if z.shape == () else z


def _get_pop_array(
    inputs: ArrayInputs, target: float
) -> tuple[float, float, float, float]:
    """
    Estimates the probability of profit (PoP) of an options trading strategy using
    an array of terminal stock prices.

    ### Parameters

    `inputs`: input data used to estimate the probability of profit.

    `target`: target return.

    ### Returns

    Probability of reaching the target return, expected value above the target,
    probability of missing the target return, and expected value below the
    target.
    """

    if inputs.array.shape[0] == 0:
        raise ValueError("The array is empty!")

    tmp1 = inputs.array[inputs.array >= target]
    tmp2 = inputs.array[inputs.array < target]

    probability_of_reaching_target = tmp1.shape[0] / inputs.array.shape[0]
    probability_of_missing_target = 1.0 - probability_of_reaching_target

    expected_return_above_target = round(tmp1.mean(), 2) if tmp1.shape[0] > 0 else 0.0
    expected_return_below_target = round(tmp2.mean(), 2) if tmp2.shape[0] > 0 else 0.0

    return (
        probability_of_reaching_target,
        expected_return_above_target,
        probability_of_missing_target,
        expected_return_below_target,
    )


def _get_profit_range(
    s: np.ndarray, profit: np.ndarray, target: float = 0.01
) -> tuple[list[Range], list[Range]]:
    """
    Returns lists of stock price ranges: one representing the ranges where the
    options trade returns are equal to or greater than the target, and the other
    representing the ranges where they fall short.

    ### Parameters

    `s`: array of stock prices.

    `profit`: array of profits and losses.

    `target`: target profit.

    ### Returns

    Lists of stock price pairs.
    """

    profit_range = []
    loss_range = []

    crossings = _get_sign_changes(profit, target)
    n_crossings = len(crossings)

    if n_crossings == 0:
        if profit[0] >= target:
            return [(0.0, float("inf"))], [(0.0, 0.0)]
        else:
            return [(0.0, 0.0)], [(0.0, float("inf"))]

    lb_profit = hb_profit = None
    lb_loss = hb_loss = None

    for i, index in enumerate(crossings):
        if i == 0:
            if profit[index] < profit[index - 1]:
                lb_profit = 0.0
                hb_profit = s[index - 1]
                lb_loss = s[index]

                if n_crossings == 1:
                    hb_loss = float("inf")
            else:
                lb_profit = s[index]
                lb_loss = 0.0
                hb_loss = s[index - 1]

                if n_crossings == 1:
                    hb_profit = float("inf")
        elif i == n_crossings - 1:
            if profit[index] > profit[index - 1]:
                lb_profit = s[index]
                hb_profit = float("inf")
                hb_loss = s[index - 1]
            else:
                hb_profit = s[index - 1]
                lb_loss = s[index]
                hb_loss = float("inf")
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


def _get_sign_changes(profit: np.ndarray, target: float) -> list[int]:
    """
    Returns a list of the indices in the array of profits where the sign changes.

    ### Parameters

    `profit`: array of profits and losses.

    `target`: target profit.

    ### Returns

    List of indices.
    """

    p_temp = profit - target + 1e-10

    sign_changes = (np.sign(p_temp[:-1]) * np.sign(p_temp[1:])) < 0

    return list(np.where(sign_changes)[0] + 1)
