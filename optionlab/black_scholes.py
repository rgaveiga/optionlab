"""
This module defines functions that calculate quantities, such as option prices
and the Greeks, related to the Black-Scholes model.
"""

from __future__ import division

from typing import cast

from scipy import optimize, stats
from scipy.special import ndtr
from numpy import exp, isscalar, pi, where
from numpy.lib.scimath import log, sqrt

from optionlab.models import BlackScholesInfo, OptionType, FloatOrNdarray


def get_bs_info(
    s: float,
    x: FloatOrNdarray,
    r: float,
    vol: float,
    years_to_maturity: float,
    y: float = 0.0,
) -> BlackScholesInfo:
    """
    Provides information about call and put options calculated using the Black-Scholes
    formula.

    ### Parameters

    `s`: stock price.

    `x`: strike price(s).

    `r`: annualized risk-free interest rate.

    `vol`: annualized volatility.

    `years_to_maturity`: time remaining to maturity, in years.

    `y`: annualized dividend yield.

    ### Returns

    Information calculated using the Black-Scholes formula.
    """

    sqrt_time = sqrt(years_to_maturity)
    discount_y = exp(-y * years_to_maturity)
    discount_r = exp(-r * years_to_maturity)
    dividend_adjusted_spot = s * discount_y

    d1 = (log(s / x) + (r - y + vol * vol / 2.0) * years_to_maturity) / (
        vol * sqrt_time
    )
    d2 = d1 - vol * sqrt_time

    cdf_d1 = ndtr(d1)
    cdf_d2 = ndtr(d2)
    cdf_minus_d1 = ndtr(-d1)
    cdf_minus_d2 = ndtr(-d2)
    pdf_d1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    discounted_strike = x * discount_r
    call_price = dividend_adjusted_spot * cdf_d1 - discounted_strike * cdf_d2
    put_price = discounted_strike * cdf_minus_d2 - dividend_adjusted_spot * cdf_minus_d1

    call_delta = discount_y * cdf_d1
    put_delta = discount_y * (cdf_d1 - 1.0)

    theta_decay = dividend_adjusted_spot * vol * pdf_d1 / (2.0 * sqrt_time)
    call_theta = -(
        theta_decay
        + r * discounted_strike * cdf_d2
        - y * dividend_adjusted_spot * cdf_d1
    )
    put_theta = -(
        theta_decay
        - r * discounted_strike * cdf_minus_d2
        + y * dividend_adjusted_spot * cdf_minus_d1
    )

    gamma = discount_y * pdf_d1 / (s * vol * sqrt_time)
    vega = dividend_adjusted_spot * pdf_d1 * sqrt_time / 100

    rho_factor = x * years_to_maturity * discount_r / 100
    call_rho = rho_factor * cdf_d2
    put_rho = -rho_factor * cdf_minus_d2

    call_itm_prob = discount_y * cdf_d2
    put_itm_prob = discount_y * cdf_minus_d2

    mu = (r - y - 0.5 * vol * vol) / (vol * vol)
    lam = sqrt((mu * mu) + 2.0 * r / (vol * vol))
    sigma = vol * sqrt_time
    z = log(x / s) / sigma + lam * sigma
    exp1 = mu + lam
    exp2 = mu - lam
    ratio = x / s
    touch_high = ratio**exp1
    touch_low = ratio**exp2

    call_prob_of_touch_value = touch_high * ndtr(-z) + touch_low * ndtr(
        2.0 * lam * sigma - z
    )
    put_prob_of_touch_value = touch_high * ndtr(z) + touch_low * ndtr(
        z - 2.0 * lam * sigma
    )

    if isscalar(x):
        scalar_x = cast(float, x)
        call_prob_of_touch = 1.0 if s >= scalar_x else call_prob_of_touch_value
        put_prob_of_touch = 1.0 if s <= scalar_x else put_prob_of_touch_value
    else:
        call_prob_of_touch = where(s >= x, 1.0, call_prob_of_touch_value)
        put_prob_of_touch = where(s <= x, 1.0, put_prob_of_touch_value)

    return BlackScholesInfo(
        call_price=call_price,
        put_price=put_price,
        call_delta=call_delta,
        put_delta=put_delta,
        call_theta=call_theta,
        put_theta=put_theta,
        gamma=gamma,
        vega=vega,
        call_rho=call_rho,
        put_rho=put_rho,
        call_itm_prob=call_itm_prob,
        put_itm_prob=put_itm_prob,
        call_prob_of_touch=call_prob_of_touch,
        put_prob_of_touch=put_prob_of_touch,
    )


def get_option_price(
    option_type: OptionType,
    s0: FloatOrNdarray,
    x: FloatOrNdarray,
    r: float,
    years_to_maturity: float,
    d1: FloatOrNdarray,
    d2: FloatOrNdarray,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the price of an option.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `s0`: spot price(s) of the underlying asset.

    `x`: strike price(s).

    `r`: annualize risk-free interest rate.

    `years_to_maturity`: time remaining to maturity, in years.

    `d1`: `d1` in Black-Scholes formula.

    `d2`: `d2` in Black-Scholes formula.

    `y`: annualized dividend yield.

    ### Returns

    Option price(s).
    """

    s = s0 * exp(-y * years_to_maturity)

    if option_type == "call":
        return s * stats.norm.cdf(d1) - x * exp(
            -r * years_to_maturity
        ) * stats.norm.cdf(d2)
    elif option_type == "put":
        return x * exp(-r * years_to_maturity) * stats.norm.cdf(
            -d2
        ) - s * stats.norm.cdf(-d1)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def get_delta(
    option_type: OptionType,
    d1: FloatOrNdarray,
    years_to_maturity: float,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the option's Greek Delta.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `d1`: `d1` in Black-Scholes formula.

    `years_to_maturity`: time remaining to maturity, in years.

    `y`: annualized dividend yield.

    ### Returns

    Option's Greek Delta.
    """

    yfac = exp(-y * years_to_maturity)

    if option_type == "call":
        return yfac * stats.norm.cdf(d1)
    elif option_type == "put":
        return yfac * (stats.norm.cdf(d1) - 1.0)
    else:
        raise ValueError("Option must be either 'call' or 'put'!")


def get_gamma(
    s0: float,
    vol: float,
    years_to_maturity: float,
    d1: FloatOrNdarray,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the option's Greek Gamma.

    ### Parameters

    `s0`: spot price of the underlying asset.

    `vol`: annualized volatitily.

    `years_to_maturity`: time remaining to maturity, in years.

    `d1`: `d1` in Black-Scholes formula.

    `y`: annualized divident yield.

    ### Returns

    Option's Greek Gamma.
    """

    yfac = exp(-y * years_to_maturity)

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    return yfac * cdf_d1_prime / (s0 * vol * sqrt(years_to_maturity))


def get_theta(
    option_type: OptionType,
    s0: float,
    x: FloatOrNdarray,
    r: float,
    vol: float,
    years_to_maturity: float,
    d1: FloatOrNdarray,
    d2: FloatOrNdarray,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the option's Greek Theta.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `s0`: spot price of the underlying asset.

    `x`: strike price(s).

    `r`: annualized risk-free interest rate.

    `vol`: annualized volatility.

    `years_to_maturity`: time remaining to maturity, in years.

    `d1`: `d1` in Black-Scholes formula.

    `d2`: `d2` in Black-Scholes formula.

    `y`: annualized dividend yield.

    ### Returns

    Option's Greek Theta.
    """

    s = s0 * exp(-y * years_to_maturity)

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    if option_type == "call":
        return -(
            s * vol * cdf_d1_prime / (2.0 * sqrt(years_to_maturity))
            + r * x * exp(-r * years_to_maturity) * stats.norm.cdf(d2)
            - y * s * stats.norm.cdf(d1)
        )
    elif option_type == "put":
        return -(
            s * vol * cdf_d1_prime / (2.0 * sqrt(years_to_maturity))
            - r * x * exp(-r * years_to_maturity) * stats.norm.cdf(-d2)
            + y * s * stats.norm.cdf(-d1)
        )
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def get_vega(
    s0: float,
    years_to_maturity: float,
    d1: FloatOrNdarray,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the option's Greek Vega.

    ### Parameters

    `s0`: spot price of the underlying asset.

    `years_to_maturity`: time remaining to maturity, in years.

    `d1`: `d1` in Black-Scholes formula.

    `y`: annualized dividend yield.

    ### Returns

    Option's Greek Vega.
    """

    s = s0 * exp(-y * years_to_maturity)

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    return s * cdf_d1_prime * sqrt(years_to_maturity) / 100


def get_rho(
    option_type: OptionType,
    x: FloatOrNdarray,
    r: float,
    years_to_maturity: float,
    d2: FloatOrNdarray,
) -> FloatOrNdarray:
    """
    Returns the option's Greek Rho.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `x`: strike price(s).

    `r`: annualized risk-free interest rate.

    `years_to_maturity`: time remaining to maturity, in years.

    `d2`: `d2` in Black-Scholes formula.

    ### Returns

    Option's Greek Rho.
    """

    if option_type == "call":
        return (
            x
            * years_to_maturity
            * exp(-r * years_to_maturity)
            * stats.norm.cdf(d2)
            / 100
        )
    elif option_type == "put":
        return (
            -x
            * years_to_maturity
            * exp(-r * years_to_maturity)
            * stats.norm.cdf(-d2)
            / 100
        )
    else:
        raise ValueError("Option must be either 'call' or 'put'!")


def get_d1(
    s0: FloatOrNdarray,
    x: FloatOrNdarray,
    r: float,
    vol: FloatOrNdarray,
    years_to_maturity: float,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns `d1` used in Black-Scholes formula.

    ### Parameters

    `s0`: spot price(s) of the underlying asset.

    `x`: strike price(s).

    `r`: annualized risk-free interest rate.

    `vol`: annualized volatility(ies).

    `years_to_maturity`: time remaining to maturity, in years.

    `y`: annualized divident yield.

    ### Returns

    `d1` in Black-Scholes formula.
    """

    return (log(s0 / x) + (r - y + vol * vol / 2.0) * years_to_maturity) / (
        vol * sqrt(years_to_maturity)
    )


def get_d2(
    s0: FloatOrNdarray,
    x: FloatOrNdarray,
    r: float,
    vol: FloatOrNdarray,
    years_to_maturity: float,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns `d2` used in Black-Scholes formula.

    ### Parameters

    `s0`: spot price(s) of the underlying asset.

    `x`: strike price(s).

    `r`: annualized risk-free interest rate.

    `vol`: annualized volatility(ies).

    `years_to_maturity`: time remaining to maturity, in years.

    `y`: annualized divident yield.

    ### Returns

    `d2` in Black-Scholes formula.
    """

    return (log(s0 / x) + (r - y - vol * vol / 2.0) * years_to_maturity) / (
        vol * sqrt(years_to_maturity)
    )


def get_implied_vol(
    option_type: OptionType,
    oprice: float,
    s0: float,
    x: float,
    r: float,
    years_to_maturity: float,
    y: float = 0.0,
) -> float:
    """
    Returns the implied volatility of an option.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `oprice`: market price of an option.

    `s0`: spot price of the underlying asset.

    `x`: strike price.

    `r`: annualized risk-free interest rate.

    `years_to_maturity`: time remaining to maturity, in years.

    `y`: annualized dividend yield.

    ### Returns

    Option's implied volatility.
    """

    min_vol = 0.001
    max_vol = 1.0

    def price_diff(vol: float) -> float:
        d1 = get_d1(s0, x, r, vol, years_to_maturity, y)
        d2 = get_d2(s0, x, r, vol, years_to_maturity, y)
        return float(
            get_option_price(option_type, s0, x, r, years_to_maturity, d1, d2, y)
            - oprice
        )

    min_diff = price_diff(min_vol)
    max_diff = price_diff(max_vol)

    if min_diff >= 0.0:
        return min_vol
    if max_diff <= 0.0:
        return max_vol

    return float(optimize.brentq(price_diff, min_vol, max_vol, xtol=5e-7, maxiter=50))


def get_itm_probability(
    option_type: OptionType,
    d2: FloatOrNdarray,
    years_to_maturity: float,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the probability(ies) that the option(s) will expire in-the-money (ITM).

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `d2`: `d2` in Black-Scholes formula.

    `years_to_maturity`: time remaining to maturity, in years.

    `y`: annualized dividend yield.

    ### Returns

    Probability(ies) that the option(s) will expire in-the-money (ITM).
    """

    yfac = exp(-y * years_to_maturity)

    if option_type == "call":
        return yfac * stats.norm.cdf(d2)
    elif option_type == "put":
        return yfac * stats.norm.cdf(-d2)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def get_probability_of_touch(
    option_type: OptionType,
    s: float,
    x: FloatOrNdarray,
    r: float,
    vol: float,
    years_to_maturity: float,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the probability(ies) that the option(s) will ever get in-the-money (ITM)
    before expiration.

    > [!NOTE]
    > This function implements equations 2.66 and 2.67 (see pages 80 and 81) in
    > *The complete guide to option pricing formulas*, 2nd edition, authored by
    > Espen Gaarder Haug, PhD, and published by McGraw-Hill.

    ### Parameters

    `option_type`: either *'call'* or *'put'*.

    `s`: stock price.

    `x`: strike price(s).

    `r`: annualized risk-free interest rate.

    `vol`: annualized volatility.

    `years_to_maturity`: time remaining to maturity, in years.

    `y`: annualized dividend yield.

    ### Returns

    Probability(ies) that the option(s) will ever get in-the-money (ITM) before
    expiration.
    """

    mu = (r - y - 0.5 * vol * vol) / (vol * vol)
    lam = sqrt((mu * mu) + 2.0 * r / (vol * vol))
    sigma = vol * sqrt(years_to_maturity)
    z = log(x / s) / sigma + lam * sigma
    exp1 = mu + lam
    exp2 = mu - lam

    if option_type == "call":
        if s >= x:
            return 1.0
        else:
            return ((x / s) ** exp1) * stats.norm.cdf(-z) + (
                (x / s) ** exp2
            ) * stats.norm.cdf(2.0 * lam * sigma - z)
    elif option_type == "put":
        if s <= x:
            return 1.0
        else:
            return ((x / s) ** exp1) * stats.norm.cdf(z) + (
                (x / s) ** exp2
            ) * stats.norm.cdf(z - 2.0 * lam * sigma)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")
