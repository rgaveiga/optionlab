from __future__ import division

import numpy as np
from scipy import stats
from numpy import exp, round, arange, abs, argmin, pi
from numpy.lib.scimath import log, sqrt

from optionlab.models import BlackScholesInfo, OptionType


def get_bs_info(
    s: float, x: float, r: float, vol: float, years_to_maturity: float, y: float = 0.0
) -> BlackScholesInfo:
    """
    get_bs_info(s, x, r, vol, years_to_maturity, y) -> provides informaton about call and
    put options using the Black-Scholes formula, taking the current stock price
    's', the option strike 'x', the annualized risk-free rate 'r', the annualized
    volatility 'vol', the time remaining to maturity in units of year, and
    the annualized stock's dividend yield 'y' as arguments.
    """
    d1, d2 = get_d1_d2(s, x, r, vol, years_to_maturity, y)
    call_price = get_option_price("call", s, x, r, years_to_maturity, d1, d2, y)
    put_price = get_option_price("put", s, x, r, years_to_maturity, d1, d2, y)
    call_delta = get_delta("call", d1, years_to_maturity, y)
    put_delta = get_delta("put", d1, years_to_maturity, y)
    call_theta = get_theta("call", s, x, r, vol, years_to_maturity, d1, d2, y)
    put_theta = get_theta("put", s, x, r, vol, years_to_maturity, d1, d2, y)
    gamma = get_gamma(s, vol, years_to_maturity, d1, y)
    vega = get_vega(s, years_to_maturity, d1, y)
    call_itm_prob = get_itm_probability("call", d2, years_to_maturity, y)
    put_itm_prob = get_itm_probability("put", d2, years_to_maturity, y)

    return BlackScholesInfo(
        call_price=call_price,
        put_price=put_price,
        call_delta=call_delta,
        put_delta=put_delta,
        call_theta=call_theta,
        put_theta=put_theta,
        gamma=gamma,
        vega=vega,
        call_itm_prob=call_itm_prob,
        put_itm_prob=put_itm_prob,
    )


def get_option_price(
    option_type: OptionType,
    s0: np.ndarray | float,
    x: np.ndarray | float,
    r: float,
    years_to_maturity: float,
    d1: float,
    d2: float,
    y: float = 0.0,
) -> float:
    """
    get_option_price(option_type, s0, x, r, years_to_maturity, d1, d2, y) -> returns the price of
    an option (call or put) given the current stock price 's0' and the option
    strike 'x', as well as the annualized risk-free rate 'r', the time remaining
    to maturity in units of year, 'd1' and 'd2' as defined in the Black-Scholes
    formula, and the stocks's annualized dividend yield 'y' (default is zero,
    i.e., the stock does not pay dividends).
    """
    if y > 0.0:
        s = s0 * exp(-y * years_to_maturity)
    else:
        s = s0

    if option_type == "call":
        return round(
            s * stats.norm.cdf(d1)
            - x * exp(-r * years_to_maturity) * stats.norm.cdf(d2),
            2,
        )
    elif option_type == "put":
        return round(
            x * exp(-r * years_to_maturity) * stats.norm.cdf(-d2)
            - s * stats.norm.cdf(-d1),
            2,
        )
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def get_delta(
    option_type: OptionType, d1: float, years_to_maturity: float = 0.0, y: float = 0.0
) -> float:
    """
    get_delta(option_type, d1, years_to_maturity, y) -> computes the Greek Delta for an option
    (call or put) taking 'd1' as defined in the Black-Scholes formula as a mandatory
    argument. Optionally, the time remaining to maturity in units of year and
    the stocks's annualized dividend yield 'y' (default is zero,i.e., the stock
    does not pay dividends) may be passed as arguments. The Greek Delta estimates
    how the option price varies as the stock price increases or decreases by $1.
    """
    if y > 0.0 and years_to_maturity > 0.0:
        yfac = exp(-y * years_to_maturity)
    else:
        yfac = 1.0

    if option_type == "call":
        return yfac * stats.norm.cdf(d1)
    elif option_type == "put":
        return yfac * (stats.norm.cdf(d1) - 1.0)
    else:
        raise ValueError("Option must be either 'call' or 'put'!")


def get_gamma(
    s0: float, vol: float, years_to_maturity: float, d1: float, y: float = 0.0
) -> float:
    """
    get_gamma(s0, vol, years_to_maturity, d1, y) -> computes the Greek Gamma for an option
    taking the current stock price 's0', the annualized volatity 'vol', the time
    remaining to maturity in units of year, 'd1' as defined in the Black-Scholes
    formula and the stocks's annualized dividend yield 'y' (default is zero,i.e.,
    the stock does not pay dividends) as arguments. The Greek Gamma provides the
    variation of Greek Delta as stock price increases or decreases by $1.
    """
    if y > 0.0:
        yfac = exp(-y * years_to_maturity)
    else:
        yfac = 1.0

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    return yfac * cdf_d1_prime / (s0 * vol * sqrt(years_to_maturity))


def get_theta(
    option_type: OptionType,
    s0: float,
    x: np.ndarray | float,
    r: float,
    vol: float,
    years_to_maturity: float,
    d1: float,
    d2: float,
    y: float = 0.0,
) -> float:
    """
    get_theta(option_type, s0, x, r, vol, years_to_maturity, d1, d2, y) -> computes the Greek Theta
    for an option (call or put) taking the current stock price 's0', the exercise
    price 'x', the annualized risk-free rate 'r', the time remaining to maturity
    in units of year , the annualized volatility 'vol', 'd1' and 'd2' as defined
    in the Black-Scholes formula, and the stocks's annualized dividend yield 'y'
    (default is zero, i.e., the stock does not pay dividends) as arguments. The
    Greek Theta estimates the value lost per year of an option as the maturity
    gets closer.
    """
    if y > 0.0:
        s = s0 * exp(-y * years_to_maturity)
    else:
        s = s0

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


def get_vega(s0: float, years_to_maturity: float, d1: float, y: float = 0.0) -> float:
    """
    get_vega(s0, years_to_maturity, d1) -> computes the Greek Vega for an option taking
    the current stock price 's0', the time remaining to maturity in units of year,
    'd1' as defined in the Black-Scholes formula, and the stocks's annualized
    dividend yield 'y' (default is zero, i.e., the stock does not pay dividends)
    as arguments. The Greek Vega estimates the amount that the option price changes
    for every 1% change in the annualized volatility of the underlying asset.
    """
    if y > 0.0:
        s = s0 * exp(-y * years_to_maturity)
    else:
        s = s0

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    return s * cdf_d1_prime * sqrt(years_to_maturity) / 100


def get_d1_d2(
    s0: np.ndarray | float,
    x: np.ndarray | float,
    r: float,
    vol: float | np.ndarray,
    years_to_maturity: float,
    y: float = 0.0,
) -> tuple[float, float]:
    """
    get_d1_d2(s0, x, r, vol, years_to_maturity, y) -> returns 'd1' and 'd2' taking the
    current stock price 's0', the exercise price 'x', the annualized risk-free
    rate 'r', the annualized volatility 'vol', the time remaining to option
    expiration in units of year, and the stocks's annualized dividend yield 'y'
    (default is zero, i.e., the stock does not pay dividends) as arguments.
    """
    d1 = (log(s0 / x) + (r - y + vol * vol / 2.0) * years_to_maturity) / (
        vol * sqrt(years_to_maturity)
    )
    d2 = d1 - vol * sqrt(years_to_maturity)

    return d1, d2


def get_implied_vol(
    option_type: OptionType,
    oprice: float,
    s0: float,
    x: float,
    r: float,
    years_to_maturity: float,
    y: float = 0.0,
) -> np.ndarray:
    """
    get_implied_vol(option_type, oprice, s0, x, r, years_to_maturity, y) -> estimates the implied
    volatility taking the option type (call or put), the option price, the current
    stock price 's0', the option strike 'x', the annualized risk-free rate 'r',
    the time remaining to maturity in units of year, and the stocks's annualized
    dividend yield 'y' (default is zero,i.e., the stock does not pay dividends)
    as arguments.
    """
    vol = 0.001 * arange(1, 1001)
    d1, d2 = get_d1_d2(s0, x, r, vol, years_to_maturity, y)
    dopt = abs(
        get_option_price(option_type, s0, x, r, years_to_maturity, d1, d2, y) - oprice
    )

    return vol[argmin(dopt)]


def get_itm_probability(
    option_type: OptionType, d2: float, years_to_maturity: float = 0.0, y: float = 0.0
) -> float:
    """
    get_itm_probability(option_type, d2, years_to_maturity, y) -> returns the estimated probability
    that an option (either call or put) will be in-the-money at maturity, taking
    'd2' as defined in the Black-Scholes formula as a mandatory argument. Optionally,
    the time remaining to maturity in units of year and the stocks's annualized
    dividend yield 'y' (default is zero,i.e., the stock does not pay dividends)
    may be passed as arguments.
    """
    if y > 0.0 and years_to_maturity > 0.0:
        yfac = exp(-y * years_to_maturity)
    else:
        yfac = 1.0

    if option_type == "call":
        return yfac * stats.norm.cdf(d2)
    elif option_type == "put":
        return yfac * stats.norm.cdf(-d2)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")
