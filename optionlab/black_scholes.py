from __future__ import division

from scipy import stats
from numpy import exp, round, arange, abs, argmin, pi
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

    Parameters
    ----------
    s : float
        Stock price.
    x : float | numpy.ndarray
        Strike price(s).
    r : float
        Annualized risk-free interest rate.
    vol : float
        Annualized volatility.
    years_to_maturity : float
        Time remaining to maturity, in years.
    y : float, optional
        Annualized dividend yield. The default is 0.0.

    Returns
    -------
    BlackScholesInfo
        Information calculated using the Black-Scholes formula. See the documentation
        for `BlackScholesInfo`.
    """

    d1 = get_d1(s, x, r, vol, years_to_maturity, y)
    d2 = get_d2(s, x, r, vol, years_to_maturity, y)
    call_price = get_option_price("call", s, x, r, years_to_maturity, d1, d2, y)
    put_price = get_option_price("put", s, x, r, years_to_maturity, d1, d2, y)
    call_delta = get_delta("call", d1, years_to_maturity, y)
    put_delta = get_delta("put", d1, years_to_maturity, y)
    call_theta = get_theta("call", s, x, r, vol, years_to_maturity, d1, d2, y)
    put_theta = get_theta("put", s, x, r, vol, years_to_maturity, d1, d2, y)
    gamma = get_gamma(s, vol, years_to_maturity, d1, y)
    vega = get_vega(s, years_to_maturity, d1, y)
    call_rho = get_rho("call", x, r, years_to_maturity, d2)
    put_rho = get_rho("put", x, r, years_to_maturity, d2)
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
        call_rho=call_rho,
        put_rho=put_rho,
        call_itm_prob=call_itm_prob,
        put_itm_prob=put_itm_prob,
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

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either **call** or **put**.
    s0 : float | numpy.ndarray
        Spot price(s) of the underlying asset.
    x : float | numpy.ndarray
        Strike price(s).
    r : float
        Annualize risk-free interest rate.
    years_to_maturity : float
        Time remaining to maturity, in years.
    d1 : float | numpy.ndarray
        `d1` in Black-Scholes formula.
    d2 : float | numpy.ndarray
        `d2` in Black-Scholes formula.
    y : float, optional
        Annualized dividend yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
        Option price(s).
    """

    s = s0 * exp(-y * years_to_maturity)

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
    option_type: OptionType,
    d1: FloatOrNdarray,
    years_to_maturity: float,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the option's Greek Delta.

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either **call** or **put**.
    d1 : float | numpy.ndarray
        `d1` in Black-Scholes formula.
    years_to_maturity : float
        Time remaining to maturity, in years.
    y : float, optional
        Annualized dividend yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
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

    Parameters
    ----------
    s0 : float
        Spot price of the underlying asset.
    vol : float
        Annualized volatitily.
    years_to_maturity : float
        Time remaining to maturity, in years.
    d1 : float | numpy.ndarray
        `d1` in Black-Scholes formula.
    y : float, optional
        Annualized divident yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
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

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either **call** or **put**.
    s0 : float
        Spot price of the underlying asset.
    x : float | numpy.ndarray
        Strike price(s).
    r : float
        Annualized risk-free interest rate.
    vol : float
        Annualized volatility.
    years_to_maturity : float
        Time remaining to maturity, in years.
    d1 : float | numpy.ndarray
        `d1` in Black-Scholes formula.
    d2 : float | numpy.ndarray
        `d2` in Black-Scholes formula.
    y : float, optional
        Annualized dividend yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
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

    Parameters
    ----------
    s0 : float
        Spot price of the underlying asset.
    years_to_maturity : float
        Time remaining to maturity, in years.
    d1 : float | numpy.ndarray
        `d1` in Black-Scholes formula.
    y : float, optional
        Annualized dividend yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
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

    Parameters
    ----------
    option_type : OptionType
        `OptionType` literal value, which must be either **call** or **put**.
    x : float | numpy.ndarray
        Strike price(s).
    r : float
        Annualized risk-free interest rate.
    years_to_maturity : float
        Time remaining to maturity, in years.
    d2 : float | numpy.ndarray
        `d2` in Black-Scholes formula.

    Returns
    -------
    float | numpy.ndarray
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

    Parameters
    ----------
    s0 : float | numpy.ndarray
        Spot price(s) of the underlying asset.
    x : float | numpy.ndarray
        Strike price(s).
    r : float
        Annualized risk-free interest rate.
    vol : float | numpy.ndarray
        Annualized volatility(ies).
    years_to_maturity : float
        Time remaining to maturity, in years.
    y : float, optional
        Annualized divident yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
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

    Parameters
    ----------
    s0 : float | numpy.ndarray
        Spot price(s) of the underlying asset.
    x : float | numpy.ndarray
        Strike price(s).
    r : float
        Annualized risk-free interest rate.
    vol : float | numpy.ndarray
        Annualized volatility(ies).
    years_to_maturity : float
        Time remaining to maturity, in years.
    y : float, optional
        Annualized divident yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
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

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either **call** or **put**.
    oprice : float
        Market price of an option.
    s0 : float
        Spot price of the underlying asset.
    x : float
        Strike price.
    r : float
        Annualized risk-free interest rate.
    years_to_maturity : float
        Time remaining to maturity, in years.
    y : float, optional
        Annualized dividend yield. The default is 0.0.

    Returns
    -------
    float
        Implied volatility of the option.
    """

    vol = 0.001 * arange(1, 1001)
    d1 = get_d1(s0, x, r, vol, years_to_maturity, y)
    d2 = get_d2(s0, x, r, vol, years_to_maturity, y)
    dopt = abs(
        get_option_price(option_type, s0, x, r, years_to_maturity, d1, d2, y) - oprice
    )

    return vol[argmin(dopt)]


def get_itm_probability(
    option_type: OptionType,
    d2: FloatOrNdarray,
    years_to_maturity: float,
    y: float = 0.0,
) -> FloatOrNdarray:
    """
    Returns the In-The-Money probability of an option.

    Parameters
    ----------
    option_type : str
        `OptionType` literal value, which must be either **call** or **put**.
    d2 : float | numpy.ndarray
        `d2` in Black-Scholes formula.
    years_to_maturity : float
        Time remaining to maturity, in years.
    y : float, optional
        Annualized dividend yield. The default is 0.0.

    Returns
    -------
    float | numpy.ndarray
        In-The-Money probability(ies).
    """

    yfac = exp(-y * years_to_maturity)

    if option_type == "call":
        return yfac * stats.norm.cdf(d2)
    elif option_type == "put":
        return yfac * stats.norm.cdf(-d2)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")
