from __future__ import division
from scipy import stats
from numpy import exp, round, arange, abs, argmin, pi
from numpy.lib.scimath import log, sqrt


def getoptionprice(optype, s0, x, r, time2maturity, d1, d2, y=0.0):
    """
    getoptionprice(optype,s0,x,r,time2maturity,d1,d2,y) -> returns the price of
    an option (call or put) given the current stock price 's0' and the option
    strike 'x', as well as the annualized risk-free rate 'r', the time remaining
    to maturity in units of year, 'd1' and 'd2' as defined in the Black-Scholes
    formula, and the stocks's annualized dividend yield 'y' (default is zero,
    i.e., the stock does not pay dividends).
    """
    if y > 0.0:
        s = s0 * exp(-y * time2maturity)
    else:
        s = s0

    if optype == "call":
        return round(
            s * stats.norm.cdf(d1) - x * exp(-r * time2maturity) * stats.norm.cdf(d2), 2
        )
    elif optype == "put":
        return round(
            x * exp(-r * time2maturity) * stats.norm.cdf(-d2) - s * stats.norm.cdf(-d1),
            2,
        )
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def getimpliedvol(optype, oprice, s0, x, r, time2maturity, y=0.0):
    """
    getimpliedvol(optype,oprice,s0,x,r,time2maturity,y) -> estimates the implied
    volatility taking the option type (call or put), the option price, the current
    stock price 's0', the option strike 'x', the annualized risk-free rate 'r',
    the time remaining to maturity in units of year, and the stocks's annualized
    dividend yield 'y' (default is zero,i.e., the stock does not pay dividends)
    as arguments.
    """
    vol = 0.001 * arange(1, 1001)
    d1, d2 = get_d1_d2(s0, x, r, vol, time2maturity, y)
    dopt = abs(getoptionprice(optype, s0, x, r, time2maturity, d1, d2, y) - oprice)

    return vol[argmin(dopt)]


def getdelta(optype, d1, time2maturity=0.0, y=0.0):
    """
    getdelta(optype,d1,time2maturity,y) -> computes the Greek Delta for an option
    (call or put) taking 'd1' as defined in the Black-Scholes formula as a mandatory
    argument. Optionally, the time remaining to maturity in units of year and
    the stocks's annualized dividend yield 'y' (default is zero,i.e., the stock
    does not pay dividends) may be passed as arguments. The Greek Delta estimates
    how the option price varies as the stock price increases or decreases by $1.
    """
    if y > 0.0 and time2maturity > 0.0:
        yfac = exp(-y * time2maturity)
    else:
        yfac = 1.0

    if optype == "call":
        return yfac * stats.norm.cdf(d1)
    elif optype == "put":
        return yfac * (stats.norm.cdf(d1) - 1.0)
    else:
        raise ValueError("Option must be either 'call' or 'put'!")


def getgamma(s0, vol, time2maturity, d1, y=0.0):
    """
    getgamma(s0,vol,time2maturity,d1,y) -> computes the Greek Gamma for an option
    taking the current stock price 's0', the annualized volatity 'vol', the time
    remaining to maturity in units of year, 'd1' as defined in the Black-Scholes
    formula and the stocks's annualized dividend yield 'y' (default is zero,i.e.,
    the stock does not pay dividends) as arguments. The Greek Gamma provides the
    variation of Greek Delta as stock price increases or decreases by $1.
    """
    if y > 0.0:
        yfac = exp(-y * time2maturity)
    else:
        yfac = 1.0

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    return yfac * cdf_d1_prime / (s0 * vol * sqrt(time2maturity))


def gettheta(optype, s0, x, r, vol, time2maturity, d1, d2, y=0.0):
    """
    gettheta(optype,s0,x,r,vol,time2maturity,d1,d2,y) -> computes the Greek Theta
    for an option (call or put) taking the current stock price 's0', the exercise
    price 'x', the annualized risk-free rate 'r', the time remaining to maturity
    in units of year , the annualized volatility 'vol', 'd1' and 'd2' as defined
    in the Black-Scholes formula, and the stocks's annualized dividend yield 'y'
    (default is zero, i.e., the stock does not pay dividends) as arguments. The
    Greek Theta estimates the value lost per year of an option as the maturity
    gets closer.
    """
    if y > 0.0:
        s = s0 * exp(-y * time2maturity)
    else:
        s = s0

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    if optype == "call":
        return -(
            s * vol * cdf_d1_prime / (2.0 * sqrt(time2maturity))
            + r * x * exp(-r * time2maturity) * stats.norm.cdf(d2)
            - y * s * stats.norm.cdf(d1)
        )
    elif optype == "put":
        return -(
            s * vol * cdf_d1_prime / (2.0 * sqrt(time2maturity))
            - r * x * exp(-r * time2maturity) * stats.norm.cdf(-d2)
            + y * s * stats.norm.cdf(-d1)
        )
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def getvega(s0, time2maturity, d1, y=0.0):
    """
    getvega(s0,time2maturity,d1) -> computes the Greek Vega for an option taking
    the current stock price 's0', the time remaining to maturity in units of year,
    'd1' as defined in the Black-Scholes formula, and the stocks's annualized
    dividend yield 'y' (default is zero, i.e., the stock does not pay dividends)
    as arguments. The Greek Vega estimates the amount that the option price changes
    for every 1% change in the annualized volatility of the underlying asset.
    """
    if y > 0.0:
        s = s0 * exp(-y * time2maturity)
    else:
        s = s0

    cdf_d1_prime = exp(-0.5 * d1 * d1) / sqrt(2.0 * pi)

    return s * cdf_d1_prime * sqrt(time2maturity) / 100


def get_d1_d2(s0, x, r, vol, time2maturity, y=0.0):
    """
    get_d1_d2(s0,x,r,vol,time2maturity,y) -> returns 'd1' and 'd2' taking the
    current stock price 's0', the exercise price 'x', the annualized risk-free
    rate 'r', the annualized volatility 'vol', the time remaining to option
    expiration in units of year, and the stocks's annualized dividend yield 'y'
    (default is zero, i.e., the stock does not pay dividends) as arguments.
    """
    d1 = (log(s0 / x) + (r - y + vol * vol / 2.0) * time2maturity) / (
        vol * sqrt(time2maturity)
    )
    d2 = d1 - vol * sqrt(time2maturity)

    return d1, d2


def getitmprob(optype, d2, time2maturity=0.0, y=0.0):
    """
    getitmprob(optype,d2,time2maturity,y) -> returns the estimated probability
    that an option (either call or put) will be in-the-money at maturity, taking
    'd2' as defined in the Black-Scholes formula as a mandatory argument. Optionally,
    the time remaining to maturity in units of year and the stocks's annualized
    dividend yield 'y' (default is zero,i.e., the stock does not pay dividends)
    may be passed as arguments.
    """
    if y > 0.0 and time2maturity > 0.0:
        yfac = exp(-y * time2maturity)
    else:
        yfac = 1.0

    if optype == "call":
        return yfac * stats.norm.cdf(d2)
    elif optype == "put":
        return yfac * stats.norm.cdf(-d2)
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def getBSinfo(s, x, r, vol, time2maturity, y=0.0):
    """
    getBSinfo(s,x,r,vol,time2maturity,y) -> provides informaton about call and
    put options using the Black-Scholes formula, taking the current stock price
    's', the option strike 'x', the annualized risk-free rate 'r', the annualized
    volatility 'vol', the time remaining to maturity in units of year, and
    the annualized stock's dividend yield 'y' as arguments.
    """
    d1, d2 = get_d1_d2(s, x, r, vol, time2maturity, y)
    callprice = getoptionprice("call", s, x, r, time2maturity, d1, d2, y)
    putprice = getoptionprice("put", s, x, r, time2maturity, d1, d2, y)
    calldelta = getdelta("call", d1, time2maturity, y)
    putdelta = getdelta("put", d1, time2maturity, y)
    calltheta = gettheta("call", s, x, r, vol, time2maturity, d1, d2, y)
    puttheta = gettheta("put", s, x, r, vol, time2maturity, d1, d2, y)
    gamma = getgamma(s, vol, time2maturity, d1, y)
    vega = getvega(s, time2maturity, d1, y)
    callitmprob = getitmprob("call", d2, time2maturity, y)
    putitmprob = getitmprob("put", d2, time2maturity, y)

    return (
        callprice,
        putprice,
        calldelta,
        putdelta,
        calltheta,
        puttheta,
        gamma,
        vega,
        callitmprob,
        putitmprob,
    )
