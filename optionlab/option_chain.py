from numpy import round, array
from optionlab.black_scholes import (
    get_d1_d2,
    getoptionprice,
    getdelta,
    getgamma,
    gettheta,
    getvega,
)


def createBSoptionchain(s0, minx, maxx, vol, r, time2maturity, n, y=0.0):
    """
    createBSoptionchain(s0,minx,maxx,vol,r,time2maturity,n,y) -> generates an
    equally spaced option chain calculated with the Black-Scholes model.

    Arguments:
    ----------
    s0: stock price.
    minx: lowest strike.
    maxx: highest strike.
    vol: annualized volatility.
    r: annualized risk-free interest rate.
    time2maturity: time left before maturity.
    n: number of strikes in the option chain.
    y: Annualized dividend yield (default is zero).
    """
    deltax = (maxx - minx) / (n - 1)
    x = round(array([(minx + i * deltax) for i in range(n)]), 2)
    d1, d2 = get_d1_d2(s0, x, r, vol, time2maturity, y)

    return {
        "strikes": x,
        "calls": {
            "price": getoptionprice("call", s0, x, r, time2maturity, d1, d2, y),
            "delta": getdelta("call", d1, time2maturity, y),
            "theta": gettheta("call", s0, x, r, vol, time2maturity, d1, d2, y),
        },
        "puts": {
            "price": getoptionprice("put", s0, x, r, time2maturity, d1, d2, y),
            "delta": getdelta("put", d1, time2maturity, y),
            "theta": gettheta("put", s0, x, r, vol, time2maturity, d1, d2, y),
        },
        "gamma": getgamma(s0, vol, time2maturity, d1, y),
        "vega": getvega(s0, time2maturity, d1, y),
    }
