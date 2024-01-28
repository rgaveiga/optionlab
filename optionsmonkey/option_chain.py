from numpy import round, array
from optionsmonkey.black_scholes import (
    get_d1_d2,
    get_option_price,
    get_delta,
    get_gamma,
    get_theta,
    get_vega,
)


def create_bs_option_chain(
    s0: float,
    minx: float,
    maxx: float,
    vol: float,
    r: float,
    time2maturity: float,
    n: int,
    y: float = 0.0,
):
    """
    Generates an equally spaced option chain calculated with the Black-Scholes model.

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
            "price": get_option_price("call", s0, x, r, time2maturity, d1, d2, y),
            "delta": get_delta("call", d1, time2maturity, y),
            "theta": get_theta("call", s0, x, r, vol, time2maturity, d1, d2, y),
        },
        "puts": {
            "price": get_option_price("put", s0, x, r, time2maturity, d1, d2, y),
            "delta": get_delta("put", d1, time2maturity, y),
            "theta": get_theta("put", s0, x, r, vol, time2maturity, d1, d2, y),
        },
        "gamma": get_gamma(s0, vol, time2maturity, d1, y),
        "vega": get_vega(s0, time2maturity, d1, y),
    }
