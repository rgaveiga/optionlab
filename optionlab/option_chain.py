"""Functions for creating synthetic option chains from models."""

import numpy as np

from optionlab.black_scholes import get_bs_info
from optionlab.models import FloatOrNdarray


OptionChain = dict[str, FloatOrNdarray | dict[str, FloatOrNdarray]]
"""Black-Scholes option chain indexed by strike."""


def create_bs_option_chain(
    s0: float,
    min_strike: float,
    max_strike: float,
    vol: float,
    r: float,
    years_to_maturity: float,
    n: int,
    y: float = 0.0,
) -> OptionChain:
    """
    Creates an equally spaced option chain calculated using the Black-Scholes model.

    ### Parameters

    `s0`: stock price.

    `min_strike`: lowest strike price, included in the option chain.

    `max_strike`: highest strike price, included in the option chain.

    `vol`: annualized volatility.

    `r`: annualized risk-free interest rate.

    `years_to_maturity`: time remaining to maturity, in years.

    `n`: number of strikes in the option chain. It must be greater than one.

    `y`: annualized dividend yield.

    ### Returns

    A dictionary containing the strikes and the call and put prices and Greeks as
    NumPy arrays.
    """

    if (
        not np.isfinite(min_strike)
        or not np.isfinite(max_strike)
        or min_strike <= 0
        or max_strike <= 0
    ):
        raise ValueError("Strike limits must be finite and greater than zero")
    if max_strike <= min_strike:
        raise ValueError("Maximum strike must be greater than minimum strike")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError("Number of strikes must be an integer greater than one")

    strikes = np.linspace(min_strike, max_strike, n)
    info = get_bs_info(s0, strikes, r, vol, years_to_maturity, y)

    return {
        "strikes": strikes,
        "calls": {
            "price": info.call_price,
            "delta": info.call_delta,
            "theta": info.call_theta,
        },
        "puts": {
            "price": info.put_price,
            "delta": info.put_delta,
            "theta": info.put_theta,
        },
        "gamma": info.gamma,
        "vega": info.vega,
    }
