from functools import lru_cache

import numpy as np
from numpy import exp
from numpy.random import seed as np_seed_number, normal, laplace
from numpy.lib.scimath import log, sqrt

from optionlab.models import BlackScholesModelInputs, LaplaceInputs


def create_price_array(
    inputs_data: BlackScholesModelInputs | LaplaceInputs | dict,
    n: int = 100_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generates terminal stock prices.

    Parameters
    ----------
    inputs_data : BlackScholesModelInputs | LaplaceInputs | dict
        Input data used to generate the terminal stock prices. See the documentation
        for `BlackScholesModelInputs` and `LaplaceInputs` for more details.
    n : int, optional
        Number of terminal stock prices. The default is 100,000.
    seed : int | None, optional
        Seed for random number generation. The default is None.

    Returns
    -------
    numpy.ndarray
        Array of terminal prices.
    """

    inputs: BlackScholesModelInputs | LaplaceInputs

    if isinstance(inputs_data, dict):
        input_type = inputs_data["model"]

        if input_type in ("black-scholes", "normal"):
            inputs = BlackScholesModelInputs.model_validate(inputs_data)
        elif input_type == "laplace":
            inputs = LaplaceInputs.model_validate(inputs_data)
        else:
            raise ValueError("Inputs are not valid!")
    else:
        inputs = inputs_data

        if isinstance(inputs, BlackScholesModelInputs):
            input_type = "black-scholes"
        elif isinstance(inputs, LaplaceInputs):
            input_type = "laplace"
        else:
            raise ValueError("Inputs are not valid!")

    np_seed_number(seed)

    if input_type in ("black-scholes", "normal"):
        arr = _get_array_price_from_BS(inputs, n)
    elif input_type == "laplace":
        arr = _get_array_price_from_laplace(inputs, n)

    np_seed_number(None)

    return arr


@lru_cache
def _get_array_price_from_BS(inputs: BlackScholesModelInputs, n: int) -> np.ndarray:
    return exp(
        normal(
            (
                log(inputs.stock_price)
                + (
                    inputs.interest_rate
                    - inputs.dividend_yield
                    - 0.5 * inputs.volatility * inputs.volatility
                )
                * inputs.years_to_target_date
            ),
            inputs.volatility * sqrt(inputs.years_to_target_date),
            n,
        )
    )


@lru_cache
def _get_array_price_from_laplace(inputs: LaplaceInputs, n: int) -> np.ndarray:
    return exp(
        laplace(
            (log(inputs.stock_price) + inputs.mu * inputs.years_to_target_date),
            (inputs.volatility * sqrt(inputs.years_to_target_date)) / sqrt(2.0),
            n,
        )
    )
