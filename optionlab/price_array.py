"""
This module defines the `create_price_array` function, which samples terminal stock
prices directly from the Black-Scholes distribution.

The terminal price array can later be used to calculate the probability of profit 
(PoP) of a strategy using the `optionlab.engine.run_strategy` function.
"""

import numpy as np

from optionlab.models import BlackScholesModelInputs


def create_price_array(
    inputs_data: BlackScholesModelInputs | dict,
    n: int = 100_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generates terminal stock prices.

    Parameters
    ----------
    `inputs_data`: input data used to generate the terminal stock prices.

    `n`: number of terminal stock prices.

    `seed`: seed for a local random number generator. Reusing the same seed and
    inputs produces the same values in a new array without changing NumPy's global
    random state.

    Returns
    -------
    Array of terminal prices.
    """

    inputs: BlackScholesModelInputs

    if isinstance(inputs_data, dict):
        input_type = inputs_data["model"]

        if input_type == "black-scholes":
            inputs = BlackScholesModelInputs.model_validate(inputs_data)
        else:
            raise ValueError("Inputs are not valid!")
    else:
        inputs = inputs_data

        if isinstance(inputs, BlackScholesModelInputs):
            input_type = "black-scholes"
        else:
            raise ValueError("Inputs are not valid!")

    rng = np.random.default_rng(seed)
    return _get_array_price_from_BS(inputs, n, rng)


def _get_array_price_from_BS(
    inputs: BlackScholesModelInputs, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Generates risk-neutral Black-Scholes terminal stock prices."""

    time = inputs.years_to_target_date
    location = (
        np.log(inputs.stock_price)
        + (inputs.interest_rate - inputs.dividend_yield - 0.5 * inputs.volatility**2)
        * time
    )
    scale = inputs.volatility * np.sqrt(time)
    return np.exp(rng.normal(location, scale, n))
