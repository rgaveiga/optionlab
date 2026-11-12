"""
This module defines functions that generate and plot underlying asset price paths.
"""

import matplotlib.pyplot as plt
import numpy as np


def create_price_path_bs(
    s0: float,
    days_to_maturity: float | int,
    volatility: float,
    interest_rate: float,
    n_intervals: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """
    Generates underlying asset price paths on an equally spaced time grid, assuming
    a geometric brownian motion and risk neutrality.

    Parameters
    ----------
    `s0`: initial underlying asset price.

    `days_to_maturity`: number of trading days until maturity.

    `volatility`: annual volatility, expressed as a decimal.

    `interest_rate`: continuously compounded annual risk-free rate, expressed
    as a decimal.

    `n_intervals`: number of time steps between today and maturity.

    `n_paths`: number of independent paths.

    `seed`: seed for random number generator.

    Returns
    -------
    Array of price paths. Each row is a price path, and each column is
    a Monte Carlo step, including the initial prices at step zero.

    """
    if not np.isfinite(s0) or s0 <= 0:
        raise ValueError("Initial asset price must be finite and positive.")
    if not np.isfinite(days_to_maturity) or days_to_maturity < 0:
        raise ValueError("Time to maturity must be finite and nonnegative.")
    if not np.isfinite(volatility) or volatility < 0:
        raise ValueError("Volatility must be finite and nonnegative.")
    if not np.isfinite(interest_rate):
        raise ValueError("Interest rate must be finite.")
    for name, count in (("n_intervals", n_intervals), ("n_paths", n_paths)):
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer.")
    if n_intervals < 1 or n_paths < 0:
        raise ValueError(
            "Number of intervals must be positive and number of paths nonnegative."
        )

    dt = days_to_maturity / 252.0 / n_intervals
    rng = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_intervals + 1), dtype=np.float64)
    rng.standard_normal(out=paths)
    paths *= volatility * np.sqrt(dt)
    paths += (interest_rate - 0.5 * volatility**2) * dt
    paths[:, 0] = 0.0
    np.cumsum(paths, axis=1, out=paths)
    np.exp(paths, out=paths)
    paths *= s0
    return paths


def plot_price_paths(paths: np.ndarray) -> None:
    """
    Plots all underlying asset price paths from the initial to the final step.

    ### Parameters

    `paths`: array of price paths. Each row is a price path, and each column is
    a Monte Carlo step, including the initial prices at step zero.

    ### Returns

    `None`.
    """
    _, ax = plt.subplots()
    ax.plot(paths.T)
    ax.set_xlim(0, paths.shape[1] - 1)
    ax.set_xlabel("Monte Carlo steps")
    ax.set_ylabel("Asset prices ($)")
