import random

import numpy as np
import pytest

from optionlab.mc import create_price_path_bs


@pytest.mark.parametrize("n_intervals", [1, 12, 252])
def test_zero_volatility_reaches_maturity(n_intervals):
    paths = create_price_path_bs(100.0, 126, 0.0, -0.05, n_intervals, 3, 42)
    times = np.linspace(0.0, 0.5, n_intervals + 1)
    expected = np.broadcast_to(100.0 * np.exp(-0.05 * times), paths.shape)
    assert paths.shape == (3, n_intervals + 1)
    assert paths.dtype == np.float64
    np.testing.assert_allclose(paths, expected, rtol=1e-13)
    np.testing.assert_array_equal(paths[:, 0], 100.0)


def test_seed_and_global_random_states():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    first = create_price_path_bs(100.0, 252, 0.2, 0.05, 12, 10, 42)
    repeated = create_price_path_bs(100.0, 252, 0.2, 0.05, 12, 10, 42)
    different = create_price_path_bs(100.0, 252, 0.2, 0.05, 12, 10, 43)
    np.testing.assert_array_equal(first, repeated)
    assert not np.shares_memory(first, repeated)
    assert not np.array_equal(first, different)
    assert random.getstate() == python_state
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    np.testing.assert_array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]


def test_black_scholes_path_distribution():
    paths = create_price_path_bs(100.0, 252, 0.2, 0.05, 4, 100_000, 12345)
    assert np.all(paths > 0.0)
    np.testing.assert_array_equal(paths[:, 0], 100.0)
    log_returns = np.log(paths[:, 1:] / 100.0)
    times = np.arange(1, 5) / 4.0
    np.testing.assert_allclose(
        log_returns.mean(axis=0), (0.05 - 0.2**2 / 2) * times, atol=0.002
    )
    # Brownian covariance checks that successive prices share their history.
    expected_covariance = 0.2**2 * np.minimum.outer(times, times)
    np.testing.assert_allclose(
        np.cov(log_returns, rowvar=False), expected_covariance, rtol=0.025
    )


def test_zero_time_and_empty_paths():
    paths = create_price_path_bs(100.0, 0, 0.2, 0.05, 12, 3, 42)
    np.testing.assert_array_equal(paths, np.full((3, 13), 100.0))
    empty = create_price_path_bs(100.0, 252, 0.2, 0.05, 12, 0, 42)
    assert empty.shape == (0, 13)


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("s0", 0, ValueError),
        ("s0", np.nan, ValueError),
        ("days_to_maturity", -1, ValueError),
        ("days_to_maturity", np.inf, ValueError),
        ("volatility", -0.2, ValueError),
        ("volatility", np.nan, ValueError),
        ("interest_rate", np.inf, ValueError),
        ("n_intervals", 0, ValueError),
        ("n_paths", -1, ValueError),
        ("n_intervals", 1.5, TypeError),
        ("n_paths", True, TypeError),
    ],
)
def test_invalid_inputs(name, value, error):
    inputs = dict(
        s0=100.0,
        days_to_maturity=252,
        volatility=0.2,
        interest_rate=0.05,
        n_intervals=12,
        n_paths=3,
        seed=42,
    )
    inputs[name] = value
    with pytest.raises(error):
        create_price_path_bs(**inputs)
