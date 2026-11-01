import numpy as np
import pytest

from optionlab.models import BlackScholesModelInputs
from optionlab.price_array import _get_array_price_from_BS, create_price_array


class RecordingGenerator:
    def __init__(self):
        self.arguments = None

    def normal(self, location, scale, size):
        self.arguments = (location, scale, size)
        return np.full(size, location)


def assert_random_state_equal(actual, expected):
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_black_scholes_parameters_and_dividends():
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=0.2,
        interest_rate=0.05,
        dividend_yield=0.02,
        years_to_target_date=1.5,
    )
    rng = RecordingGenerator()

    prices = _get_array_price_from_BS(inputs, 3, rng)

    expected_location = np.log(100.0) + (0.05 - 0.02 - 0.2**2 / 2.0) * 1.5
    expected_scale = 0.2 * np.sqrt(1.5)
    assert rng.arguments == pytest.approx((expected_location, expected_scale, 3))
    np.testing.assert_allclose(prices, np.exp(expected_location))


@pytest.mark.parametrize("volatility", [0.0, 0.2])
def test_black_scholes_zero_time_is_deterministic(volatility):
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=volatility,
        interest_rate=0.05,
        dividend_yield=0.02,
        years_to_target_date=0.0,
    )
    prices = create_price_array(inputs, n=10, seed=1)
    np.testing.assert_allclose(prices, 100.0)


def test_black_scholes_log_return_moments():
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=0.2,
        interest_rate=0.05,
        dividend_yield=0.02,
        years_to_target_date=1.5,
    )
    prices = create_price_array(inputs, n=200_000, seed=12345)
    log_returns = np.log(prices / inputs.stock_price)
    expected_mean = (0.05 - 0.02 - 0.2**2 / 2.0) * 1.5
    expected_variance = 0.2**2 * 1.5
    assert log_returns.mean() == pytest.approx(expected_mean, abs=0.0015)
    assert log_returns.var() == pytest.approx(expected_variance, rel=0.015)


def test_seed_controls_new_independent_arrays():
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=0.2,
        interest_rate=0.05,
        years_to_target_date=1.0,
    )
    first = create_price_array(inputs, n=100, seed=1)
    repeated = create_price_array(inputs, n=100, seed=1)
    different = create_price_array(inputs, n=100, seed=2)

    assert first is not repeated
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)

    first[0] = -1.0
    fresh = create_price_array(inputs, n=100, seed=1)
    assert fresh[0] != -1.0


def test_global_random_state_is_preserved():
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=0.2,
        interest_rate=0.05,
        years_to_target_date=1.0,
    )
    np.random.seed(2468)
    state = np.random.get_state()
    create_price_array(inputs, n=10, seed=1)
    assert_random_state_equal(np.random.get_state(), state)


def test_dictionary_and_model_inputs_match():
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=0.2,
        interest_rate=0.05,
        dividend_yield=0.02,
        years_to_target_date=1.0,
    )
    from_model = create_price_array(inputs, n=100, seed=1)
    from_dictionary = create_price_array(inputs.model_dump(), n=100, seed=1)
    np.testing.assert_array_equal(from_dictionary, from_model)
