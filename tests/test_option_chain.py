import numpy as np
import pytest

from optionlab.black_scholes import get_bs_info
from optionlab.option_chain import create_bs_option_chain


S0 = 100.0
RATE = 0.05
VOL = 0.3
YEARS = 60 / 365


def test_create_bs_option_chain_returns_equally_spaced_bs_arrays():
    chain = create_bs_option_chain(S0, 90.001, 90.009, VOL, RATE, YEARS, 5, 0.02)

    expected_strikes = np.linspace(90.001, 90.009, 5)
    expected_info = get_bs_info(S0, expected_strikes, RATE, VOL, YEARS, 0.02)

    np.testing.assert_array_equal(chain["strikes"], expected_strikes)
    np.testing.assert_allclose(chain["calls"]["price"], expected_info.call_price)
    np.testing.assert_allclose(chain["calls"]["delta"], expected_info.call_delta)
    np.testing.assert_allclose(chain["calls"]["theta"], expected_info.call_theta)
    np.testing.assert_allclose(chain["puts"]["price"], expected_info.put_price)
    np.testing.assert_allclose(chain["puts"]["delta"], expected_info.put_delta)
    np.testing.assert_allclose(chain["puts"]["theta"], expected_info.put_theta)
    np.testing.assert_allclose(chain["gamma"], expected_info.gamma)
    np.testing.assert_allclose(chain["vega"], expected_info.vega)

    assert set(chain) == {"strikes", "calls", "puts", "gamma", "vega"}
    assert set(chain["calls"]) == {"price", "delta", "theta"}
    assert set(chain["puts"]) == {"price", "delta", "theta"}
    assert all(
        isinstance(values, np.ndarray) and values.shape == (5,)
        for values in (
            chain["strikes"],
            *chain["calls"].values(),
            *chain["puts"].values(),
            chain["gamma"],
            chain["vega"],
        )
    )


def test_create_bs_option_chain_accepts_two_strikes():
    chain = create_bs_option_chain(S0, 90.0, 110.0, VOL, RATE, YEARS, 2)

    np.testing.assert_array_equal(chain["strikes"], np.array([90.0, 110.0]))


@pytest.mark.parametrize(
    ("min_strike", "max_strike", "n", "message"),
    [
        (0.0, 110.0, 5, "Strike limits must be finite and greater than zero"),
        (90.0, np.inf, 5, "Strike limits must be finite and greater than zero"),
        (110.0, 90.0, 5, "Maximum strike must be greater than minimum strike"),
        (90.0, 90.0, 5, "Maximum strike must be greater than minimum strike"),
        (90.0, 110.0, 1, "Number of strikes must be an integer greater than one"),
        (90.0, 110.0, 2.5, "Number of strikes must be an integer greater than one"),
        (90.0, 110.0, True, "Number of strikes must be an integer greater than one"),
    ],
)
def test_create_bs_option_chain_rejects_invalid_range(
    min_strike, max_strike, n, message
):
    with pytest.raises(ValueError, match=message):
        create_bs_option_chain(S0, min_strike, max_strike, VOL, RATE, YEARS, n)
