import numpy as np
import pytest

from optionlab.black_scholes import (
    get_bs_info,
    get_d1,
    get_d2,
    get_delta,
    get_gamma,
    get_implied_vol,
    get_itm_probability,
    get_probability_of_touch,
    get_rho,
    get_theta,
    get_vega,
)

S0 = 100.0
RATE = 0.05
VOL = 0.3
YEARS = 60 / 365


@pytest.mark.parametrize("y", [0.0, 0.03])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_put_call_parity(strike, y):
    bs = get_bs_info(S0, strike, RATE, VOL, YEARS, y)

    parity = S0 * np.exp(-y * YEARS) - strike * np.exp(-RATE * YEARS)

    assert bs.call_price - bs.put_price == pytest.approx(parity)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_implied_vol_round_trip(option_type):
    bs = get_bs_info(S0, 105.0, RATE, VOL, YEARS)
    price = bs.call_price if option_type == "call" else bs.put_price

    implied_vol = get_implied_vol(option_type, float(price), S0, 105.0, RATE, YEARS)

    assert implied_vol == pytest.approx(VOL, abs=1e-4)


def test_implied_vol_clamps_at_upper_bound():
    # Quote far above any Black-Scholes price for vol <= 1.0
    assert get_implied_vol("call", 50.0, S0, 100.0, RATE, 30 / 365) == 1.0


def test_implied_vol_clamps_at_lower_bound():
    # Deep ITM call quoted below its minimum theoretical value
    assert get_implied_vol("call", 51.0, S0, 50.0, RATE, 0.5) == 0.001


def test_standalone_getters_match_get_bs_info():
    x = 105.0
    y = 0.02
    bs = get_bs_info(S0, x, RATE, VOL, YEARS, y)
    d1 = get_d1(S0, x, RATE, VOL, YEARS, y)
    d2 = get_d2(S0, x, RATE, VOL, YEARS, y)

    assert get_delta("call", d1, YEARS, y) == pytest.approx(bs.call_delta)
    assert get_delta("put", d1, YEARS, y) == pytest.approx(bs.put_delta)
    assert get_gamma(S0, VOL, YEARS, d1, y) == pytest.approx(bs.gamma)
    assert get_vega(S0, YEARS, d1, y) == pytest.approx(bs.vega)
    assert get_theta("call", S0, x, RATE, VOL, YEARS, d1, d2, y) == pytest.approx(
        bs.call_theta
    )
    assert get_theta("put", S0, x, RATE, VOL, YEARS, d1, d2, y) == pytest.approx(
        bs.put_theta
    )
    assert get_rho("call", x, RATE, YEARS, d2) == pytest.approx(bs.call_rho)
    assert get_rho("put", x, RATE, YEARS, d2) == pytest.approx(bs.put_rho)
    assert get_itm_probability("call", d2, YEARS, y) == pytest.approx(bs.call_itm_prob)
    assert get_itm_probability("put", d2, YEARS, y) == pytest.approx(bs.put_itm_prob)


def test_probability_of_touch_itm_is_certain():
    assert get_probability_of_touch("call", S0, 90.0, RATE, VOL, YEARS) == 1.0
    assert get_probability_of_touch("put", S0, 110.0, RATE, VOL, YEARS) == 1.0


def test_probability_of_touch_otm_in_unit_interval():
    call_touch = get_probability_of_touch("call", S0, 110.0, RATE, VOL, YEARS)
    put_touch = get_probability_of_touch("put", S0, 90.0, RATE, VOL, YEARS)

    assert 0.0 < call_touch < 1.0
    assert 0.0 < put_touch < 1.0


def test_probability_of_touch_matches_get_bs_info():
    bs = get_bs_info(S0, 110.0, RATE, VOL, YEARS)

    assert get_probability_of_touch("call", S0, 110.0, RATE, VOL, YEARS) == (
        pytest.approx(bs.call_prob_of_touch)
    )


def test_invalid_option_type_raises():
    d1 = get_d1(S0, 105.0, RATE, VOL, YEARS)
    d2 = get_d2(S0, 105.0, RATE, VOL, YEARS)

    with pytest.raises(ValueError):
        get_delta("stock", d1, YEARS)

    with pytest.raises(ValueError):
        get_theta("stock", S0, 105.0, RATE, VOL, YEARS, d1, d2)

    with pytest.raises(ValueError):
        get_rho("stock", 105.0, RATE, YEARS, d2)

    with pytest.raises(ValueError):
        get_itm_probability("stock", d2, YEARS)

    with pytest.raises(ValueError):
        get_probability_of_touch("stock", S0, 105.0, RATE, VOL, YEARS)


def test_get_bs_info_vectorized_strikes():
    strikes = np.array([90.0, 100.0, 110.0])

    bs_vec = get_bs_info(S0, strikes, RATE, VOL, YEARS)

    for i, strike in enumerate(strikes):
        bs_scalar = get_bs_info(S0, float(strike), RATE, VOL, YEARS)

        assert bs_vec.call_price[i] == pytest.approx(bs_scalar.call_price)
        assert bs_vec.put_price[i] == pytest.approx(bs_scalar.put_price)
        assert bs_vec.call_delta[i] == pytest.approx(bs_scalar.call_delta)
        assert bs_vec.gamma[i] == pytest.approx(bs_scalar.gamma)
        assert bs_vec.call_prob_of_touch[i] == pytest.approx(
            bs_scalar.call_prob_of_touch
        )
        assert bs_vec.put_prob_of_touch[i] == pytest.approx(bs_scalar.put_prob_of_touch)
