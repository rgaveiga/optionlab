import numpy as np
import pytest
from scipy import stats

from optionlab.models import ArrayInputs, BlackScholesModelInputs
from optionlab.support import (
    create_price_seq,
    get_pl_profile,
    get_pl_profile_bs,
    get_pl_profile_stock,
    get_pop,
    _get_payoff,
    _get_pl_option,
    _get_profit_range,
    _get_sign_changes,
)


def test_create_price_seq_length_and_endpoints():
    seq = create_price_seq(100.0, 101.0)

    assert seq.shape[0] == 101
    assert seq[0] == pytest.approx(100.0)
    assert seq[-1] == pytest.approx(101.0)
    assert np.all(np.diff(seq) == pytest.approx(0.01))
    assert np.array_equal(seq, np.round(seq, 2))


def test_create_price_seq_invalid_range():
    with pytest.raises(ValueError):
        create_price_seq(101.0, 100.0)

    with pytest.raises(ValueError):
        create_price_seq(100.0, 100.0)


def test_create_price_seq_custom_step():
    seq = create_price_seq(100.0, 101.0, 0.1)

    assert seq.shape[0] == 11
    assert seq[0] == pytest.approx(100.0)
    assert seq[-1] == pytest.approx(101.0)
    assert np.all(np.diff(seq) == pytest.approx(0.1))


def test_create_price_seq_default_step_unchanged():
    assert np.array_equal(
        create_price_seq(100.0, 101.0), create_price_seq(100.0, 101.0, 0.01)
    )


def test_create_price_seq_coarse_step_keeps_cent_endpoints():
    seq = create_price_seq(68.99, 268.99, 0.1)

    assert seq.shape[0] == 2001
    assert seq[0] == pytest.approx(68.99)
    assert seq[-1] == pytest.approx(268.99)


def test_create_price_seq_invalid_step():
    with pytest.raises(ValueError):
        create_price_seq(100.0, 101.0, 0.0)

    with pytest.raises(ValueError):
        create_price_seq(100.0, 101.0, -0.01)


def test_get_payoff_call_and_put():
    s = np.array([90.0, 100.0, 110.0])

    call_payoff = _get_payoff("call", s, 100.0)
    put_payoff = _get_payoff("put", s, 100.0)

    assert np.allclose(call_payoff, [0.0, 0.0, 10.0])
    assert np.allclose(put_payoff, [10.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        _get_payoff("stock", s, 100.0)


def test_get_pl_option_buy_sell_symmetry():
    s = np.array([90.0, 100.0, 110.0])

    buy_pl = _get_pl_option("call", 5.0, "buy", s, 100.0)
    sell_pl = _get_pl_option("call", 5.0, "sell", s, 100.0)

    assert np.allclose(buy_pl, [-5.0, -5.0, 5.0])
    assert np.allclose(buy_pl, -sell_pl)

    with pytest.raises(ValueError):
        _get_pl_option("call", 5.0, "hold", s, 100.0)


def test_get_pl_profile():
    s = np.array([90.0, 100.0, 110.0])

    buy_profile, buy_cost = get_pl_profile("call", "buy", 100.0, 5.0, 100, s)
    sell_profile, sell_cost = get_pl_profile("call", "sell", 100.0, 5.0, 100, s)

    assert np.allclose(buy_profile, [-500.0, -500.0, 500.0])
    assert buy_cost == pytest.approx(-500.0)
    assert np.allclose(sell_profile, [500.0, 500.0, -500.0])
    assert sell_cost == pytest.approx(500.0)

    with pytest.raises(ValueError):
        get_pl_profile("call", "hold", 100.0, 5.0, 100, s)


def test_get_pl_profile_commission():
    s = np.array([90.0, 110.0])

    profile, cost = get_pl_profile("call", "buy", 100.0, 5.0, 100, s, commission=1.5)
    profile_no_comm, cost_no_comm = get_pl_profile("call", "buy", 100.0, 5.0, 100, s)

    assert np.allclose(profile, profile_no_comm - 1.5)
    assert cost == pytest.approx(cost_no_comm - 1.5)


def test_get_pl_profile_stock():
    s = np.array([90.0, 100.0, 110.0])

    buy_profile, buy_cost = get_pl_profile_stock(100.0, "buy", 100, s)
    sell_profile, sell_cost = get_pl_profile_stock(100.0, "sell", 100, s)

    assert np.allclose(buy_profile, [-1000.0, 0.0, 1000.0])
    assert buy_cost == pytest.approx(-10000.0)
    assert np.allclose(buy_profile, -sell_profile)
    assert sell_cost == pytest.approx(10000.0)

    with pytest.raises(ValueError):
        get_pl_profile_stock(100.0, "hold", 100, s)


def test_get_pl_profile_bs_converges_to_payoff():
    s = np.array([80.0, 90.0, 110.0, 120.0])
    strike, premium, n = 100.0, 5.0, 100

    profile, cost = get_pl_profile_bs(
        "call", "buy", strike, premium, 0.0, 1e-6, 0.3, n, s
    )

    intrinsic_profile = n * (_get_payoff("call", s, strike) - premium)

    assert np.allclose(profile, intrinsic_profile, atol=0.05)
    assert cost == pytest.approx(-premium * n)


def test_get_pl_profile_bs_buy_sell_symmetry():
    s = np.array([80.0, 100.0, 120.0])

    buy_profile, _ = get_pl_profile_bs(
        "put", "buy", 100.0, 5.0, 0.01, 30 / 365, 0.3, 100, s
    )
    sell_profile, _ = get_pl_profile_bs(
        "put", "sell", 100.0, 5.0, 0.01, 30 / 365, 0.3, 100, s
    )

    assert np.allclose(buy_profile, -sell_profile)


def test_profit_range_always_profitable():
    s = np.arange(0.0, 10.0)
    profit = np.full(s.shape, 100.0)

    profit_range, loss_range = _get_profit_range(s, profit)

    assert profit_range == [(0.0, float("inf"))]
    assert loss_range == [(0.0, 0.0)]


def test_profit_range_never_profitable():
    s = np.arange(0.0, 10.0)
    profit = np.full(s.shape, -100.0)

    profit_range, loss_range = _get_profit_range(s, profit)

    assert profit_range == [(0.0, 0.0)]
    assert loss_range == [(0.0, float("inf"))]


def test_profit_range_single_rising_crossing():
    s = np.arange(0.0, 6.0)
    profit = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0])

    profit_range, loss_range = _get_profit_range(s, profit)

    assert profit_range == [(s[3], float("inf"))]
    assert loss_range == [(0.0, s[2])]


def test_profit_range_single_falling_crossing():
    s = np.arange(0.0, 6.0)
    profit = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    profit_range, loss_range = _get_profit_range(s, profit)

    assert profit_range == [(0.0, s[2])]
    assert loss_range == [(s[3], float("inf"))]


def test_profit_range_middle_bump():
    s = np.arange(0.0, 6.0)
    profit = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0])

    profit_range, loss_range = _get_profit_range(s, profit)

    assert profit_range == [(s[2], s[3])]
    assert loss_range == [(0.0, s[1]), (s[4], float("inf"))]


def test_get_sign_changes():
    assert _get_sign_changes(np.array([-1.0, 1.0]), 0.01) == [1]
    assert _get_sign_changes(np.array([1.0, 1.0]), 0.01) == []
    assert _get_sign_changes(np.array([-1.0, 1.0, -1.0]), 0.01) == [1, 2]


def test_get_sign_changes_value_at_target():
    # A profit exactly equal to the target counts as reaching it (epsilon shift).
    assert _get_sign_changes(np.array([0.01, 1.0]), 0.01) == []
    assert _get_sign_changes(np.array([0.01, -1.0]), 0.01) == [1]


def test_get_pop_array_inputs():
    s = np.arange(0.0, 4.0)
    profit = np.array([-3.0, -1.0, 1.0, 2.0])

    pop = get_pop(s, profit, ArrayInputs(array=np.array([1.0, 2.0, -1.0, -3.0])))

    assert pop.probability_of_reaching_target == pytest.approx(0.5)
    assert pop.probability_of_missing_target == pytest.approx(0.5)
    assert pop.expected_return_above_target == pytest.approx(1.5)
    assert pop.expected_return_below_target == pytest.approx(-2.0)


def test_array_inputs_empty_raises():
    with pytest.raises(ValueError) as err:
        ArrayInputs(array=np.array([]))

    assert "The array is empty!" in str(err.value)


def test_get_pop_black_scholes_probabilities_sum_to_one():
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=0.2,
        years_to_target_date=0.1,
        interest_rate=0.05,
    )
    s = create_price_seq(50.0, 150.0)
    profit = s - 100.0

    pop = get_pop(s, profit, inputs)

    assert pop.probability_of_reaching_target + pop.probability_of_missing_target == (
        pytest.approx(1.0, abs=1e-3)
    )
    assert pop.reaching_target_range[0][0] == pytest.approx(100.01)
    assert pop.reaching_target_range[0][1] == float("inf")

    expected_prob = 1.0 - stats.norm.cdf(
        (np.log(100.01) - (np.log(100.0) + (0.05 - 0.5 * 0.2**2) * 0.1))
        / (0.2 * np.sqrt(0.1))
    )

    assert pop.probability_of_reaching_target == pytest.approx(expected_prob, abs=1e-6)


def test_get_pop_black_scholes_expected_returns_sign():
    inputs = BlackScholesModelInputs(
        stock_price=100.0,
        volatility=0.2,
        years_to_target_date=0.1,
        interest_rate=0.05,
    )
    s = create_price_seq(50.0, 150.0)
    profit = s - 100.0

    pop = get_pop(s, profit, inputs, calculate_expectation=True)

    assert pop.expected_return_above_target > 0.0
    assert pop.expected_return_below_target < 0.0

    pop_no_expectation = get_pop(s, profit, inputs, calculate_expectation=False)

    assert pop_no_expectation.expected_return_above_target == 0.0
    assert pop_no_expectation.expected_return_below_target == 0.0


def test_pl_profiles_can_reuse_output_buffer():
    s = np.array([80.0, 100.0, 120.0])
    out = np.empty_like(s)

    expected_option, _ = get_pl_profile("call", "buy", 100.0, 5.0, 100, s)
    option_profile, _ = get_pl_profile("call", "buy", 100.0, 5.0, 100, s, out=out)
    assert option_profile is out
    np.testing.assert_allclose(option_profile, expected_option)

    expected_stock, _ = get_pl_profile_stock(100.0, "sell", 100, s)
    stock_profile, _ = get_pl_profile_stock(100.0, "sell", 100, s, out=out)
    assert stock_profile is out
    np.testing.assert_allclose(stock_profile, expected_stock)

    expected_bs, _ = get_pl_profile_bs(
        "put", "buy", 100.0, 5.0, 0.01, 30 / 365, 0.3, 100, s
    )
    bs_profile, _ = get_pl_profile_bs(
        "put", "buy", 100.0, 5.0, 0.01, 30 / 365, 0.3, 100, s, out=out
    )
    assert bs_profile is out
    np.testing.assert_allclose(bs_profile, expected_bs)
