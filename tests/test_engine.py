import numpy as np
import pytest

from optionlab import run_strategy
from optionlab.support import get_pl_profile_bs

COVERED_CALL_LEGS = [
    {"type": "stock", "n": 100, "action": "buy"},
    {"type": "call", "strike": 185.0, "premium": 4.1, "n": 100, "action": "sell"},
]

NAKED_CALL_LEG = [
    {"type": "call", "strike": 185.0, "premium": 4.1, "n": 100, "action": "sell"},
]


@pytest.fixture
def nvidia_days(nvidia):
    return nvidia | {
        "start_date": None,
        "target_date": None,
        "days_to_target_date": 24,
    }


def test_array_model_pop_close_to_black_scholes(nvidia_days):
    bs_payload = nvidia_days | {"strategy": COVERED_CALL_LEGS}
    bs_outputs = run_strategy(bs_payload)

    time_to_target = 24 / 252
    rng = np.random.default_rng(42)
    log_mean = (
        np.log(nvidia_days["stock_price"])
        + (nvidia_days["interest_rate"] - 0.5 * nvidia_days["volatility"] ** 2)
        * time_to_target
    )
    log_sigma = nvidia_days["volatility"] * np.sqrt(time_to_target)
    terminal_prices = rng.lognormal(log_mean, log_sigma, 200_000)

    array_payload = bs_payload | {"model": "array", "array": terminal_prices}
    array_outputs = run_strategy(array_payload)

    assert array_outputs.probability_of_profit == pytest.approx(
        bs_outputs.probability_of_profit, abs=0.01
    )
    assert array_outputs.expected_profit_if_profitable > 0.0
    assert array_outputs.expected_loss_if_unprofitable < 0.0
    assert array_outputs.strategy_cost == pytest.approx(bs_outputs.strategy_cost)


def test_array_model_option_expiring_after_target_uses_volatility(nvidia_days):
    terminal_prices = np.array([140.0, 170.0, 200.0])
    option_leg = {
        "type": "call",
        "strike": 185.0,
        "premium": 4.1,
        "n": 100,
        "action": "buy",
        "expiration": 30,
    }
    outputs = run_strategy(
        nvidia_days
        | {
            "strategy": [option_leg],
            "model": "array",
            "array": terminal_prices,
        }
    )

    expected_profit, _ = get_pl_profile_bs(
        option_type="call",
        action="buy",
        x=185.0,
        val=4.1,
        r=nvidia_days["interest_rate"],
        target_to_maturity_years=(30 - 24) / 252,
        volatility=nvidia_days["volatility"],
        n=100,
        s=terminal_prices,
    )
    wrong_interest_as_volatility, _ = get_pl_profile_bs(
        option_type="call",
        action="buy",
        x=185.0,
        val=4.1,
        r=nvidia_days["interest_rate"],
        target_to_maturity_years=(30 - 24) / 252,
        volatility=nvidia_days["interest_rate"],
        n=100,
        s=terminal_prices,
    )

    assert not np.allclose(expected_profit, wrong_interest_as_volatility)
    np.testing.assert_allclose(outputs.data.strategy_profit_mc, expected_profit)


def test_closed_option_leg_bought(nvidia):
    payload = nvidia | {
        "strategy": [
            {
                "type": "call",
                "strike": 165.0,
                "premium": 12.65,
                "n": 100,
                "action": "buy",
                "prev_pos": -7.5,
            }
        ]
    }

    outputs = run_strategy(payload)

    expected_cost = -(12.65 - 7.5) * 100

    assert outputs.strategy_cost == pytest.approx(expected_cost)
    assert outputs.minimum_return_in_the_domain == pytest.approx(expected_cost)
    assert outputs.maximum_return_in_the_domain == pytest.approx(expected_cost)
    assert np.allclose(outputs.data.strategy_profit, expected_cost)
    assert outputs.implied_volatility == [0.0]
    assert outputs.delta == [0.0]
    assert outputs.gamma == [0.0]


def test_closed_option_leg_sold(nvidia):
    payload = nvidia | {
        "strategy": [
            {
                "type": "put",
                "strike": 165.0,
                "premium": 12.65,
                "n": 100,
                "action": "sell",
                "prev_pos": -7.5,
            }
        ]
    }

    outputs = run_strategy(payload)

    expected_cost = (12.65 - 7.5) * 100

    assert outputs.strategy_cost == pytest.approx(expected_cost)
    assert np.allclose(outputs.data.strategy_profit, expected_cost)


def test_closed_position_leg_shifts_profit(nvidia):
    base_payload = nvidia | {"strategy": NAKED_CALL_LEG}
    base_outputs = run_strategy(base_payload)

    payload = nvidia | {
        "strategy": [{"type": "closed", "prev_pos": 1500.0}] + NAKED_CALL_LEG
    }
    outputs = run_strategy(payload)

    assert outputs.strategy_cost == pytest.approx(base_outputs.strategy_cost + 1500.0)
    assert outputs.minimum_return_in_the_domain == pytest.approx(
        base_outputs.minimum_return_in_the_domain + 1500.0
    )
    assert outputs.maximum_return_in_the_domain == pytest.approx(
        base_outputs.maximum_return_in_the_domain + 1500.0
    )


def test_integer_expiration_beyond_target(nvidia_days):
    payload = nvidia_days | {
        "strategy": [NAKED_CALL_LEG[0] | {"expiration": 30}],
    }

    outputs = run_strategy(payload)

    assert 0.0 < outputs.probability_of_profit < 1.0


def test_integer_expiration_matching_target(nvidia_days):
    base_outputs = run_strategy(nvidia_days | {"strategy": NAKED_CALL_LEG})
    outputs = run_strategy(
        nvidia_days | {"strategy": [NAKED_CALL_LEG[0] | {"expiration": 24}]}
    )

    assert outputs.probability_of_profit == pytest.approx(
        base_outputs.probability_of_profit
    )


def test_integer_expiration_before_target_raises(nvidia_days):
    payload = nvidia_days | {
        "strategy": [NAKED_CALL_LEG[0] | {"expiration": 20}],
    }

    with pytest.raises(ValueError) as err:
        run_strategy(payload)

    assert "Days remaining to maturity" in str(err.value)


def test_discard_nonbusiness_days_changes_result(nvidia):
    payload = nvidia | {"strategy": COVERED_CALL_LEGS}

    default_outputs = run_strategy(payload)
    calendar_outputs = run_strategy(payload | {"discard_nonbusiness_days": False})

    assert default_outputs.probability_of_profit != pytest.approx(
        calendar_outputs.probability_of_profit
    )


def test_run_strategy_accepts_dict(nvidia):
    outputs = run_strategy(nvidia | {"strategy": NAKED_CALL_LEG})

    assert 0.0 < outputs.probability_of_profit < 1.0


def test_unknown_leg_type_rejected(nvidia):
    with pytest.raises(ValueError):
        run_strategy(nvidia | {"strategy": [{"type": "banana", "n": 100}]})


def test_price_step_coarse_grid_close_to_default(nvidia):
    payload = nvidia | {"strategy": COVERED_CALL_LEGS, "calculations": ["pop"]}

    default_outputs = run_strategy(payload)
    coarse_outputs = run_strategy(payload | {"price_step": 0.1})

    assert coarse_outputs.data.stock_price_array.shape[0] == 2001
    assert default_outputs.data.stock_price_array.shape[0] == 20001
    # Coarse grids locate breakeven crossings at step resolution, so PoP can
    # shift by roughly step * lognormal density near the breakeven
    assert coarse_outputs.probability_of_profit == pytest.approx(
        default_outputs.probability_of_profit, abs=5e-3
    )
    assert coarse_outputs.strategy_cost == pytest.approx(
        default_outputs.strategy_cost
    )
    assert coarse_outputs.minimum_return_in_the_domain == pytest.approx(
        default_outputs.minimum_return_in_the_domain, abs=15.0
    )
    assert coarse_outputs.maximum_return_in_the_domain == pytest.approx(
        default_outputs.maximum_return_in_the_domain
    )


def test_price_step_must_be_positive(nvidia):
    with pytest.raises(ValueError):
        run_strategy(nvidia | {"strategy": COVERED_CALL_LEGS, "price_step": 0.0})

    with pytest.raises(ValueError):
        run_strategy(nvidia | {"strategy": COVERED_CALL_LEGS, "price_step": -0.01})


def test_array_model_does_not_populate_per_leg_profit_mc(nvidia_days):
    rng = np.random.default_rng(42)
    terminal_prices = rng.lognormal(np.log(nvidia_days["stock_price"]), 0.1, 10_000)

    payload = nvidia_days | {
        "strategy": COVERED_CALL_LEGS,
        "model": "array",
        "array": terminal_prices,
    }
    outputs = run_strategy(payload)

    assert outputs.data.profit_mc.size == 0
    assert outputs.data.strategy_profit_mc.shape == terminal_prices.shape
    assert 0.0 < outputs.probability_of_profit < 1.0
