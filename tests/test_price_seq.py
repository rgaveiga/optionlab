import pytest

from optionlab import run_strategy
from tests.test_core import COVERED_CALL_LEGS, with_expiration


@pytest.mark.parametrize(
    ("price_step", "expected_pop", "expected_profit"),
    [
        (0.05, 0.5465561805405352, 1449.08),
        (0.20, 0.5457504007541361, 1449.08),
        (0.50, 0.5457504007541361, 1449.08),
        (1.00, 0.5457504007541361, 1449.07),
    ],
)
def test_covered_call_price_resolution(
    nvidia, price_step, expected_pop, expected_profit
):
    payload = nvidia | {
        "strategy": with_expiration(COVERED_CALL_LEGS, nvidia["target_date"]),
        "price_step": price_step,
        "calculations": ["pop", "expectation"],
    }

    outputs = run_strategy(payload)

    assert outputs.probability_of_profit == pytest.approx(expected_pop)
    assert outputs.expected_profit_if_profitable == pytest.approx(expected_profit)
    assert outputs.expected_loss_if_unprofitable == pytest.approx(-1703.74)
    assert outputs.implied_volatility == []
    assert outputs.delta == []


@pytest.mark.parametrize(
    ("price_step", "expected_pop", "expected_profit", "expected_loss"),
    [
        (0.05, 0.5996008032388984, 1380.88, -692.87),
        (0.20, 0.5965539037891692, 1380.80, -692.85),
        (0.50, 0.5910310742609183, 1380.19, -692.85),
        (1.00, 0.5627644236750425, 1374.10, -692.78),
    ],
)
def test_calendar_spread_price_resolution(
    price_step, expected_pop, expected_profit, expected_loss
):
    payload = {
        "stock_price": 127.14,
        "start_date": "2021-01-18",
        "target_date": "2021-01-29",
        "volatility": 0.427,
        "interest_rate": 0.0009,
        "min_stock": 63.57,
        "max_stock": 190.71,
        "strategy": [
            {
                "type": "call",
                "strike": 127.00,
                "premium": 4.60,
                "n": 1000,
                "action": "sell",
            },
            {
                "type": "call",
                "strike": 127.00,
                "premium": 5.90,
                "n": 1000,
                "action": "buy",
                "expiration": "2021-02-12",
            },
        ],
        "price_step": price_step,
        "calculations": ["pop", "expectation"],
    }

    outputs = run_strategy(payload)

    assert outputs.probability_of_profit == pytest.approx(expected_pop)
    assert outputs.expected_profit_if_profitable == pytest.approx(expected_profit)
    assert outputs.expected_loss_if_unprofitable == pytest.approx(expected_loss)
    assert outputs.implied_volatility == []
    assert outputs.delta == []
