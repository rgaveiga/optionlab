import pytest

from optionlab import run_strategy
from tests.profit_reference import assert_reference
from tests.test_core import COVERED_CALL_LEGS, with_expiration


@pytest.mark.parametrize("price_step", [0.05, 0.20, 0.50, 1.00])
def test_covered_call_price_resolution(nvidia, price_step):
    payload = nvidia | {
        "strategy": with_expiration(COVERED_CALL_LEGS, nvidia["target_date"]),
        "price_step": price_step,
        "calculations": ["pop", "expectation"],
    }

    outputs = run_strategy(payload)
    assert_reference(outputs)

    assert outputs.implied_volatility == []
    assert outputs.delta == []


@pytest.mark.parametrize("price_step", [0.05, 0.20, 0.50, 1.00])
def test_calendar_spread_price_resolution(price_step):
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
    assert_reference(outputs)

    assert outputs.implied_volatility == []
    assert outputs.delta == []
