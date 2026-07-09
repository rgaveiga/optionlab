import datetime as dt

import pytest

from optionlab import run_strategy
from optionlab.models import Inputs, Option, Stock
from numpy import array

STOCK_LEG = {"type": "stock", "n": 100, "action": "buy"}


def assert_inputs_validation_error(payload, expected_message):
    with pytest.raises(ValueError) as err:
        Inputs.model_validate(payload)

    assert expected_message in str(err.value)


def test_only_one_closed_position(nvidia):
    payload = nvidia | {
        "strategy": [
            {"type": "closed", "prev_pos": 100},
            {"type": "closed", "prev_pos": 100},
        ],
    }

    assert_inputs_validation_error(
        payload,
        "Only one position of type 'closed' is allowed!",
    )


def test_validate_dates(nvidia):
    invalid_payloads = [
        (
            nvidia
            | {
                "start_date": dt.date(2023, 1, 14),
                "target_date": dt.date(2023, 1, 10),
                "strategy": [{"type": "closed", "prev_pos": 100}],
            },
            "Start date must be before target date!",
        ),
        (
            nvidia
            | {
                "start_date": dt.date(2023, 1, 14),
                "target_date": dt.date(2023, 1, 17),
                "strategy": [
                    {
                        "type": "call",
                        "strike": 185.0,
                        "premium": 4.1,
                        "n": 100,
                        "action": "sell",
                        "expiration": dt.date(2023, 1, 16),
                    }
                ],
            },
            "Expiration dates must be after or on target date!",
        ),
        (
            nvidia
            | {
                "start_date": None,
                "target_date": None,
                "days_to_target_date": 30,
                "strategy": [
                    {"type": "stock", "n": 100, "action": "buy"},
                    {
                        "type": "call",
                        "strike": 185.0,
                        "premium": 4.1,
                        "n": 100,
                        "action": "sell",
                        "expiration": dt.date(2023, 1, 17),
                    },
                ],
            },
            "You can't mix a strategy expiration with a days_to_target_date.",
        ),
    ]

    for payload, expected_message in invalid_payloads:
        assert_inputs_validation_error(payload, expected_message)


def test_array_with_no_array(nvidia):
    payload = nvidia | {
        "model": "array",
        "strategy": [
            {"type": "closed", "prev_pos": 100},
        ],
    }
    expected_message = (
        "Array of terminal stock prices must be provided if model is 'array'."
    )

    assert_inputs_validation_error(payload, expected_message)
    assert_inputs_validation_error(payload | {"array": array([])}, expected_message)


def test_stock_leg_field_bounds():
    with pytest.raises(ValueError):
        Stock.model_validate(STOCK_LEG | {"n": 0})

    with pytest.raises(ValueError):
        Stock.model_validate(STOCK_LEG | {"action": "hold"})


def test_option_leg_field_bounds():
    option_leg = {
        "type": "call",
        "strike": 185.0,
        "premium": 4.1,
        "n": 100,
        "action": "sell",
    }

    for invalid in ({"n": 0}, {"strike": 0.0}, {"premium": 0.0}, {"type": "future"}):
        with pytest.raises(ValueError):
            Option.model_validate(option_leg | invalid)


def test_option_expiration_as_nonpositive_int():
    option_leg = {
        "type": "call",
        "strike": 185.0,
        "premium": 4.1,
        "n": 100,
        "action": "sell",
    }

    with pytest.raises(ValueError) as err:
        Option.model_validate(option_leg | {"expiration": 0})

    assert "must be greater than 0" in str(err.value)


def test_inputs_field_bounds(nvidia):
    payload = nvidia | {"strategy": [STOCK_LEG]}

    for invalid in (
        {"stock_price": 0.0},
        {"volatility": -0.1},
        {"interest_rate": -0.01},
        {"min_stock": -1.0},
        {"max_stock": -1.0},
        {"strategy": []},
    ):
        assert_inputs_validation_error(payload | invalid, "")


def test_no_dates_and_no_days_to_target(nvidia):
    payload = nvidia | {
        "start_date": None,
        "target_date": None,
        "strategy": [STOCK_LEG],
    }

    assert_inputs_validation_error(
        payload,
        "Either start_date and target_date or days_to_maturity must be provided",
    )


def test_outputs_str_excludes_internal_data(nvidia):
    outputs = run_strategy(
        nvidia
        | {
            "strategy": [
                {
                    "type": "call",
                    "strike": 185.0,
                    "premium": 4.1,
                    "n": 100,
                    "action": "sell",
                }
            ]
        }
    )

    text = str(outputs)

    assert "Probability of profit" in text
    assert "stock_price_array" not in text
    assert "inputs" not in text
