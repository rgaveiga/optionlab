import datetime as dt

import pytest

from optionlab.models import Inputs
from numpy import array


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
    expected_message = "Array of terminal stock prices must be provided if model is 'array'."

    assert_inputs_validation_error(payload, expected_message)
    assert_inputs_validation_error(payload | {"array": array([])}, expected_message)
