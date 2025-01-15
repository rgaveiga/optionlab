import datetime as dt

import pytest

from optionlab.models import Inputs
from numpy import array


def test_only_one_closed_position(nvidia):
    inputs = nvidia | {
        # The covered call strategy is defined
        "strategy": [
            {"type": "closed", "prev_pos": 100},
            {"type": "closed", "prev_pos": 100},
        ],
    }

    with pytest.raises(ValueError) as err:
        Inputs.model_validate(inputs)

    assert "Only one position of type 'closed' is allowed!" in str(err.value)


def test_validate_dates(nvidia):
    strategy = [{"type": "closed", "prev_pos": 100}]
    inputs = nvidia | {
        "start_date": dt.date(2023, 1, 14),
        "target_date": dt.date(2023, 1, 10),
        "strategy": strategy,
    }

    with pytest.raises(ValueError) as err:
        Inputs.model_validate(inputs)

    assert "Start date must be before target date!" in str(err.value)

    inputs = nvidia | {
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
    }

    with pytest.raises(ValueError) as err:
        Inputs.model_validate(inputs)

    assert "Expiration dates must be after or on target date!" in str(err.value)

    inputs = nvidia | {
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
    }

    with pytest.raises(ValueError) as err:
        Inputs.model_validate(inputs)

    assert "You can't mix a strategy expiration with a days_to_target_date." in str(
        err.value
    )


def test_array_distribution_with_no_array(nvidia):
    inputs = nvidia | {
        "model": "array",
        "strategy": [
            {"type": "closed", "prev_pos": 100},
        ],
    }

    with pytest.raises(ValueError) as err:
        Inputs.model_validate(inputs)

    assert (
        "Array of terminal stock prices must be provided if model is 'array'."
        in str(err.value)
    )

    inputs |= {"array": array([])}

    with pytest.raises(ValueError) as err:
        Inputs.model_validate(inputs)

    assert (
        "Array of terminal stock prices must be provided if model is 'array'."
        in str(err.value)
    )
