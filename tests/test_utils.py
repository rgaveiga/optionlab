import datetime as dt

import numpy as np
import pytest

from optionlab import run_strategy
from optionlab.utils import get_nonbusiness_days, get_pl, pl_to_csv

COVERED_CALL_LEGS = [
    {"type": "stock", "n": 100, "action": "buy"},
    {"type": "call", "strike": 185.0, "premium": 4.1, "n": 100, "action": "sell"},
]


@pytest.fixture
def covered_call_outputs(nvidia):
    return run_strategy(nvidia | {"strategy": COVERED_CALL_LEGS})


def test_nonbusiness_days_with_us_holiday():
    # 2023-01-16 (MLK Day) through 2023-02-17: 8 weekend days + 1 holiday
    assert get_nonbusiness_days(dt.date(2023, 1, 16), dt.date(2023, 2, 17)) == 9


def test_nonbusiness_days_plain_week():
    # Monday to next Monday, no holidays: one weekend
    assert get_nonbusiness_days(dt.date(2023, 3, 6), dt.date(2023, 3, 13)) == 2


def test_nonbusiness_days_country_specific():
    # Tiradentes (2023-04-21, Friday) is a holiday in Brazil but not in the US
    start, end = dt.date(2023, 4, 17), dt.date(2023, 4, 24)

    us_count = get_nonbusiness_days(start, end, country="US")
    br_count = get_nonbusiness_days(start, end, country="BR")

    assert us_count == 2
    assert br_count == 3


def test_nonbusiness_days_invalid_range():
    with pytest.raises(ValueError):
        get_nonbusiness_days(dt.date(2023, 1, 16), dt.date(2023, 1, 16))

    with pytest.raises(ValueError):
        get_nonbusiness_days(dt.date(2023, 1, 16), dt.date(2023, 1, 10))


def test_get_pl_whole_strategy(covered_call_outputs):
    s, pl = get_pl(covered_call_outputs)

    assert np.array_equal(s, covered_call_outputs.data.stock_price_array)
    assert np.array_equal(pl, covered_call_outputs.data.strategy_profit)


def test_get_pl_leg_zero(covered_call_outputs):
    s, pl = get_pl(covered_call_outputs, leg=0)

    assert np.array_equal(pl, covered_call_outputs.data.profit[0])
    assert not np.array_equal(pl, covered_call_outputs.data.strategy_profit)


def test_get_pl_leg_one(covered_call_outputs):
    _, pl = get_pl(covered_call_outputs, leg=1)

    assert np.array_equal(pl, covered_call_outputs.data.profit[1])


def test_get_pl_leg_out_of_range_falls_back_to_strategy(covered_call_outputs):
    _, pl = get_pl(covered_call_outputs, leg=5)

    assert np.array_equal(pl, covered_call_outputs.data.strategy_profit)


def test_pl_to_csv_whole_strategy(covered_call_outputs, tmp_path):
    filename = str(tmp_path / "pl.csv")

    pl_to_csv(covered_call_outputs, filename=filename)

    with open(filename) as f:
        header = f.readline()

    assert "StockPrice,Profit/Loss" in header

    arr = np.loadtxt(filename, delimiter=",", skiprows=1)

    assert np.allclose(arr[:, 0], covered_call_outputs.data.stock_price_array)
    assert np.allclose(arr[:, 1], covered_call_outputs.data.strategy_profit)


def test_pl_to_csv_leg_zero(covered_call_outputs, tmp_path):
    filename = str(tmp_path / "pl_leg0.csv")

    pl_to_csv(covered_call_outputs, filename=filename, leg=0)

    arr = np.loadtxt(filename, delimiter=",", skiprows=1)

    assert np.allclose(arr[:, 1], covered_call_outputs.data.profit[0])
