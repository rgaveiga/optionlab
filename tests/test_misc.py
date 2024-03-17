import datetime as dt
import time

import pytest

from optionlab.utils import get_nonbusiness_days


def test_holidays():
    start_date = dt.date(2024, 1, 1)
    end_date = dt.date(2024, 12, 31)

    us_nonbusiness_days = get_nonbusiness_days(start_date, end_date, country="US")

    assert us_nonbusiness_days == 115

    china_nonbusiness_days = get_nonbusiness_days(start_date, end_date, country="China")

    assert china_nonbusiness_days == 123

    brazil_nonbusiness_days = get_nonbusiness_days(
        start_date, end_date, country="Brazil"
    )

    assert brazil_nonbusiness_days == 109

    germany_nonbusiness_days = get_nonbusiness_days(
        start_date, end_date, country="Germany"
    )

    assert germany_nonbusiness_days == 113

    uk_nonbusiness_days = get_nonbusiness_days(start_date, end_date, country="UK")

    assert uk_nonbusiness_days == 110


@pytest.mark.benchmark
def test_holidays_benchmark(days: int = 366):

    start_date = dt.date(2024, 1, 1)

    for i in range(days):
        end_date = start_date + dt.timedelta(days=1)

        get_nonbusiness_days(start_date, end_date, country="US")


def test_benchmark_holidays(benchmark):
    start_time = time.time()
    benchmark(test_holidays_benchmark)

    assert time.time() - start_time < 2  # takes avg. ~1.1ms on M1
