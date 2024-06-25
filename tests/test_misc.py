import datetime as dt
import time

import pytest

from optionlab import create_price_samples
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


def test_cache_price_samples():
    create_price_samples.cache_clear()

    sample1 = create_price_samples(168.99, 0.483, 23 / 365, 0.045)

    cache_info1 = create_price_samples.cache_info()
    assert cache_info1.misses == 1
    assert cache_info1.hits == 0
    assert cache_info1.currsize == 1
    assert sample1.sum() == pytest.approx(16955828.375046223, rel=0.01)

    sample2 = create_price_samples(168.99, 0.483, 23 / 365, 0.045)

    cache_info2 = create_price_samples.cache_info()
    assert cache_info2.misses == 1
    assert cache_info2.hits == 1
    assert cache_info2.currsize == 1
    assert sample2.sum() == pytest.approx(16955828.375046223, rel=0.01)

    sample3 = create_price_samples(167, 0.483, 23 / 365, 0.045)

    cache_info3 = create_price_samples.cache_info()
    assert cache_info3.misses == 2
    assert cache_info3.hits == 1
    assert cache_info3.currsize == 2
    assert sample3.sum() == pytest.approx(16741936.007518211, rel=0.01)
