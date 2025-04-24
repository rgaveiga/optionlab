import datetime as dt
import time

import pytest

from optionlab.models import BlackScholesModelInputs
from optionlab.price_array import create_price_array, _get_array_price_from_BS
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
    _get_array_price_from_BS.cache_clear()

    stock_price = 168.99
    volatility = 0.483
    interest_rate = 0.045
    years_to_target = 24 / 365

    sample1 = create_price_array(
        inputs_data=BlackScholesModelInputs(
            stock_price=stock_price,
            volatility=volatility,
            interest_rate=interest_rate,
            years_to_target_date=years_to_target,
        ),
        seed=0,
    )

    # cache_info1 = create_price_samples.cache_info()
    # assert cache_info1.misses == 1
    # assert cache_info1.hits == 0
    # assert cache_info1.currsize == 1
    assert sample1.sum() == pytest.approx(16951655.848562226, rel=0.01)

    sample2 = create_price_array(
        inputs_data=BlackScholesModelInputs(
            stock_price=stock_price,
            volatility=volatility,
            interest_rate=interest_rate,
            years_to_target_date=years_to_target,
        ),
        seed=1,
    )

    # cache_info2 = create_price_samples.cache_info()
    # assert cache_info2.misses == 2
    # assert cache_info2.hits == 0
    # assert cache_info2.currsize == 2
    assert sample2.sum() == pytest.approx(16959678.71517979, rel=0.01)

    stock_price = 167.0

    sample3 = create_price_array(
        inputs_data={
            "model": "black-scholes",
            "stock_price": stock_price,
            "volatility": volatility,
            "interest_rate": interest_rate,
            "years_to_target_date": years_to_target,
        },
        seed=0,
    )

    # cache_info3 = create_price_samples.cache_info()
    # assert cache_info3.misses == 2
    # assert cache_info3.hits == 1
    # assert cache_info3.currsize == 2
    assert sample3.sum() == pytest.approx(16752035.781465728, rel=0.01)

    sample4 = create_price_array(
        inputs_data={
            "model": "laplace",
            "stock_price": 168.99,
            "volatility": 0.483,
            "mu": 0.05,
            "years_to_target_date": 24 / 365,
        },
        seed=0,
    )

    # cache_info4 = create_price_samples.cache_info()
    # assert cache_info4.misses == 3
    # assert cache_info4.hits == 1
    # assert cache_info4.currsize == 3
    assert sample4.sum() == pytest.approx(17083995.574185822, rel=0.01)
