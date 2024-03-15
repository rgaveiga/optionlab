import datetime as dt

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
