from __future__ import division

import datetime as dt
from datetime import timedelta
from functools import lru_cache

from holidays import country_holidays

from optionlab.models import Country


@lru_cache
def get_nonbusiness_days(
    start_date: dt.date, end_date: dt.date, country: Country = "US"
):
    """
    get_nonbusiness_days -> returns the number of non-business days between
    the start and end date.

    Arguments
    ---------
    start_date: Start date, provided as a 'datetime.date' object.
    end_date: End date, provided as a 'datetime.date' object.
    country: Country for which the holidays will be counted as non-business days
             (default is "US").
    """

    if end_date > start_date:
        n_days = (end_date - start_date).days
    else:
        raise ValueError("End date must be after start date!")

    nonbusiness_days = 0
    holidays = country_holidays(country)

    for i in range(n_days):
        current_date = start_date + timedelta(days=i)

        if current_date.weekday() >= 5 or current_date.strftime("%Y-%m-%d") in holidays:
            nonbusiness_days += 1

    return nonbusiness_days
