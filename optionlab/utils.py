from __future__ import division

import datetime as dt
from datetime import timedelta

from optionlab.__holidays__ import getholidays
from optionlab.models import Country


def get_nonbusiness_days(startdate: dt.date, enddate: dt.date, country: Country = "US"):
    """
    get_nonbusiness_days -> returns the number of non-business days between
    the start and end date.

    Arguments
    ---------
    startdate: Start date, provided as a 'datetime.date' object.
    enddate: End date, provided as a 'datetime.date' object.
    country: Country for which the holidays will be counted as non-business days
             (default is "US").
    """

    if enddate > startdate:
        ndays = (enddate - startdate).days
    else:
        raise ValueError("End date must be after start date!")

    nonbusinessdays = 0
    holidays = getholidays(country)

    for i in range(ndays):
        currdate = startdate + timedelta(days=i)

        if currdate.weekday() >= 5 or currdate.strftime("%Y-%m-%d") in holidays:
            nonbusinessdays += 1

    return nonbusinessdays
