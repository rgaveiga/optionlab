from __future__ import division

import datetime as dt
from datetime import timedelta
from functools import lru_cache

import numpy as np
from holidays import country_holidays

from optionlab.models import Country, EngineData


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


def get_pl(data: EngineData, leg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    get_pl -> returns the profit/loss profile of either a leg or the whole
    strategy.

    Parameters
    ----------
    leg : int, optional
        Index of the leg. Default is None (whole strategy).

    Returns
    -------
    stock prices : numpy array
        Sequence of stock prices within the bounds of the stock price domain.
    P/L profile : numpy array
        Profit/loss profile of either a leg or the whole strategy.
    """
    if data.profit.size > 0 and leg and leg < data.profit.shape[0]:
        return data.stock_price_array, data.profit[leg]

    return data.stock_price_array, data.strategy_profit


def pl_to_csv(
    data: EngineData, filename: str = "pl.csv", leg: int | None = None
) -> None:
    """
    pl_to_csv -> saves the profit/loss data to a .csv file.

    Parameters
    ----------
    filename : string, optional
        Name of the .csv file. Default is 'pl.csv'.
    leg : int, optional
        Index of the leg. Default is None (whole strategy).

    Returns
    -------
    None.
    """
    if data.profit.size > 0 and leg and leg < data.profit.shape[0]:
        arr = np.stack((data.stock_price_array, data.profit[leg]))
    else:
        arr = np.stack((data.stock_price_array, data.strategy_profit))

    np.savetxt(
        filename, arr.transpose(), delimiter=",", header="StockPrice,Profit/Loss"
    )
