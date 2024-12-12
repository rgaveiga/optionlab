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
) -> int:
    """
    Returns the number of non-business days (i.e., weekends and holidays) between 
    the start and end date.

    Parameters
    ----------
    start_date : dt.date
        Start date.
    end_date : dt.date
        End date.
    country : Country, optional
        Country of the stock exchange. The default is "US".

    Returns
    -------
    nonbusiness_days : int
        Number of weekends and holidays between the start and end date.
    """

    if end_date > start_date:
        n_days = (end_date - start_date).days
    else:
        raise ValueError("End date must be after start date!")

    nonbusiness_days: int = 0
    holidays = country_holidays(country)

    for i in range(n_days):
        current_date = start_date + timedelta(days=i)

        if current_date.weekday() >= 5 or current_date.strftime("%Y-%m-%d") in holidays:
            nonbusiness_days += 1

    return nonbusiness_days


def get_pl(data: EngineData, leg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the stock prices and the corresponding profit/loss profile of either 
    a leg or the whole strategy.

    Parameters
    ----------
    data : EngineData
        Stock price and profit/loss data.
    leg : int | None, optional
        Index of a strategy leg. The default is None, which means the whole strategy.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Array of stock prices and array or profits/losses.
    """

    if data.profit.size > 0 and leg and leg < data.profit.shape[0]:
        return data.stock_price_array, data.profit[leg]

    return data.stock_price_array, data.strategy_profit


def pl_to_csv(
    data: EngineData, filename: str = "pl.csv", leg: int | None = None
) -> None:
    """
    Saves the stock prices and corresponding profit/loss profile of either a leg 
    or the whole strategy to a CSV file.

    Parameters
    ----------
    data : EngineData
        Stock price and profit/loss data.
    filename : str, optional
        Name of the CSV file. The default is "pl.csv".
    leg : int | None, optional
        Index of a strategy leg. The default is None, which means the whole strategy.

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
