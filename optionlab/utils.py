from __future__ import division

import datetime as dt
from datetime import timedelta
from functools import lru_cache

import numpy as np
from holidays import country_holidays

from optionlab.models import Outputs


@lru_cache
def get_nonbusiness_days(
    start_date: dt.date, end_date: dt.date, country: str = "US"
) -> int:
    """
    Returns the number of non-business days (i.e., weekends and holidays) between
    the start and end date.

    Parameters
    ----------
    start_date : datetime.date
        Start date.
    end_date : datetime.date
        End date.
    country : str, optional
        Country of the stock exchange. A list of available countries can be found
        in the `holidays library documentation <https://holidays.readthedocs.io/en/latest/>`_.
        The default value is 'US'.

    Returns
    -------
    int
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


def get_pl(outputs: Outputs, leg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the stock prices and the corresponding profit/loss profile of either
    a leg or the whole strategy.

    Parameters
    ----------
    outputs : Outputs
        Output data from a strategy calculation.
    leg : int | None, optional
        Index of a strategy leg. The default is None, which means the whole strategy.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Array of stock prices and array or profits/losses.
    """

    if outputs.data.profit.size > 0 and leg and leg < outputs.data.profit.shape[0]:
        return outputs.data.stock_price_array, outputs.data.profit[leg]

    return outputs.data.stock_price_array, outputs.data.strategy_profit


def pl_to_csv(
    outputs: Outputs, filename: str = "pl.csv", leg: int | None = None
) -> None:
    """
    Saves the stock prices and corresponding profit/loss profile of either a leg
    or the whole strategy to a CSV file.

    Parameters
    ----------
    outputs : Outputs
        Output data from a strategy calculation.
    filename : str, optional
        Name of the CSV file. The default is 'pl.csv'.
    leg : int | None, optional
        Index of a strategy leg. The default is None, which means the whole strategy.

    Returns
    -------
    None.
    """

    if outputs.data.profit.size > 0 and leg and leg < outputs.data.profit.shape[0]:
        arr = np.stack((outputs.data.stock_price_array, outputs.data.profit[leg]))
    else:
        arr = np.stack((outputs.data.stock_price_array, outputs.data.strategy_profit))

    np.savetxt(
        filename, arr.transpose(), delimiter=",", header="StockPrice,Profit/Loss"
    )
