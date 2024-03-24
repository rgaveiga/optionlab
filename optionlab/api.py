import datetime as dt

import pandas as pd
from yfinance import Ticker

from optionlab.models import OptionsChain


def get_options_chain(ticker: str, expiration_date: dt.date) -> OptionsChain:
    stock = Ticker(ticker)
    res = stock.option_chain(expiration_date.strftime("%Y-%m-%d"))
    return OptionsChain(
        calls=res.calls,
        puts=res.puts,
        underlying=res.underlying,
    )


def get_stock_history(ticker: str, num_of_months: int) -> pd.DataFrame:
    stock = Ticker(ticker)
    return stock.history(period=f"{num_of_months}mo")
