import datetime as dt

import pandas as pd

from optionlab.api import get_options_chain, get_stock_history
from optionlab.models import UnderlyingAsset
from optionlab.utils import get_fridays_date


def test_get_options_chain():

    next_friday_date = get_fridays_date()

    try:
        options = get_options_chain("MSFT", next_friday_date)
    except ValueError as err:
        first_available_date = dt.datetime.strptime(
            str(err).split("[")[1].split(", ")[0], "%Y-%m-%d"
        ).date()
        options = get_options_chain("MSFT", first_available_date)

    assert isinstance(options.calls, pd.DataFrame)
    assert isinstance(options.puts, pd.DataFrame)
    assert isinstance(options.underlying, UnderlyingAsset)


def test_get_stock_history():

    hist_df = get_stock_history("MSFT", 1)

    assert isinstance(hist_df, pd.DataFrame)
