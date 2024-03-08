import pytest
import datetime as dt


@pytest.fixture
def nvidia():
    stockprice = 168.99
    return dict(
        stockprice=stockprice,
        volatility=0.483,
        startdate=dt.date(2023, 1, 16).strftime("%Y-%m-%d"),
        targetdate=dt.date(2023, 2, 17).strftime("%Y-%m-%d"),
        interestrate=0.045,
        minstock=stockprice - 100.0,
        maxstock=stockprice + 100.0,
    )
