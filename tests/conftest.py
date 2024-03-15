import pytest
import datetime as dt


@pytest.fixture
def nvidia():
    stockprice = 168.99
    return dict(
        stock_price=stockprice,
        volatility=0.483,
        start_date=dt.date(2023, 1, 16),
        target_date=dt.date(2023, 2, 17),
        interest_rate=0.045,
        min_stock=stockprice - 100.0,
        max_stock=stockprice + 100.0,
    )
