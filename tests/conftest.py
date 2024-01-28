import pytest


@pytest.fixture
def nvidia():
    stockprice = 168.99
    return dict(
        stockprice=stockprice,
        volatility=0.483,
        startdate="2023-01-16",
        targetdate="2023-02-17",
        interestrate=0.045,
        minstock=stockprice - 100.0,
        maxstock=stockprice + 100.0,
    )
