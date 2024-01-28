from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field

OptionType = Literal["call", "put"]
Range = tuple[float, float]
Country = Literal[
    "US",
    "Canada",
    "Mexico",
    "Brazil",
    "China",
    "India",
    "India",
    "South Korea",
    "Russia",
    "Japan",
    "UK",
    "France",
    "Germany",
    "Italy",
    "Australia",
]


class BaseStrategy(BaseModel):
    action: Literal["buy", "sell"]
    prevpos: float | None = None


class StockStrategy(BaseStrategy):
    type: Literal["stock"]
    n: int
    premium: float | None = None


class OptionStrategy(BaseStrategy):
    type: OptionType
    strike: float
    premium: float
    n: int
    expiration: str | int | None = None


class ClosedPosition(BaseModel):
    type: Literal["closed"]
    prevpos: float


Strategy = StockStrategy | OptionStrategy | ClosedPosition


class Inputs(BaseModel):
    """
    stockprice : float
            Spot price of the underlying.
        volatility : float
            Annualized volatility.
        interestrate : float
            Annualized risk-free interest rate.
        minstock : float
            Minimum value of the stock in the stock price domain.
        maxstock : float
            Maximum value of the stock in the stock price domain.
        strategy : list
            A Python list containing the strategy legs as Python dictionaries.
            For options, the dictionary should contain up to 7 keys:
                "type" : string
                    Either 'call' or 'put'. It is mandatory.
                "strike" : float
                    Option strike price. It is mandatory.
                "premium" : float
                    Option premium. It is mandatory.
                "n" : int
                    Number of options. It is mandatory
                "action" : string
                    Either 'buy' or 'sell'. It is mandatory.
                "prevpos" : float
                    Premium effectively paid or received in a previously opened
                    position. If positive, it means that the position remains
                    open and the payoff calculation takes this price into
                    account, not the current price of the option. If negative,
                    it means that the position is closed and the difference
                    between this price and the current price is considered in
                    the payoff calculation.
                "expiration" : string | int
                    Expiration date in 'YYYY-MM-DD' format or number of days
                    left before maturity, depending on the value in 'use_dates'
                    (see below).
            For stocks, the dictionary should contain up to 4 keys:
                "type" : string
                    It must be 'stock'. It is mandatory.
                "n" : int
                    Number of shares. It is mandatory.
                "action" : string
                    Either 'buy' or 'sell'. It is mandatory.
                "prevpos" : float
                    Stock price effectively paid or received in a previously
                    opened position. If positive, it means that the position
                    remains open and the payoff calculation takes this price
                    into account, not the current price of the stock. If
                    negative, it means that the position is closed and the
                    difference between this price and the current price is
                    considered in the payoff calculation.
            For a non-determined previously opened position to be closed, which
            might consist of any combination of calls, puts and stocks, the
            dictionary must contain two keys:
                "type" : string
                    It must be 'closed'. It is mandatory.
                "prevpos" : float
                    The total value of the position to be closed, which can be
                    positive if it made a profit or negative if it is a loss.
                    It is mandatory.
        dividendyield : float, optional
            Annualized dividend yield. Default is 0.0.
        profittarg : float, optional
            Target profit level. Default is None, which means it is not
            calculated.
        losslimit : float, optional
            Limit loss level. Default is None, which means it is not calculated.
        optcommission : float
            Broker commission for options transactions. Default is 0.0.
        stockcommission : float
            Broker commission for stocks transactions. Default is 0.0.
        compute_the_greeks : logical, optional
            Whether or not Black-Scholes formulas should be used to compute the
            Greeks. Default is False.
        compute_expectation : logical, optional
            Whether or not the strategy's average profit and loss must be
            computed from a numpy array of random terminal prices generated from
            the chosen distribution. Default is False.
        use_dates : logical, optional
            Whether the target and maturity dates are provided or not. If False,
            the number of days remaining to the target date and maturity are
            provided. Default is True.
        discard_nonbusinessdays : logical, optional
            Whether to discard Saturdays and Sundays (and maybe holidays) when
            counting the number of days between two dates. Default is True.
        country : string, optional
            Country for which the holidays will be considered if 'discard_nonbusinessdyas'
            is True. Default is 'US'.
        startdate : string, optional
            Start date in the calculations, in 'YYYY-MM-DD' format. Default is "".
            Mandatory if 'use_dates' is True.
        targetdate : string, optional
            Target date in the calculations, in 'YYYY-MM-DD' format. Default is "".
            Mandatory if 'use_dates' is True.
        days2targetdate : int, optional
            Number of days remaining until the target date. Not considered if
            'use_dates' is True. Default is 30 days.
        distribution : string, optional
            Statistical distribution used to compute probabilities. It can be
            'black-scholes', 'normal', 'laplace' or 'array'. Default is 'black-scholes'.
        nmcprices : int, optional
            Number of random terminal prices to be generated when calculationg
            the average profit and loss of a strategy. Default is 100,000.
    """

    stockprice: float = Field(gt=0)
    volatility: float
    interestrate: float = Field(gt=0, le=0.2)
    minstock: float
    maxstock: float
    strategy: list[Strategy] = Field(..., discriminator="type")
    dividendyield: float = 0.0
    profittarg: float | None = None
    losslimit: float | None = None
    optcommission: float = 0.0
    stockcommission: float = 0.0
    compute_expectation: bool = False
    use_dates: bool = True
    discard_nonbusinessdays: bool = True
    country: Country = "US"
    startdate: str = ""
    targetdate: str = ""
    days2targetdate: int = 30
    distribution: Literal["black-scholes", "normal", "laplace", "array"] = (
        "black-scholes"
    )
    nmcprices: float = 100000


class BlackScholesInfo(BaseModel):
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    call_theta: float
    put_theta: float
    gamma: float
    vega: float
    call_itm_prob: float
    put_itm_prob: float


class Outputs(BaseModel):
    """
    probability_of_profit : float
        Probability of the strategy yielding at least $0.01.
    profit_ranges : list
        A Python list of minimum and maximum stock prices defining
        ranges in which the strategy makes at least $0.01.
    strategy_cost : float
        Total strategy cost.
    per_leg_cost : list
        A Python list of costs, one per strategy leg.
    implied_volatility : list
        A Python list of implied volatilities, one per strategy leg.
    in_the_money_probability : list
        A Python list of ITM probabilities, one per strategy leg.
    delta : list
        A Python list of Delta values, one per strategy leg.
    gamma : list
        A Python list of Gamma values, one per strategy leg.
    theta : list
        A Python list of Theta values, one per strategy leg.
    vega : list
        A Python list of Vega values, one per strategy leg.
    minimum_return_in_the_domain : float
        Minimum return of the strategy within the stock price domain.
    maximum_return_in_the_domain : float
        Maximum return of the strategy within the stock price domain.
    probability_of_profit_target : float
        Probability of the strategy yielding at least the profit target.
    project_target_ranges : list
        A Python list of minimum and maximum stock prices defining
        ranges in which the strategy makes at least the profit
        target.
    probability_of_loss_limit : float
        Probability of the strategy losing at least the loss limit.
    average_profit_from_mc : float
        Average profit as calculated from Monte Carlo-created terminal
        stock prices for which the strategy is profitable.
    average_loss_from_mc : float
        Average loss as calculated from Monte Carlo-created terminal
        stock prices for which the strategy ends in loss.
    probability_of_profit_from_mc : float
        Probability of the strategy yielding at least $0.01 as calculated
        from Monte Carlo-created terminal stock prices.
    """

    probability_of_profit: float
    profit_ranges: list[Range]
    per_leg_cost: Range
    strategy_cost: float
    minimum_return_in_the_domain: float
    maximum_return_in_the_domain: float
    implied_volatility: Range
    in_the_money_probability: Range
    delta: Range
    gamma: Range
    theta: Range
    vega: Range
    probability_of_profit_target: float | None = None
    project_target_ranges: Range | None = None
    probability_of_loss_limit: float | None = None
    average_profit_from_mc: float | None = None
    average_loss_from_mc: float | None = None
    probability_of_profit_from_mc: float | None = None
