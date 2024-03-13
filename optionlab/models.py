import datetime as dt
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

OptionType = Literal["call", "put"]
Action = Literal["buy", "sell"]
StrategyType = Literal["stock"] | OptionType | Literal["closed"]
Range = tuple[float, float]
Distribution = Literal["black-scholes", "normal", "laplace", "array"]
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
    action: Action
    prev_pos: float | None = None


class StockStrategy(BaseStrategy):
    """
    "type" : string
        It must be 'stock'. It is mandatory.
    "n" : int
        Number of shares. It is mandatory.
    "action" : string
        Either 'buy' or 'sell'. It is mandatory.
    "prev_pos" : float
        Stock price effectively paid or received in a previously
        opened position. If positive, it means that the position
        remains open and the payoff calculation takes this price
        into account, not the current price of the stock. If
        negative, it means that the position is closed and the
        difference between this price and the current price is
        considered in the payoff calculation.
    """

    type: Literal["stock"] = "stock"
    n: int = Field(gt=0)
    premium: float | None = None


class OptionStrategy(BaseStrategy):
    """
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
    "prev_pos" : float
        Premium effectively paid or received in a previously opened
        position. If positive, it means that the position remains
        open and the payoff calculation takes this price into
        account, not the current price of the option. If negative,
        it means that the position is closed and the difference
        between this price and the current price is considered in
        the payoff calculation.
    "expiration" : string | int, optional.
        Expiration date or days to maturity. If not defined, will use `target_date` or `days_to_target_date`.
    """

    type: OptionType
    strike: float = Field(gt=0)
    premium: float = Field(gt=0)
    n: int = Field(gt=0)
    expiration: dt.date | int | None = None

    @field_validator("expiration")
    def validate_expiration(cls, v: dt.date | int | None) -> dt.date | int | None:
        if isinstance(v, int) and v <= 0:
            raise ValueError("If expiration is an integer, it must be greater than 0.")
        return v


class ClosedPosition(BaseModel):
    """
    "type" : string
        It must be 'closed'. It is mandatory.
    "prev_pos" : float
        The total value of the position to be closed, which can be
        positive if it made a profit or negative if it is a loss.
        It is mandatory.
    """

    type: Literal["closed"] = "closed"
    prev_pos: float


Strategy = StockStrategy | OptionStrategy | ClosedPosition


class Inputs(BaseModel):
    """
    stock_price : float
        Spot price of the underlying.
    volatility : float
        Annualized volatility.
    interest_rate : float
        Annualized risk-free interest rate.
    min_stock : float
        Minimum value of the stock in the stock price domain.
    max_stock : float
        Maximum value of the stock in the stock price domain.
    strategy : list
        A list of `Strategy`
    dividend_yield : float, optional
        Annualized dividend yield. Default is 0.0.
    profit_target : float, optional
        Target profit level. Default is None, which means it is not
        calculated.
    loss_limit : float, optional
        Limit loss level. Default is None, which means it is not calculated.
    opt_commission : float
        Broker commission for options transactions. Default is 0.0.
    stock_commission : float
        Broker commission for stocks transactions. Default is 0.0.
    compute_expectation : logical, optional
        Whether or not the strategy's average profit and loss must be
        computed from a numpy array of random terminal prices generated from
        the chosen distribution. Default is False.
    discard_nonbusinessdays : logical, optional
        Whether to discard Saturdays and Sundays (and maybe holidays) when
        counting the number of days between two dates. Default is True.
    country : string, optional
        Country for which the holidays will be considered if 'discard_nonbusinessdyas'
        is True. Default is 'US'.
    start_date : dt.date, optional
        Start date in the calculations. If not provided, days_to_target_date must be provided.
    target_date : dt.date, optional
        Start date in the calculations. If not provided, days_to_target_date must be provided.
    days_to_target_date : int, optional
        Days to maturity. If not provided, start_date and end_date must be provided.
    distribution : string, optional
        Statistical distribution used to compute probabilities. It can be
        'black-scholes', 'normal', 'laplace' or 'array'. Default is 'black-scholes'.
    nmc_prices : int, optional
        Number of random terminal prices to be generated when calculationg
        the average profit and loss of a strategy. Default is 100,000.
    """

    stock_price: float = Field(gt=0)
    volatility: float
    interest_rate: float = Field(gt=0, le=0.2)
    min_stock: float
    max_stock: float
    strategy: list[Strategy] = Field(..., min_length=1, discriminator="type")
    dividend_yield: float = 0.0
    profit_target: float | None = None
    loss_limit: float | None = None
    opt_commission: float = 0.0
    stock_commission: float = 0.0
    compute_expectation: bool = False
    discard_nonbusiness_days: bool = True
    country: Country = "US"
    start_date: dt.date | None = None
    target_date: dt.date | None = None
    days_to_target_date: int = Field(0, ge=0)
    distribution: Distribution = "black-scholes"
    nmc_prices: float = 100000

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: list[Strategy]) -> list[Strategy]:
        types = [strategy.type for strategy in v]
        if types.count("closed") > 1:
            raise ValueError("Only one position of type 'closed' is allowed!")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "Inputs":
        expiration_dates = [
            strategy.expiration
            for strategy in self.strategy
            if isinstance(strategy, OptionStrategy)
            and isinstance(strategy.expiration, dt.date)
        ]
        if self.start_date and self.target_date:
            if any(
                expiration_date < self.target_date
                for expiration_date in expiration_dates
            ):
                raise ValueError("Expiration dates must be after or on target date!")
            if self.start_date >= self.target_date:
                raise ValueError("Start date must be before target date!")
            return self
        if self.days_to_target_date:
            if len(expiration_dates) > 0:
                raise ValueError(
                    "You can't mix a strategy expiration with a days_to_target_date."
                )
            return self
        raise ValueError(
            "Either start_date and target_date or days_to_maturity must be provided"
        )


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


class OptionInfo(BaseModel):
    price: float
    delta: float
    theta: float


class Outputs(BaseModel):
    """
    probability_of_profit : float
        Probability of the strategy yielding at least $0.01.
    profit_ranges : list
        A list of minimum and maximum stock prices defining
        ranges in which the strategy makes at least $0.01.
    strategy_cost : float
        Total strategy cost.
    per_leg_cost : list
        A list of costs, one per strategy leg.
    implied_volatility : list
        A Python list of implied volatilities, one per strategy leg.
    in_the_money_probability : list
        A list of ITM probabilities, one per strategy leg.
    delta : list
        A list of Delta values, one per strategy leg.
    gamma : list
        A list of Gamma values, one per strategy leg.
    theta : list
        A list of Theta values, one per strategy leg.
    vega : list
        A list of Vega values, one per strategy leg.
    minimum_return_in_the_domain : float
        Minimum return of the strategy within the stock price domain.
    maximum_return_in_the_domain : float
        Maximum return of the strategy within the stock price domain.
    probability_of_profit_target : float, optional
        Probability of the strategy yielding at least the profit target.
    project_target_ranges : list, optional
        A list of minimum and maximum stock prices defining
        ranges in which the strategy makes at least the profit
        target.
    probability_of_loss_limit : float, optional
        Probability of the strategy losing at least the loss limit.
    average_profit_from_mc : float, optional
        Average profit as calculated from Monte Carlo-created terminal
        stock prices for which the strategy is profitable.
    average_loss_from_mc : float, optional
        Average loss as calculated from Monte Carlo-created terminal
        stock prices for which the strategy ends in loss.
    probability_of_profit_from_mc : float, optional
        Probability of the strategy yielding at least $0.01 as calculated
        from Monte Carlo-created terminal stock prices.
    """

    probability_of_profit: float
    profit_ranges: list[Range]
    per_leg_cost: list[float]
    strategy_cost: float
    minimum_return_in_the_domain: float
    maximum_return_in_the_domain: float
    implied_volatility: list[float]
    in_the_money_probability: list[float]
    delta: list[float]
    gamma: list[float]
    theta: list[float]
    vega: list[float]
    probability_of_profit_target: float | None = None
    project_target_ranges: list[Range] | None = None
    probability_of_loss_limit: float | None = None
    average_profit_from_mc: float | None = None
    average_loss_from_mc: float | None = None
    probability_of_profit_from_mc: float | None = None

    @property
    def return_in_the_domain_ratio(self) -> float:
        return abs(
            self.maximum_return_in_the_domain / self.minimum_return_in_the_domain
        )
