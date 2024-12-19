import datetime as dt
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

OptionType = Literal["call", "put"]
Action = Literal["buy", "sell"]
StrategyType = Literal["stock"] | OptionType | Literal["closed"]
Range = tuple[float, float]
Distribution = Literal["black-scholes", "normal", "laplace", "array"]
# Country = Literal[   #TODO: Remove this.
#     "US",
#     "Canada",
#     "Mexico",
#     "Brazil",
#     "China",
#     "India",
#     "SouthKorea",
#     "Russia",
#     "Japan",
#     "UK",
#     "France",
#     "Germany",
#     "Italy",
#     "Australia"
# ]


class BaseLeg(BaseModel):
    n: int = Field(gt=0)
    action: Action
    prev_pos: float | None = None


class Stock(BaseLeg):
    """
    Defines the attributes of a stock leg in a strategy.

    Attributes
    ----------
    type : str
        It must be 'stock'. It is mandatory.
    n : int
        Number of shares. It is mandatory.
    action : str
        `Action` literal value, which must be either 'buy' or 'sell'. It is mandatory.
    prev_pos : float | None, optional
        Stock price effectively paid or received in a previously opened position.
        If positive, the position remains open and the payoff calculation considers
        this price instead of the current stock price. If negative, the position
        is closed and the difference between this price and the current price is
        included in the payoff calculation. The default is None, which means this
        stock position is not a previously opened position.
    """

    type: Literal["stock"] = "stock"


class Option(BaseLeg):
    """
    Defines the attributes of an option leg in a strategy.

    Attributes
    ----------
    type : str
        `OptionType` literal value, which must be either 'call' or 'put'. It is
        mandatory.
    strike : float
        Option strike price. It is mandatory.
    premium : float
        Option premium. It is mandatory.
    n : int
        Number of options. It is mandatory
    action : str
        `Action` literal value, which must be either 'buy' or 'sell'.
    prev_pos : float | None, optional
        Premium effectively paid or received in a previously opened position. If
        positive, the position remains open and the payoff calculation considers
        this price instead of the current price of the option. If negative, the
        position is closed and the difference between this price and the current
        price is included in the payoff calculation. The default is None, which
        means this option position is not a previously opened position.
    expiration : str | int | None, optional.
        Expiration date or number of days remaining to maturity. The default is
        None.
    """

    type: OptionType
    strike: float = Field(gt=0)
    premium: float = Field(gt=0)
    expiration: dt.date | int | None = None

    @field_validator("expiration")
    def validate_expiration(cls, v: dt.date | int | None) -> dt.date | int | None:
        if isinstance(v, int) and v <= 0:
            raise ValueError("If expiration is an integer, it must be greater than 0.")
        return v


class ClosedPosition(BaseModel):
    """
    Defines the attributes of a previously closed position in a strategy.

    Attributes
    ----------
    type : str
        It must be 'closed'. It is mandatory.
    prev_pos : float
        The total amount of the closed position. If positive, it resulted in a
        profit; if negative, it incurred a loss. It is mandatory.
    """

    type: Literal["closed"] = "closed"
    prev_pos: float


StrategyLeg = Stock | Option | ClosedPosition


# TODO: Check the conditions imposed to interest-rate and its optionality; the validator can be removed
class ProbabilityOfProfitInputs(BaseModel):
    """
    Defines the input for the probability of profit calculation from a statistical
    distribution.

    Attributes
    ----------
    source : str
        Statistical distribution. It can be 'black-scholes' (the same as 'normal')
        or 'laplace'.
    stock_price : float
        Stock price.
    volatility : float
        Annualized volatility of the underlying asset.
    years_to_maturity : float
        Time remaining to maturity, in years.
    interest_rate : float
        Annualized risk-free interest rate.
    dividend_yield : float
        Annualized dividend yield.
    """

    source: Literal["black-scholes", "normal", "laplace"]
    stock_price: float = Field(gt=0)
    volatility: float = Field(gt=0, le=1)
    years_to_maturity: float = Field(gt=0)
    interest_rate: float | None = Field(None, gt=0, le=0.2)
    dividend_yield: float = Field(0, ge=0, le=1)

    # @model_validator(mode="after")
    # def validate_black_scholes_model(self) -> "ProbabilityOfProfitInputs":
    #     if self.source == "black-scholes" and not self.interest_rate:
    #         raise ValueError(
    #             "Risk-free interest rate must be provided for 'black-scholes' PoP calculations!"
    #         )
    #     return self


# TODO: Check if source is really required here.
class ProbabilityOfProfitArrayInputs(BaseModel):
    """
    Defines the input for the probability of profit calculation when using an
    array of terminal stock prices provided by the user.

    Attributes
    ----------
    array : numpy.ndarray
        Array of terminal stock prices.
    """

    #    source: Literal["array"] = "array"   # It seems it is not necessary.
    array: np.ndarray
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("array", mode="before")
    @classmethod
    def validate_arrays(cls, v: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(v)
        if arr.shape[0] == 0:
            raise ValueError("The array of stock prices is empty!")
        return arr


class Inputs(BaseModel):
    """
    Defines the input data for a strategy calculation.

    Attributes
    ----------
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
        A list of strategy legs.
    dividend_yield : float, optional
        Annualized dividend yield. The default is 0.0.
    profit_target : float, optional
        Target profit level. The default is None, which means it is not
        calculated.
    loss_limit : float, optional
        Limit loss level. The default is None, which means it is not calculated.
    opt_commission : float
        Brokerage commission for options transactions. The default is 0.0.
    stock_commission : float
        Brokerage commission for stocks transactions. The default is 0.0.
    compute_expectation : bool, optional
        Computes the strategy's average profit and loss from a numpy array of random
        terminal prices generated from a distribution. The default is False.
    discard_nonbusinessdays : bool, optional
        Discards weekends and holidays when counting the number of days between
        two dates. The default is True.
    country : str, optional
        Country whose holidays will be counted if `discard_nonbusinessdays` is
        set to True. The default is 'US'.
    start_date : datetime.date, optional
        Start date in the calculations. If not provided, `days_to_target_date`
        must be provided.
    target_date : datetime.date, optional
        Target date in the calculations. If not provided, `days_to_target_date`
        must be provided.
    days_to_target_date : int, optional
        Days remaining to the target date. If not provided, `start_date` and
        `target_date` must be provided.
    distribution : str, optional
        Statistical distribution used to compute probabilities. It can be
        'black-scholes' (the same as 'normal'), 'laplace' or 'array'. The default
        is 'black-scholes'.
    mc_prices_number : int, optional
        Number of random terminal prices to be generated when calculationg
        the average profit and loss of a strategy. Default is 100,000.
    """

    stock_price: float = Field(gt=0)
    volatility: float
    interest_rate: float = Field(gt=0, le=0.2)
    min_stock: float
    max_stock: float
    strategy: list[StrategyLeg] = Field(..., min_length=1)
    dividend_yield: float = 0.0
    profit_target: float | None = None
    loss_limit: float | None = None
    opt_commission: float = 0.0
    stock_commission: float = 0.0
    compute_expectation: bool = False
    discard_nonbusiness_days: bool = True
    country: str = "US"
    start_date: dt.date | None = None
    target_date: dt.date | None = None
    days_to_target_date: int = Field(0, ge=0)
    distribution: Distribution = "black-scholes"
    mc_prices_number: int = 100_000
    array_prices: list[float] | None = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: list[StrategyLeg]) -> list[StrategyLeg]:
        types = [strategy.type for strategy in v]
        if types.count("closed") > 1:
            raise ValueError("Only one position of type 'closed' is allowed!")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "Inputs":
        expiration_dates = [
            strategy.expiration
            for strategy in self.strategy
            if isinstance(strategy, Option) and isinstance(strategy.expiration, dt.date)
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

    @model_validator(mode="after")
    def validate_compute_expectation(self) -> "Inputs":
        if self.distribution != "array":
            return self
        if not self.array_prices:
            raise ValueError(
                "Array of prices must be provided if distribution is 'array'."
            )
        if len(self.array_prices) == 0:
            raise ValueError(
                "Array of prices must be provided if distribution is 'array'."
            )
        return self


class BlackScholesInfo(BaseModel):
    """
    Defines the data returned by a calculation using the Black-Scholes model.

    Attributes
    ----------
    call_price : float
        Price of a call option.
    put_price : float
        Price of a put option.
    call_delta : float
        Delta of a call option.
    put_delta : float
        Delta of a put option.
    gamma : float
        Gamma of an option.
    vega : float
        Vega of an option.
    call_itm_prob : float
        In-the-money probability of a call option.
    put_itm_prob : float
        In-the-money probability of a put option.
    """

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


# TODO: Remove this.
# class OptionInfo(BaseModel):
#     price: float
#     delta: float
#     theta: float


def init_empty_array() -> np.ndarray:
    return np.array([])


class EngineDataResults(BaseModel):
    stock_price_array: np.ndarray
    terminal_stock_prices: np.ndarray
    profit: np.ndarray = Field(default_factory=init_empty_array)
    profit_mc: np.ndarray = Field(default_factory=init_empty_array)
    strategy_profit: np.ndarray = Field(default_factory=init_empty_array)
    strategy_profit_mc: np.ndarray = Field(default_factory=init_empty_array)
    strike: list[float] = []
    premium: list[float] = []
    n: list[int] = []
    action: list[Action | Literal["n/a"]] = []
    type: list[StrategyType] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EngineData(EngineDataResults):
    inputs: Inputs
    _previous_position: list[float] = []
    _use_bs: list[bool] = []
    _profit_ranges: list[Range] = []
    _profit_target_range: list[Range] = []
    _loss_limit_ranges: list[Range] = []
    _days_to_maturity: list[int] = []
    _days_in_year: int = 365
    days_to_target: int = 30
    implied_volatility: list[float | np.ndarray] = []
    itm_probability: list[float] = []
    delta: list[float] = []
    gamma: list[float] = []
    vega: list[float] = []
    theta: list[float] = []
    cost: list[float] = []
    profit_probability: float = 0.0
    profit_target_probability: float = 0.0
    loss_limit_probability: float = 0.0


class Outputs(BaseModel):
    """
    Defines the output data from a strategy calculation.

    Attributes
    ----------
    probability_of_profit : float
        Probability of the strategy yielding at least $0.01.
    profit_ranges : list
        A list of minimum and maximum stock prices defining ranges in which the
        strategy makes at least $0.01.
    strategy_cost : float
        Total strategy cost.
    per_leg_cost : list
        A list of costs, one per strategy leg.
    implied_volatility : list
        List of implied volatilities, one per strategy leg.
    in_the_money_probability : list
        List of ITM probabilities, one per strategy leg.
    delta : list
        List of Delta values, one per strategy leg.
    gamma : list
        List of Gamma values, one per strategy leg.
    theta : list
        List of Theta values, one per strategy leg.
    vega : list
        List of Vega values, one per strategy leg.
    minimum_return_in_the_domain : float
        Minimum return of the strategy within the stock price domain.
    maximum_return_in_the_domain : float
        Maximum return of the strategy within the stock price domain.
    probability_of_profit_target : float | None, optional
        Probability of the strategy yielding at least the profit target. The
        default is None.
    profit_target_ranges : list | None, optional
        List of minimum and maximum stock prices defining ranges in which the
        strategy makes at least the profit target. The default is None.
    probability_of_loss_limit : float | None, optional
        Probability of the strategy losing at least the loss limit. The default
        is None.
    average_profit_from_mc : float | None, optional
        Average profit calculated from terminal stock prices created by Monte
        Carlo simulations. The default is None.
    average_loss_from_mc : float | None, optional
        Average loss calculated from terminal stock prices created by Monte Carlo
        simulations. The default is None.
    probability_of_profit_from_mc : float | None, optional
        Probability of the strategy yielding at least $0.01 calculated from
        terminal prices created by Monte Carlo simulations. The default is None.
    data : EngineDataResults
        Further data from the strategy calculation that can be used in the
        post-processing of the outputs.
    inputs : Inputs
        Input data used in the strategy calculation.
    """

    inputs: Inputs
    data: EngineDataResults
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
    profit_target_ranges: list[Range] | None = None
    probability_of_loss_limit: float | None = None
    average_profit_from_mc: float | None = None
    average_loss_from_mc: float | None = None
    probability_of_profit_from_mc: float | None = None


# TODO: Remove this.
# @property
# def return_in_the_domain_ratio(self) -> float:
#     return abs(
#         self.maximum_return_in_the_domain / self.minimum_return_in_the_domain
#     )
