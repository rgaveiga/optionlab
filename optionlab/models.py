"""
This module implements Pydantic models that represent inputs and outputs 
of strategy calculations. 

It also implements constants and custom types.

From the user's point of view, the two most important classes that they will use 
to provide input and subsequently process calculation results are `Inputs` and 
`Outputs`, respectively.
"""

import datetime as dt
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

OptionType = Literal["call", "put"]
"""Option type in a strategy leg."""

Action = Literal["buy", "sell"]
"""Action taken in in a strategy leg."""

StrategyLegType = Literal["stock"] | OptionType | Literal["closed"]
"""Type of strategy leg."""

TheoreticalModel = Literal["black-scholes", "array"]
"""
Theoretical model used in probability of profit (PoP) calculations.
"""

Range = tuple[float, float]
"""Range boundaries."""

FloatOrNdarray = float | np.ndarray
"""Float or numpy array custom type."""


def init_empty_array() -> np.ndarray:
    """@private"""

    return np.array([])


class Stock(BaseModel):
    """Defines the attributes of a stock leg in a strategy."""

    type: Literal["stock"] = "stock"
    """It must be *'stock'*."""

    n: int = Field(gt=0)
    """Number of shares."""

    action: Action
    """Either *'buy'* or *'sell'*."""

    prev_pos: Optional[float] = None
    """
    Stock price effectively paid or received in a previously opened position.
    
    - If positive, the position remains open and the payoff calculation considers
    this price instead of the current stock price. 
    
    - If negative, the position is closed and the difference between this price 
    and the current price is included in the payoff calculation. 
    
    The default is `None`, which means this stock position is not a previously 
    opened position.
    """


class Option(BaseModel):
    """Defines the attributes of an option leg in a strategy."""

    type: OptionType
    """Either *'call'* or *'put'*."""

    strike: float = Field(gt=0)
    """Strike price."""

    premium: float = Field(gt=0)
    """Option premium."""

    action: Action
    """Either *'buy'* or *'sell'*."""

    n: int = Field(gt=0)
    """Number of options."""

    prev_pos: Optional[float] = None
    """
    Premium effectively paid or received in a previously opened position. 
    
    - If positive, the position remains open and the payoff calculation considers
    this price instead of the current price of the option. 
    
    - If negative, the position is closed and the difference between this price 
    and the current price is included in the payoff calculation. 
    
    The default is `None`, which means this option position is not a previously 
    opened position.
    """

    expiration: dt.date | int | None = None
    """
    Expiration date or number of days remaining to expiration. 
    
    The default is `None`, which means the expiration is the same as `Inputs.target_date` 
    or `Inputs.days_to_target_date`.
    """

    @field_validator("expiration")
    def validate_expiration(cls, v: dt.date | int | None) -> dt.date | int | None:
        """@private"""

        if isinstance(v, int) and v <= 0:
            raise ValueError("If expiration is an integer, it must be greater than 0.")
        return v


class ClosedPosition(BaseModel):
    """Defines the attributes of a previously closed position in a strategy."""

    type: Literal["closed"] = "closed"
    """It must be *'closed'*."""

    prev_pos: float
    """
    The total amount of the closed position. 
    
    - If positive, it resulted in a profit.
    
    - If negative, it incurred a loss.
    
    This amount will be added to the payoff and taken into account in the strategy 
    calculations.
    """


StrategyLeg = Stock | Option | ClosedPosition
"""Leg in a strategy."""


class TheoreticalModelInputs(BaseModel):
    """Inputs for calculations, such as the probability of profit (PoP)."""

    stock_price: float = Field(gt=0.0)
    """Stock price."""

    volatility: float = Field(gt=0.0)
    """Annualized volatility of the underlying asset."""

    years_to_target_date: float = Field(ge=0.0)
    """Time remaining until target date, in years."""


class BlackScholesModelInputs(TheoreticalModelInputs):
    """Defines the input data for the calculations using the Black-Scholes model."""

    model: Literal["black-scholes"] = "black-scholes"
    """It must be *'black-scholes'*."""

    interest_rate: float = Field(0.0, ge=0.0)
    """
    Annualized risk-free interest rate. 
    
    The default is 0.0.
    """

    dividend_yield: float = Field(0.0, ge=0.0, le=1.0)
    """
    Annualized dividend yield. 
    
    The default is 0.0.
    """

    __hash__ = object.__hash__


class LaplaceInputs(TheoreticalModelInputs):
    """
    Defines the input data for the calculations using a log-Laplace distribution of
    stock prices.
    """

    model: Literal["laplace"] = "laplace"
    """It must be '*laplace*'."""

    mu: float
    """Annualized return of the underlying asset."""

    __hash__ = object.__hash__


class ArrayInputs(BaseModel):
    """
    Defines the input data for the calculations when using an array of strategy
    returns.
    """

    model: Literal["array"] = "array"
    """It must be *'array*'."""

    array: np.ndarray
    """Array of strategy returns."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("array", mode="before")
    @classmethod
    def validate_arrays(cls, v: np.ndarray | list[float]) -> np.ndarray:
        """@private"""

        arr = np.asarray(v)
        if arr.shape[0] == 0:
            raise ValueError("The array is empty!")
        return arr


class Inputs(BaseModel):
    """Defines the input data for a strategy calculation."""

    stock_price: float = Field(gt=0.0)
    """Spot price of the underlying."""

    volatility: float = Field(ge=0.0)
    """Annualized volatility."""

    interest_rate: float = Field(ge=0.0)
    """Annualized risk-free interest rate."""

    min_stock: float = Field(ge=0.0)
    """Minimum value of the stock in the stock price domain."""

    max_stock: float = Field(ge=0.0)
    """Maximum value of the stock in the stock price domain."""

    strategy: list[StrategyLeg] = Field(..., min_length=1)
    """A list of strategy legs."""

    dividend_yield: float = Field(0.0, ge=0.0)
    """
    Annualized dividend yield. 
    
    The default is 0.0.
    """

    profit_target: Optional[float] = None
    """
    Target profit level. 
    
    The default is `None`, which means it is not calculated.
    """

    loss_limit: Optional[float] = None
    """
    Limit loss level. 
    
    The default is `None`, which means it is not calculated.
    """

    opt_commission: float = 0.0
    """
    Brokerage commission for options transactions. 
    
    The default is 0.0.
    """

    stock_commission: float = 0.0
    """
    Brokerage commission for stocks transactions. 
    
    The default is 0.0.
    """

    discard_nonbusiness_days: bool = True
    """
    Discards weekends and holidays when counting the number of days between
    two dates. 
    
    The default is `True`.
    """

    business_days_in_year: int = 252
    """
    Number of business days in a year. 
    
    The default is 252.
    """

    country: str = "US"
    """
    Country whose holidays will be counted if `discard_nonbusinessdays` is
    set to `True`. 
    
    The default is '*US*'.
    """

    start_date: dt.date | None = None
    """
    Start date in the calculations. 
    
    If not provided, `days_to_target_date` must be provided.
    """

    target_date: dt.date | None = None
    """
    Target date in the calculations. 
    
    If not provided, `days_to_target_date` must be provided.
    """

    days_to_target_date: int = Field(0, ge=0)
    """
    Days remaining to the target date. 
    
    If not provided, `start_date` and `target_date` must be provided.
    """

    model: TheoreticalModel = "black-scholes"
    """
    Theoretical model used in the calculations of probability of profit. 
    
    It can be *'black-scholes'* or *'array*'. 
    """

    array: np.ndarray = Field(default_factory=init_empty_array)
    """
    Array of terminal stock prices. 
    
    The default is an empty array.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: list[StrategyLeg]) -> list[StrategyLeg]:
        """@private"""

        types = [strategy.type for strategy in v]
        if types.count("closed") > 1:
            raise ValueError("Only one position of type 'closed' is allowed!")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "Inputs":
        """@private"""

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
    def validate_model_array(self) -> "Inputs":
        """@private"""

        if self.model != "array":
            return self
        elif self.array is None:
            raise ValueError(
                "Array of terminal stock prices must be provided if model is 'array'."
            )
        elif self.array.shape[0] == 0:
            raise ValueError(
                "Array of terminal stock prices must be provided if model is 'array'."
            )
        return self


class BlackScholesInfo(BaseModel):
    """Defines the data returned by a calculation using the Black-Scholes model."""

    call_price: FloatOrNdarray
    """Price of a call option."""

    put_price: FloatOrNdarray
    """Price of a put option."""

    call_delta: FloatOrNdarray
    """Delta of a call option."""

    put_delta: FloatOrNdarray
    """Delta of a put option."""

    call_theta: FloatOrNdarray
    """Theta of a call option."""

    put_theta: FloatOrNdarray
    """Theta of a put option."""

    gamma: FloatOrNdarray
    """Gamma of an option."""

    vega: FloatOrNdarray
    """Vega of an option."""

    call_rho: FloatOrNdarray
    """Rho of a call option."""

    put_rho: FloatOrNdarray
    """Rho of a put option."""

    call_itm_prob: FloatOrNdarray
    """Probability of expiring in-the-money probability of a call option."""

    put_itm_prob: FloatOrNdarray
    """Probability of expiring in-the-money of a put option."""

    call_prob_of_touch: FloatOrNdarray
    """Probability of a call ever getting in-the-money before expiration."""

    put_prob_of_touch: FloatOrNdarray
    """Probability of a put ever getting in-the-money before expiration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EngineDataResults(BaseModel):
    """@private"""

    stock_price_array: np.ndarray
    terminal_stock_prices: np.ndarray = Field(default_factory=init_empty_array)
    profit: np.ndarray = Field(default_factory=init_empty_array)
    profit_mc: np.ndarray = Field(default_factory=init_empty_array)
    strategy_profit: np.ndarray = Field(default_factory=init_empty_array)
    strategy_profit_mc: np.ndarray = Field(default_factory=init_empty_array)
    strike: list[float] = []
    premium: list[float] = []
    n: list[int] = []
    action: list[Action | Literal["n/a"]] = []
    type: list[StrategyLegType] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EngineData(EngineDataResults):
    """@private"""

    inputs: Inputs
    previous_position: list[float] = []
    use_bs: list[bool] = []
    profit_ranges: list[Range] = []
    profit_target_ranges: list[Range] = []
    loss_limit_ranges: list[Range] = []
    days_to_maturity: list[int] = []
    days_in_year: int = 365
    days_to_target: int = 30
    implied_volatility: list[float] = []
    itm_probability: list[float] = []
    probability_of_touch: list[float] = []
    delta: list[float] = []
    gamma: list[float] = []
    vega: list[float] = []
    rho: list[float] = []
    theta: list[float] = []
    cost: list[float] = []
    profit_probability: float = 0.0
    profit_target_probability: float = 0.0
    loss_limit_probability: float = 0.0
    expected_profit: float = 0.0
    expected_loss: float = 0.0


class Outputs(BaseModel):
    """
    Defines the output data from a strategy calculation.
    """

    probability_of_profit: float
    """
    Probability of the strategy yielding at least $0.01.
    """

    profit_ranges: list[Range]
    """
    A list of minimum and maximum stock prices defining ranges in which the
    strategy makes at least $0.01.
    """

    expected_profit_if_profitable: float = 0.0
    """
    Expected profit when the strategy is profitable. 
    
    The default is 0.0.
    """

    expected_loss_if_unprofitable: float = 0.0
    """
    Expected loss when the strategy is not profitable.
    
    The default is 0.0.
    """

    per_leg_cost: list[float]
    """
    List of leg costs.
    """

    strategy_cost: float
    """
    Total strategy cost.
    """

    minimum_return_in_the_domain: float
    """
    Minimum return of the strategy within the stock price domain.
    """

    maximum_return_in_the_domain: float
    """
    Maximum return of the strategy within the stock price domain.
    """

    implied_volatility: list[float]
    """
    List of implied volatilities, one per strategy leg.
    """

    in_the_money_probability: list[float]
    """
    List of probabilities of legs expiring in-the-money (ITM).
    """

    probability_of_touch: list[float]
    """
    List of probabilities of legs ever getting in-the-money (ITM) before expiration.
    """

    delta: list[float]
    """
    List of Delta values, one per strategy leg.
    """

    gamma: list[float]
    """
    List of Gamma values, one per strategy leg.
    """

    theta: list[float]
    """
    List of Theta values, one per strategy leg.
    """

    vega: list[float]
    """
    List of Vega values, one per strategy leg.
    """

    rho: list[float]
    """
    List of Rho values, one per strategy leg.
    """

    probability_of_profit_target: float = 0.0
    """
    Probability of the strategy yielding at least the profit target. 
    
    The default is 0.0.
    """

    profit_target_ranges: list[Range] = []
    """
    List of minimum and maximum stock prices defining ranges in which the
    strategy makes at least the profit target. 
    
    The default is [].
    """

    probability_of_loss_limit: float = 0.0
    """
    Probability of the strategy losing at least the loss limit. 
    
    The default is 0.0.
    """

    loss_limit_ranges: list[Range] = []
    """
    List of minimum and maximum stock prices defining ranges where the
    strategy loses at least the loss limit. 
    
    The default is [].
    """

    inputs: Inputs
    """@private"""

    data: EngineDataResults
    """@private"""

    def __str__(self):
        s = ""

        for key, value in self.model_dump(
            exclude={"data", "inputs"},
            exclude_none=True,
            exclude_defaults=True,
        ).items():
            s += f"{key.capitalize().replace('_',' ')}: {value}\n"

        return s


class PoPOutputs(BaseModel):
    """
    Defines the output data from a probability of profit (PoP) calculation.
    """

    probability_of_reaching_target: float = 0.0
    """
    Probability that the strategy return will be equal or greater than the
    target. 
    
    The default is 0.0.
    """

    probability_of_missing_target: float = 0.0
    """
    Probability that the strategy return will be less than the target. 
    
    The default is 0.0.
    """

    reaching_target_range: list[Range] = []
    """
    Range of stock prices where the strategy return is equal or greater than
    the target. 
    
    The default is [].
    """

    missing_target_range: list[Range] = []
    """
    Range of stock prices where the strategy return is less than the target.
    
    The default is [].
    """

    expected_return_above_target: float = 0.0
    """
    Expected value of the strategy return when the return is equal or greater
    than the target. 
    
    The default is 0.0.
    """

    expected_return_below_target: float = 0.0
    """
    Expected value of the strategy return when the return is less than the
    target. 
    
    The default is 0.0.
    """
