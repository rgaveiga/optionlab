import datetime as dt
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

OptionType = Literal["call", "put"]
Action = Literal["buy", "sell"]
StrategyType = Literal["stock"] | OptionType | Literal["closed"]
Range = tuple[float, float]
TheoreticalModel = Literal["black-scholes", "normal", "array"]


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
        It must be **stock**.
    n : int
        Number of shares.
    action : str
        `Action` literal value, which must be either **buy** or **sell**.
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
        `OptionType` literal value, which must be either **call** or **put**.
    strike : float
        Strike price.
    premium : float
        Option premium.
    n : int
        Number of options.
    action : str
        `Action` literal value, which must be either **buy** or **sell**.
    prev_pos : float | None, optional
        Premium effectively paid or received in a previously opened position. If
        positive, the position remains open and the payoff calculation considers
        this price instead of the current price of the option. If negative, the
        position is closed and the difference between this price and the current
        price is included in the payoff calculation. The default is None, which
        means this option position is not a previously opened position.
    expiration : str | int | None, optional
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
        It must be **closed**.
    prev_pos : float
        The total amount of the closed position. If positive, it resulted in a
        profit; if negative, it incurred a loss.
    """

    type: Literal["closed"] = "closed"
    prev_pos: float


StrategyLeg = Stock | Option | ClosedPosition


class TheoreticalModelInputs(BaseModel):
    stock_price: float = Field(gt=0.0)
    volatility: float = Field(gt=0.0)
    years_to_target_date: float = Field(ge=0.0)


class BlackScholesModelInputs(TheoreticalModelInputs):
    """
    Defines the input data for the calculations using the Black-Scholes model.

    Attributes
    ----------
    model : str
        It must be either **black-scholes** or **normal**.
    stock_price : float
        Stock price.
    volatility : float
        Annualized volatility of the underlying asset.
    years_to_target_date : float
        Time remaining until target date, in years.
    interest_rate : float
        Annualized risk-free interest rate. The default is zero.
    dividend_yield : float, optional
        Annualized dividend yield. The default is zero.
    """

    model: Literal["black-scholes", "normal"] = "black-scholes"
    interest_rate: float = Field(0.0, ge=0.0)
    dividend_yield: float = Field(0.0, ge=0.0, le=1.0)

    __hash__ = object.__hash__


class LaplaceInputs(TheoreticalModelInputs):
    """
    Defines the input data for the calculations using a log-Laplace distribution of
    stock prices.

    Attributes
    ----------
    model : str
        It must be **laplace**.
    stock_price : float
        Stock price.
    mu : float
        Annualized return of the underlying asset.
    volatility : float
        Annualized volatility of the underlying asset.
    years_to_target_date : float
        Time remaining until target date, in years.
    """

    model: Literal["laplace"] = "laplace"
    mu: float

    __hash__ = object.__hash__


class ArrayInputs(BaseModel):
    """
    Defines the input data for the calculations when using an array of strategy
    returns.

    Attributes
    ----------
    model : str
        It must be **array**.
    array : numpy.ndarray
        Array of strategy returns.
    """

    model: Literal["array"] = "array"
    array: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("array", mode="before")
    @classmethod
    def validate_arrays(cls, v: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(v)
        if arr.shape[0] == 0:
            raise ValueError("The array is empty!")
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
    profit_target : float | None, optional
        Target profit level. The default is None, which means it is not
        calculated.
    loss_limit : float | None, optional
        Limit loss level. The default is None, which means it is not calculated.
    opt_commission : float
        Brokerage commission for options transactions. The default is 0.0.
    stock_commission : float
        Brokerage commission for stocks transactions. The default is 0.0.
    discard_nonbusiness_days : bool, optional
        Discards weekends and holidays when counting the number of days between
        two dates. The default is True.
    business_days_in_year : int, optional
        Number of business days in a year. The default is 252.
    country : str, optional
        Country whose holidays will be counted if `discard_nonbusinessdays` is
        set to True. The default is **US**.
    start_date : datetime.date, optional
        Start date in the calculations. If not provided, `days_to_target_date`
        must be provided.
    target_date : datetime.date, optional
        Target date in the calculations. If not provided, `days_to_target_date`
        must be provided.
    days_to_target_date : int, optional
        Days remaining to the target date. If not provided, `start_date` and
        `target_date` must be provided.
    model : str, optional
        Theoretical model used in the calculations of probability of profit. It
        can be **black-scholes** (the same as **normal**) or **array**. The default
        is **black-scholes**.
    array : numpy.ndarray | None, optional
        Array of terminal stock prices. The default is None.
    """

    stock_price: float = Field(gt=0.0)
    volatility: float = Field(ge=0.0)
    interest_rate: float = Field(0.0, ge=0.0)
    min_stock: float = Field(ge=0.0)
    max_stock: float = Field(ge=0.0)
    strategy: list[StrategyLeg] = Field(..., min_length=1)
    dividend_yield: float = Field(0.0, ge=0.0)
    profit_target: float | None = None
    loss_limit: float | None = None
    opt_commission: float = 0.0
    stock_commission: float = 0.0
    discard_nonbusiness_days: bool = True
    business_days_in_year: int = 252
    country: str = "US"
    start_date: dt.date | None = None
    target_date: dt.date | None = None
    days_to_target_date: int = Field(0, ge=0)
    model: TheoreticalModel = "black-scholes"
    array: np.ndarray | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    def validate_model_array(self) -> "Inputs":
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
    """
    Defines the data returned by a calculation using the Black-Scholes model.

    Attributes
    ----------
    call_price : float | numpy.ndarray
        Price of a call option.
    put_price : float | numpy.ndarray
        Price of a put option.
    call_delta : float | numpy.ndarray
        Delta of a call option.
    put_delta : float | numpy.ndarray
        Delta of a put option.
    gamma : float | numpy.ndarray
        Gamma of an option.
    vega : float | numpy.ndarray
        Vega of an option.
    call_rho : float | numpy.ndarray
        Rho of a call option.
    put_rho : float | numpy.ndarray
        Rho of a put option.
    call_itm_prob : float | numpy.ndarray
        In-the-money probability of a call option.
    put_itm_prob : float | numpy.ndarray
        In-the-money probability of a put option.
    """

    call_price: float | np.ndarray
    put_price: float | np.ndarray
    call_delta: float | np.ndarray
    put_delta: float | np.ndarray
    call_theta: float | np.ndarray
    put_theta: float | np.ndarray
    gamma: float | np.ndarray
    vega: float | np.ndarray
    call_rho: float | np.ndarray
    put_rho: float | np.ndarray
    call_itm_prob: float | np.ndarray
    put_itm_prob: float | np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)


def init_empty_array() -> np.ndarray:
    return np.array([])


class EngineDataResults(BaseModel):
    stock_price_array: np.ndarray
    terminal_stock_prices: np.ndarray | None
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
    previous_position: list[float] | None = []
    use_bs: list[bool] | None = []
    profit_ranges: list[Range] | None = []
    profit_target_ranges: list[Range] | None = None
    loss_limit_ranges: list[Range] | None = None
    days_to_maturity: list[int] = []
    days_in_year: int = 365
    days_to_target: int = 30
    implied_volatility: list[float | np.ndarray] = []
    itm_probability: list[float] = []
    delta: list[float] = []
    gamma: list[float] = []
    vega: list[float] = []
    rho: list[float] = []
    theta: list[float] = []
    cost: list[float] = []
    profit_probability: float = 0.0
    profit_target_probability: float | None = None
    loss_limit_probability: float | None = None
    expected_profit: float = 0.0
    expected_loss: float = 0.0


class Outputs(BaseModel):
    """
    Defines the output data from a strategy calculation.

    Attributes
    ----------
    probability_of_profit : float
        Probability of the strategy yielding at least $0.01.
    profit_ranges : list[Range]
        A list of minimum and maximum stock prices defining ranges in which the
        strategy makes at least $0.01.
    strategy_cost : float
        Total strategy cost.
    per_leg_cost : list[float]
        A list of costs, one per strategy leg.
    implied_volatility : list[float]
        List of implied volatilities, one per strategy leg.
    in_the_money_probability : list[float]
        List of ITM probabilities, one per strategy leg.
    delta : list[float]
        List of Delta values, one per strategy leg.
    gamma : list[float]
        List of Gamma values, one per strategy leg.
    theta : list[float]
        List of Theta values, one per strategy leg.
    vega : list[float]
        List of Vega values, one per strategy leg.
    rho : list[float]
        List of Rho values, one per strategy leg.
    minimum_return_in_the_domain : float
        Minimum return of the strategy within the stock price domain.
    maximum_return_in_the_domain : float
        Maximum return of the strategy within the stock price domain.
    probability_of_profit_target : float | None, optional
        Probability of the strategy yielding at least the profit target. The
        default is None.
    profit_target_ranges : list[Range] | None, optional
        List of minimum and maximum stock prices defining ranges in which the
        strategy makes at least the profit target. The default is None.
    probability_of_loss_limit : float | None, optional
        Probability of the strategy losing at least the loss limit. The default
        is None.
    loss_limit_ranges : list[Range] | None, optional
        List of minimum and maximum stock prices defining ranges where the
        strategy loses at least the loss limit. The default is None.
    expected_profit : float | None, optional
        Expected profit when the strategy is profitable. The default is None.
    expected_loss : float | None, optional
        Expected loss when the strategy is not profitable. The default is None.
    data : EngineDataResults
        Further data from the strategy calculation that can be used in the
        post-processing of the outputs.
    inputs : Inputs
        Input data used in the strategy calculation.
    """

    inputs: Inputs
    data: EngineDataResults
    probability_of_profit: float
    profit_ranges: list[Range] | None
    expected_profit: float | None = None
    expected_loss: float | None = None
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
    rho: list[float]
    probability_of_profit_target: float | None = None
    profit_target_ranges: list[Range] | None = None
    probability_of_loss_limit: float | None = None
    loss_limit_ranges: list[Range] | None = None

    def __str__(self):
        s = ""

        for key, value in self.dict(
            exclude={"data", "inputs"}, exclude_none=True
        ).items():
            s += f"{key.capitalize().replace('_',' ')}: {value}\n"

        return s


class PoPOutputs(BaseModel):
    """
    Defines the output data from a probability of profit (PoP) calculation.

    Attributes
    ----------
    probability_of_reaching_target : float | None, optional
        Probability that the strategy return will be equal or greater than the
        target. The default is None.
    probability_of_missing_target : float | None, optional
        Probability that the strategy return will be less than the target. The
        default is None.
    reaching_target_range : list[Range] | None, optional
        Range of stock prices where the strategy return is equal or greater than
        the target. The default is None.
    missing_target_range : list[Range] | None, optional
        Range of stock prices where the strategy return is less than the target.
        The default is None.
    expected_return_above_target : float | None, optional
        Expected value of the strategy return when the return is equal or greater
        than the target. The default is None.
    expected_return_below_target : float | None, optional
        Expected value of the strategy return when the return is less than the
        target. The default is None.
    """

    probability_of_reaching_target: float | None = None
    probability_of_missing_target: float | None = None
    reaching_target_range: list[Range] | None = None
    missing_target_range: list[Range] | None = None
    expected_return_above_target: float | None = None
    expected_return_below_target: float | None = None
