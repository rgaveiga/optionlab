from __future__ import division
from __future__ import print_function

import datetime as dt
from typing import Literal, Any

from numpy import array, ndarray, zeros, stack, savetxt

from optionlab.black_scholes import get_bs_info, get_implied_vol
from optionlab.models import (
    Inputs,
    Action,
    StrategyType,
    Range,
    OptionStrategy,
    StockStrategy,
    ClosedPosition,
    Outputs,
    ProbabilityOfProfitInputs,
    ProbabilityOfProfitArrayInputs,
    OptionType,
)
from optionlab.support import (
    get_pl_profile,
    get_pl_profile_stock,
    get_pl_profile_bs,
    get_profit_range,
    create_price_seq,
    create_price_samples,
    get_pop,
)
from optionlab.utils import get_nonbusiness_days


class StrategyEngine:
    def __init__(self, inputs: Inputs):
        """
        __init__ -> initializes class variables.

        Returns
        -------
        None.
        """
        self.s = create_price_seq(inputs.min_stock, inputs.max_stock)
        self._s_mc: ndarray = array(inputs.array_prices or [])
        self.strike: list[float] = []
        self._premium: list[float] = []
        self._n: list[int] = []
        self.action: list[Action | Literal["n/a"]] = []
        self.type: list[StrategyType] = []
        self._expiration: list[dt.date | int] = (
            []
        )  # FIXME: remove? doesn't seem to be used
        self._prev_pos: list[float] = []
        self._use_bs: list[bool] = []
        self._profit_ranges: list[Range] = []
        self._profit_target_range: list[Range] = []
        self._loss_limit_ranges: list[Range] = []
        self._days_to_maturity: list[int] = []
        self._days_in_year = 252 if inputs.discard_nonbusiness_days else 365
        self.days_to_target = 30
        self.implied_volatility: list[float | ndarray] = []
        self.itm_probability: list[float] = []
        self.delta: list[float] = []
        self.gamma: list[float] = []
        self.vega: list[float] = []
        self.theta: list[float] = []
        self.cost: list[float] = []
        self.project_probability = 0.0
        self.project_target_probability = 0.0
        self.loss_limit_probability = 0.0
        self._distribution = inputs.distribution
        self._stock_price = inputs.stock_price
        self._volatility = inputs.volatility
        self._r = inputs.interest_rate
        self._y = inputs.dividend_yield
        self._profit_target = inputs.profit_target
        self._loss_limit = inputs.loss_limit
        self._opt_commission = inputs.opt_commission
        self._stock_commission = inputs.stock_commission
        self._n_mc_prices = inputs.mc_prices_number
        self._compute_expectation = inputs.compute_expectation

        if inputs.start_date and inputs.target_date:
            if inputs.discard_nonbusiness_days:
                n_discarded_days = get_nonbusiness_days(
                    inputs.start_date, inputs.target_date, inputs.country
                )
            else:
                n_discarded_days = 0

            self.days_to_target = (
                inputs.target_date - inputs.start_date
            ).days - n_discarded_days
        else:
            self.days_to_target = inputs.days_to_target_date

        for i, strategy in enumerate(inputs.strategy):
            self.type.append(strategy.type)

            if isinstance(strategy, OptionStrategy):
                self.strike.append(strategy.strike)
                self._premium.append(strategy.premium)
                self._n.append(strategy.n)
                self.action.append(strategy.action)
                self._prev_pos.append(strategy.prev_pos or 0.0)

                if not strategy.expiration:
                    if inputs.target_date:
                        self._expiration.append(inputs.target_date)

                    self._days_to_maturity.append(self.days_to_target)
                    self._use_bs.append(False)
                elif isinstance(strategy.expiration, dt.date) and inputs.start_date:
                    self._expiration.append(strategy.expiration)

                    if inputs.discard_nonbusiness_days:
                        n_discarded_days = get_nonbusiness_days(
                            inputs.start_date, strategy.expiration, inputs.country
                        )
                    else:
                        n_discarded_days = 0

                    self._days_to_maturity.append(
                        (strategy.expiration - inputs.start_date).days
                        - n_discarded_days
                    )

                    self._use_bs.append(strategy.expiration != inputs.target_date)
                elif isinstance(strategy.expiration, int):
                    if strategy.expiration >= self.days_to_target:
                        self._days_to_maturity.append(strategy.expiration)

                        self._use_bs.append(strategy.expiration != self.days_to_target)
                    else:
                        raise ValueError(
                            "Days remaining to maturity must be greater than or equal to the number of days remaining to the target date!"
                        )
                else:
                    raise ValueError("Expiration must be a date, an int or None.")

            elif isinstance(strategy, StockStrategy):
                self._n.append(strategy.n)
                self.action.append(strategy.action)
                self._prev_pos.append(strategy.prev_pos or 0.0)
                self.strike.append(0.0)
                self._premium.append(0.0)
                self._use_bs.append(False)
                self._days_to_maturity.append(-1)
                self._expiration.append(
                    inputs.target_date if isinstance(inputs.target_date, int) else -1
                )

            elif isinstance(strategy, ClosedPosition):
                self._prev_pos.append(strategy.prev_pos)
                self.strike.append(0.0)
                self._n.append(0)
                self._premium.append(0.0)
                self.action.append("n/a")
                self._use_bs.append(False)
                self._days_to_maturity.append(-1)
                self._expiration.append(
                    inputs.target_date if isinstance(inputs.target_date, int) else -1
                )
            else:
                raise ValueError("Type must be 'call', 'put', 'stock' or 'closed'!")

    def run(self) -> Outputs:
        """
        run -> runs calculations for an options strategy.

        Returns
        -------
        output : Outputs
            An Outputs object containing the output of a calculation.
        """

        time_to_target = self.days_to_target / self._days_in_year
        self.cost = [0.0] * len(self.type)
        self.implied_volatility = []
        self.itm_probability = []
        self.delta = []
        self.gamma = []
        self.vega = []
        self.theta = []

        self.profit = zeros((len(self.type), self.s.shape[0]))
        self.strategy_profit = zeros(self.s.shape[0])

        if self._compute_expectation and self._s_mc.shape[0] == 0:
            self._s_mc = create_price_samples(
                self._stock_price,
                self._volatility,
                time_to_target,
                self._r,
                self._distribution,
                self._y,
                self._n_mc_prices,
            )

        if self._s_mc.shape[0] > 0:
            self.profit_mc = zeros((len(self.type), self._s_mc.shape[0]))
            self.strategy_profit_mc = zeros(self._s_mc.shape[0])

        for i, type in enumerate(self.type):
            if type in ("call", "put"):
                self._run_option_calcs(i)
            elif type == "stock":
                self._run_stock_calcs(i)
            elif type == "closed":
                self._run_closed_position_calcs(i)

            self.strategy_profit += self.profit[i]

            if self._compute_expectation or self._distribution == "array":
                self.strategy_profit_mc += self.profit_mc[i]

        self._profit_ranges = get_profit_range(self.s, self.strategy_profit)

        pop_inputs: ProbabilityOfProfitInputs | ProbabilityOfProfitArrayInputs
        if self._distribution in ("normal", "laplace", "black-scholes"):
            pop_inputs = ProbabilityOfProfitInputs(
                source=self._distribution,  # type: ignore
                stock_price=self._stock_price,
                volatility=self._volatility,
                years_to_maturity=time_to_target,
                interest_rate=self._r,
                dividend_yield=self._y,
            )
        elif self._distribution == "array":
            pop_inputs = ProbabilityOfProfitArrayInputs(array=self._s_mc)
        else:
            raise ValueError("Source not supported yet!")

        self.project_probability = get_pop(self._profit_ranges, pop_inputs)

        if self._profit_target is not None:
            self._profit_target_range = get_profit_range(
                self.s, self.strategy_profit, self._profit_target
            )
            self.project_target_probability = get_pop(
                self._profit_target_range, pop_inputs
            )

        if self._loss_limit is not None:
            self._loss_limit_rangesm = get_profit_range(
                self.s, self.strategy_profit, self._loss_limit + 0.01
            )
            self.loss_limit_probability = 1.0 - get_pop(
                self._loss_limit_ranges, pop_inputs
            )

        return self._generate_outputs()

    def get_pl(self, leg: int | None = None) -> tuple[ndarray, ndarray]:
        """
        get_pl -> returns the profit/loss profile of either a leg or the whole
        strategy.

        Parameters
        ----------
        leg : int, optional
            Index of the leg. Default is None (whole strategy).

        Returns
        -------
        stock prices : numpy array
            Sequence of stock prices within the bounds of the stock price domain.
        P/L profile : numpy array
            Profit/loss profile of either a leg or the whole strategy.
        """
        if self.profit.size > 0 and leg and leg < self.profit.shape[0]:
            return self.s, self.profit[leg]

        return self.s, self.strategy_profit

    def pl_to_csv(self, filename: str = "pl.csv", leg: int | None = None) -> None:
        """
        pl_to_csv -> saves the profit/loss data to a .csv file.

        Parameters
        ----------
        filename : string, optional
            Name of the .csv file. Default is 'pl.csv'.
        leg : int, optional
            Index of the leg. Default is None (whole strategy).

        Returns
        -------
        None.
        """
        if self.profit.size > 0 and leg and leg < self.profit.shape[0]:
            arr = stack((self.s, self.profit[leg]))
        else:
            arr = stack((self.s, self.strategy_profit))

        savetxt(
            filename, arr.transpose(), delimiter=",", header="StockPrice,Profit/Loss"
        )

    def _run_option_calcs(self, i: int):
        action: Action = self.action[i]  # type: ignore
        type: OptionType = self.type[i]  # type: ignore

        if self._prev_pos[i] < 0.0:
            # Previous position is closed
            self.implied_volatility.append(0.0)
            self.itm_probability.append(0.0)
            self.delta.append(0.0)
            self.gamma.append(0.0)
            self.vega.append(0.0)
            self.theta.append(0.0)

            cost = (self._premium[i] + self._prev_pos[i]) * self._n[i]

            if self.action[i] == "buy":
                cost *= -1.0

            self.cost[i] = cost
            self.profit[i] += cost

            if self._compute_expectation or self._distribution == "array":
                self.profit_mc[i] += cost

            return

        time_to_maturity = self._days_to_maturity[i] / self._days_in_year
        bs = get_bs_info(
            self._stock_price,
            self.strike[i],
            self._r,
            self._volatility,
            time_to_maturity,
            self._y,
        )

        self.gamma.append(bs.gamma)
        self.vega.append(bs.vega)

        self.implied_volatility.append(
            get_implied_vol(
                type,
                self._premium[i],
                self._stock_price,
                self.strike[i],
                self._r,
                time_to_maturity,
                self._y,
            )
        )

        negative_multiplier = 1 if self.action[i] == "buy" else -1

        if type == "call":
            self.itm_probability.append(bs.call_itm_prob)
            self.delta.append(bs.call_delta * negative_multiplier)
            self.theta.append(bs.call_theta / self._days_in_year * negative_multiplier)
        else:
            self.itm_probability.append(bs.put_itm_prob)
            self.delta.append(bs.put_delta * negative_multiplier)
            self.theta.append(bs.put_theta / self._days_in_year * negative_multiplier)

        if self._prev_pos[i] > 0.0:  # Premium of the open position
            opt_value = self._prev_pos[i]
        else:  # Current premium
            opt_value = self._premium[i]

        if self._use_bs[i]:
            target_to_maturity = (
                self._days_to_maturity[i] - self.days_to_target
            ) / self._days_in_year

            self.profit[i], self.cost[i] = get_pl_profile_bs(
                type,
                action,
                self.strike[i],
                opt_value,
                self._r,
                target_to_maturity,
                self._volatility,
                self._n[i],
                self.s,
                self._y,
                self._opt_commission,
            )

            if self._compute_expectation or self._distribution == "array":
                self.profit_mc[i] = get_pl_profile_bs(
                    type,
                    action,
                    self.strike[i],
                    opt_value,
                    self._r,
                    target_to_maturity,
                    self._volatility,
                    self._n[i],
                    self._s_mc,
                    self._y,
                    self._opt_commission,
                )[0]
        else:
            self.profit[i], self.cost[i] = get_pl_profile(
                type,
                action,
                self.strike[i],
                opt_value,
                self._n[i],
                self.s,
                self._opt_commission,
            )

            if self._compute_expectation or self._distribution == "array":
                self.profit_mc[i] = get_pl_profile(
                    type,
                    action,
                    self.strike[i],
                    opt_value,
                    self._n[i],
                    self._s_mc,
                    self._opt_commission,
                )[0]

    def _run_stock_calcs(self, i: int):
        action: Action = self.action[i]  # type: ignore

        self.implied_volatility.append(0.0)
        self.itm_probability.append(1.0)
        self.delta.append(1.0)
        self.gamma.append(0.0)
        self.vega.append(0.0)
        self.theta.append(0.0)

        if self._prev_pos[i] < 0.0:  # Previous position is closed
            costtmp = (self._stock_price + self._prev_pos[i]) * self._n[i]

            if self.action[i] == "buy":
                costtmp *= -1.0

            self.cost[i] = costtmp
            self.profit[i] += costtmp

            if self._compute_expectation or self._distribution == "array":
                self.profit_mc[i] += costtmp

            return

        if self._prev_pos[i] > 0.0:  # Stock price at previous position
            stockpos = self._prev_pos[i]
        else:  # Spot price of the stock at start date
            stockpos = self._stock_price

        self.profit[i], self.cost[i] = get_pl_profile_stock(
            stockpos,
            action,
            self._n[i],
            self.s,
            self._stock_commission,
        )

        if self._compute_expectation or self._distribution == "array":
            self.profit_mc[i] = get_pl_profile_stock(
                stockpos,
                action,
                self._n[i],
                self._s_mc,
                self._stock_commission,
            )[0]

    def _run_closed_position_calcs(self, i: int):
        self.implied_volatility.append(0.0)
        self.itm_probability.append(0.0)
        self.delta.append(0.0)
        self.gamma.append(0.0)
        self.vega.append(0.0)
        self.theta.append(0.0)

        self.cost[i] = self._prev_pos[i]
        self.profit[i] += self._prev_pos[i]

        if self._compute_expectation or self._distribution == "array":
            self.profit_mc[i] += self._prev_pos[i]

    def _generate_outputs(self) -> Outputs:
        optional_outputs: dict[str, Any] = {}

        if self._profit_target is not None:
            optional_outputs["probability_of_profit_target"] = (
                self.project_target_probability
            )
            optional_outputs["project_target_ranges"] = self._profit_target_range

        if self._loss_limit is not None:
            optional_outputs["probability_of_loss_limit"] = self.loss_limit_probability

        if (
            self._compute_expectation or self._distribution == "array"
        ) and self._s_mc.shape[0] > 0:
            profit = self.strategy_profit_mc[self.strategy_profit_mc >= 0.01]
            loss = self.strategy_profit_mc[self.strategy_profit_mc < 0.0]
            optional_outputs["average_profit_from_mc"] = 0.0
            optional_outputs["average_loss_from_mc"] = (
                loss.mean() if loss.shape[0] > 0 else 0.0
            )

            if profit.shape[0] > 0:
                optional_outputs["average_profit_from_mc"] = profit.mean()

            if loss.shape[0] > 0:
                optional_outputs["average_loss_from_mc"] = loss.mean()

            optional_outputs["probability_of_profit_from_mc"] = (
                self.strategy_profit_mc >= 0.01
            ).sum() / self.strategy_profit_mc.shape[0]

        return Outputs.model_validate(
            optional_outputs
            | {
                "probability_of_profit": self.project_probability,
                "strategy_cost": sum(self.cost),
                "per_leg_cost": self.cost,
                "profit_ranges": self._profit_ranges,
                "minimum_return_in_the_domain": self.strategy_profit.min(),
                "maximum_return_in_the_domain": self.strategy_profit.max(),
                "implied_volatility": self.implied_volatility,
                "in_the_money_probability": self.itm_probability,
                "delta": self.delta,
                "gamma": self.gamma,
                "theta": self.theta,
                "vega": self.vega,
            }
        )

    """
    Properties
    ----------
    stock_price_array : array
        A Numpy array of consecutive stock prices, from the minimum price up to 
        the maximum price in the stock price domain. It is used to compute the 
        strategy's P/L profile.
    terminal_stock_prices : array
        A Numpy array or terminal stock prices typically generated by Monte Carlo 
        simulations. It is used to compute strategy's expected profit and loss. 
    """

    @property
    def stock_price_array(self):
        return self.s

    @property
    def terminal_stock_prices(
        self,
    ):  # FIXME: why are Monte Carlo stock prices named `terminal_stock_prices` here?
        return self._s_mc

    @terminal_stock_prices.setter
    def terminal_stock_prices(self, s: ndarray):  # TODO: likely delete this
        if isinstance(s, ndarray):
            if s.shape[0] > 0:
                self._s_mc = s
            else:
                raise ValueError("Empty terminal stock price array is not allowed!")
        else:
            raise TypeError("A numpy array is expected!")
