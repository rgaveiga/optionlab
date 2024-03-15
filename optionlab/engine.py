from __future__ import division
from __future__ import print_function

import datetime as dt
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from numpy import array, ndarray, zeros, full, stack, savetxt

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
        self.__s = array([])
        self.__s_mc: np.ndarray = array(inputs.array_prices or [])
        self.__strike: list[float] = []
        self.__premium: list[float] = []
        self.__n: list[int] = []
        self.__action: list[Action | Literal["n/a"]] = []
        self.__type: list[StrategyType] = []
        self.__expiration: list[dt.date | int] = []
        self.__prevpos: list[float] = []
        self.__usebs: list[bool] = []
        self.__profitranges: list[Range] = []
        self.__profittargrange: list[Range] = []
        self.__losslimitranges: list[Range] = []
        self.__days2maturity: list[int] = []
        self.__days2target = 30
        self.__daysinyear = 252 if inputs.discard_nonbusiness_days else 365
        self.implied_volatility: list[float] = []
        self.itm_probability: list[float] = []
        self.delta: list[float] = []
        self.gamma: list[float] = []
        self.vega: list[float] = []
        self.theta: list[float] = []
        self.cost: list[float] = []
        self.project_probability = 0.0
        self.project_target_probability = 0.0
        self.loss_limit_probability = 0.0
        self.__distribution = inputs.distribution
        self.__stockprice = inputs.stock_price
        self.__volatility = inputs.volatility
        self.__r = inputs.interest_rate
        self.__y = inputs.dividend_yield
        self.__minstock = inputs.min_stock
        self.__maxstock = inputs.max_stock
        self.__profittarg = inputs.profit_target
        self.__losslimit = inputs.loss_limit
        self.__optcommission = inputs.opt_commission
        self.__stockcommission = inputs.stock_commission
        self.__nmcprices = inputs.mc_prices_number
        self.__compute_expectation = inputs.compute_expectation

        if inputs.start_date and inputs.target_date:
            if inputs.discard_nonbusiness_days:
                n_discarded_days = get_nonbusiness_days(
                    inputs.start_date, inputs.target_date, inputs.country
                )
            else:
                n_discarded_days = 0

            self.__days2target = (
                inputs.target_date - inputs.start_date
            ).days - n_discarded_days
        else:
            self.__days2target = inputs.days_to_target_date

        for i, strategy in enumerate(inputs.strategy):
            self.__type.append(strategy.type)

            if isinstance(strategy, OptionStrategy):
                self.__strike.append(strategy.strike)
                self.__premium.append(strategy.premium)
                self.__n.append(strategy.n)
                self.__action.append(strategy.action)
                self.__prevpos.append(strategy.prev_pos or 0.0)

                if not strategy.expiration:
                    if inputs.target_date:
                        self.__expiration.append(inputs.target_date)

                    self.__days2maturity.append(self.__days2target)
                    self.__usebs.append(False)
                elif isinstance(strategy.expiration, dt.date) and inputs.start_date:
                    self.__expiration.append(strategy.expiration)

                    if inputs.discard_nonbusiness_days:
                        n_discarded_days = get_nonbusiness_days(
                            inputs.start_date, strategy.expiration, inputs.country
                        )
                    else:
                        n_discarded_days = 0

                    self.__days2maturity.append(
                        (strategy.expiration - inputs.start_date).days
                        - n_discarded_days
                    )

                    self.__usebs.append(strategy.expiration != inputs.target_date)
                elif isinstance(strategy.expiration, int):
                    if strategy.expiration >= self.__days2target:
                        self.__days2maturity.append(strategy.expiration)

                        self.__usebs.append(strategy.expiration != self.__days2target)
                    else:
                        raise ValueError(
                            "Days remaining to maturity must be greater than or equal to the number of days remaining to the target date!"
                        )
                else:
                    raise ValueError("Expiration must be a date, an int or None.")

            elif isinstance(strategy, StockStrategy):
                self.__n.append(strategy.n)
                self.__action.append(strategy.action)
                self.__prevpos.append(strategy.prev_pos or 0.0)
                self.__strike.append(0.0)
                self.__premium.append(0.0)
                self.__usebs.append(False)
                self.__days2maturity.append(-1)
                self.__expiration.append(
                    inputs.target_date if isinstance(inputs.target_date, int) else -1
                )

            elif isinstance(strategy, ClosedPosition):
                self.__prevpos.append(strategy.prev_pos)
                self.__strike.append(0.0)
                self.__n.append(0)
                self.__premium.append(0.0)
                self.__action.append("n/a")
                self.__usebs.append(False)
                self.__days2maturity.append(-1)
                self.__expiration.append(
                    inputs.target_date if isinstance(inputs.target_date, int) else -1
                )
            else:
                raise ValueError("Type must be 'call', 'put', 'stock' or 'closed'!")

    def run(self):
        """
        run -> runs calculations for an options strategy.

        Returns
        -------
        output : Outputs
            An Outputs object containing the output of a calculation.
        """

        time2target = self.__days2target / self.__daysinyear
        self.cost = [0.0 for _ in range(len(self.__type))]
        self.implied_volatility = []
        self.itm_probability = []
        self.delta = []
        self.gamma = []
        self.vega = []
        self.theta = []

        if self.__s.shape[0] == 0:
            self.__s = create_price_seq(self.__minstock, self.__maxstock)

        self.profit = zeros((len(self.__type), self.__s.shape[0]))
        self.strategyprofit = zeros(self.__s.shape[0])

        if self.__compute_expectation and self.__s_mc.shape[0] == 0:
            self.__s_mc = create_price_samples(
                self.__stockprice,
                self.__volatility,
                time2target,
                self.__r,
                self.__distribution,
                self.__y,
                self.__nmcprices,
            )

        if self.__s_mc.shape[0] > 0:
            self.profit_mc = zeros((len(self.__type), self.__s_mc.shape[0]))
            self.strategyprofit_mc = zeros(self.__s_mc.shape[0])

        for i, type in enumerate(self.__type):
            if type in ("call", "put"):
                if self.__prevpos[i] >= 0.0:
                    time_to_maturity = self.__days2maturity[i] / self.__daysinyear
                    bs = get_bs_info(
                        self.__stockprice,
                        self.__strike[i],
                        self.__r,
                        self.__volatility,
                        time_to_maturity,
                        self.__y,
                    )

                    self.gamma.append(bs.gamma)
                    self.vega.append(bs.vega)

                    if type == "call":
                        self.implied_volatility.append(
                            get_implied_vol(
                                "call",
                                self.__premium[i],
                                self.__stockprice,
                                self.__strike[i],
                                self.__r,
                                time_to_maturity,
                                self.__y,
                            )
                        )
                        self.itm_probability.append(bs.call_itm_prob)

                        if self.__action[i] == "buy":
                            self.delta.append(bs.call_delta)
                            self.theta.append(bs.call_theta / self.__daysinyear)
                        else:
                            self.delta.append(-bs.call_delta)
                            self.theta.append(-bs.call_theta / self.__daysinyear)
                    else:
                        self.implied_volatility.append(
                            get_implied_vol(
                                "put",
                                self.__premium[i],
                                self.__stockprice,
                                self.__strike[i],
                                self.__r,
                                time_to_maturity,
                                self.__y,
                            )
                        )
                        self.itm_probability.append(bs.put_itm_prob)

                        if self.__action[i] == "buy":
                            self.delta.append(bs.put_delta)
                            self.theta.append(bs.put_theta / self.__daysinyear)
                        else:
                            self.delta.append(-bs.put_delta)
                            self.theta.append(-bs.put_theta / self.__daysinyear)
                else:
                    self.implied_volatility.append(0.0)
                    self.itm_probability.append(0.0)
                    self.delta.append(0.0)
                    self.gamma.append(0.0)
                    self.vega.append(0.0)
                    self.theta.append(0.0)

                if self.__prevpos[i] < 0.0:  # Previous position is closed
                    costtmp = (self.__premium[i] + self.__prevpos[i]) * self.__n[i]

                    if self.__action[i] == "buy":
                        costtmp *= -1.0

                    self.cost[i] = costtmp
                    self.profit[i] += costtmp

                    if self.__compute_expectation or self.__distribution == "array":
                        self.profit_mc[i] += costtmp
                else:
                    if self.__prevpos[i] > 0.0:  # Premium of the open position
                        opval = self.__prevpos[i]
                    else:  # Current premium
                        opval = self.__premium[i]

                    if self.__usebs[i]:
                        self.profit[i], self.cost[i] = get_pl_profile_bs(
                            type,
                            self.__action[i],
                            self.__strike[i],
                            opval,
                            self.__r,
                            (self.__days2maturity[i] - self.__days2target)
                            / self.__daysinyear,
                            self.__volatility,
                            self.__n[i],
                            self.__s,
                            self.__y,
                            self.__optcommission,
                        )

                        if self.__compute_expectation or self.__distribution == "array":
                            self.profit_mc[i] = get_pl_profile_bs(
                                type,
                                self.__action[i],
                                self.__strike[i],
                                opval,
                                self.__r,
                                (self.__days2maturity[i] - self.__days2target)
                                / self.__daysinyear,
                                self.__volatility,
                                self.__n[i],
                                self.__s_mc,
                                self.__y,
                                self.__optcommission,
                            )[0]
                    else:
                        self.profit[i], self.cost[i] = get_pl_profile(
                            type,
                            self.__action[i],
                            self.__strike[i],
                            opval,
                            self.__n[i],
                            self.__s,
                            self.__optcommission,
                        )

                        if self.__compute_expectation or self.__distribution == "array":
                            self.profit_mc[i] = get_pl_profile(
                                type,
                                self.__action[i],
                                self.__strike[i],
                                opval,
                                self.__n[i],
                                self.__s_mc,
                                self.__optcommission,
                            )[0]
            elif type == "stock":
                self.implied_volatility.append(0.0)
                self.itm_probability.append(1.0)
                self.delta.append(1.0)
                self.gamma.append(0.0)
                self.vega.append(0.0)
                self.theta.append(0.0)

                if self.__prevpos[i] < 0.0:  # Previous position is closed
                    costtmp = (self.__stockprice + self.__prevpos[i]) * self.__n[i]

                    if self.__action[i] == "buy":
                        costtmp *= -1.0

                    self.cost[i] = costtmp
                    self.profit[i] += costtmp

                    if self.__compute_expectation or self.__distribution == "array":
                        self.profit_mc[i] += costtmp
                else:
                    if self.__prevpos[i] > 0.0:  # Stock price at previous position
                        stockpos = self.__prevpos[i]
                    else:  # Spot price of the stock at start date
                        stockpos = self.__stockprice

                    self.profit[i], self.cost[i] = get_pl_profile_stock(
                        stockpos,
                        self.__action[i],
                        self.__n[i],
                        self.__s,
                        self.__stockcommission,
                    )

                    if self.__compute_expectation or self.__distribution == "array":
                        self.profit_mc[i] = get_pl_profile_stock(
                            stockpos,
                            self.__action[i],
                            self.__n[i],
                            self.__s_mc,
                            self.__stockcommission,
                        )[0]
            elif type == "closed":
                self.implied_volatility.append(0.0)
                self.itm_probability.append(0.0)
                self.delta.append(0.0)
                self.gamma.append(0.0)
                self.vega.append(0.0)
                self.theta.append(0.0)

                self.cost[i] = self.__prevpos[i]
                self.profit[i] += self.__prevpos[i]

                if self.__compute_expectation or self.__distribution == "array":
                    self.profit_mc[i] += self.__prevpos[i]

            self.strategyprofit += self.profit[i]

            if self.__compute_expectation or self.__distribution == "array":
                self.strategyprofit_mc += self.profit_mc[i]

        self.__profitranges = get_profit_range(self.__s, self.strategyprofit)

        if self.__distribution in ("normal", "laplace", "black-scholes"):
            pop_inputs = ProbabilityOfProfitInputs(
                source=self.__distribution,
                stock_price=self.__stockprice,
                volatility=self.__volatility,
                years_to_maturity=time2target,
                interest_rate=self.__r,
                dividend_yield=self.__y,
            )
        elif self.__distribution == "array":
            pop_inputs = ProbabilityOfProfitArrayInputs(array=self.__s_mc)
        else:
            raise ValueError("Source not supported yet!")

        self.project_probability = get_pop(self.__profitranges, pop_inputs)

        if self.__profittarg is not None:
            self.__profittargrange = get_profit_range(
                self.__s, self.strategyprofit, self.__profittarg
            )
            self.project_target_probability = get_pop(
                self.__profittargrange, pop_inputs
            )

        if self.__losslimit is not None:
            self.__losslimitranges = get_profit_range(
                self.__s, self.strategyprofit, self.__losslimit + 0.01
            )
            self.loss_limit_probability = 1.0 - get_pop(
                self.__losslimitranges, pop_inputs
            )

        optional_outputs = {}

        if self.__profittarg is not None:
            optional_outputs["probability_of_profit_target"] = (
                self.project_target_probability
            )
            optional_outputs["project_target_ranges"] = self.__profittargrange

        if self.__losslimit is not None:
            optional_outputs["probability_of_loss_limit"] = self.loss_limit_probability

        if (
            self.__compute_expectation or self.__distribution == "array"
        ) and self.__s_mc.shape[0] > 0:
            tmpprof = self.strategyprofit_mc[self.strategyprofit_mc >= 0.01]
            tmploss = self.strategyprofit_mc[self.strategyprofit_mc < 0.0]
            optional_outputs["average_profit_from_mc"] = 0.0
            optional_outputs["average_loss_from_mc"] = (
                tmploss.mean() if tmploss.shape[0] > 0 else 0.0
            )

            if tmpprof.shape[0] > 0:
                optional_outputs["average_profit_from_mc"] = tmpprof.mean()

            if tmploss.shape[0] > 0:
                optional_outputs["average_loss_from_mc"] = tmploss.mean()

            optional_outputs["probability_of_profit_from_mc"] = (
                self.strategyprofit_mc >= 0.01
            ).sum() / self.strategyprofit_mc.shape[0]

        return Outputs.model_validate(
            optional_outputs
            | {
                "probability_of_profit": self.project_probability,
                "strategy_cost": sum(self.cost),
                "per_leg_cost": self.cost,
                "profit_ranges": self.__profitranges,
                "minimum_return_in_the_domain": self.strategyprofit.min(),
                "maximum_return_in_the_domain": self.strategyprofit.max(),
                "implied_volatility": self.implied_volatility,
                "in_the_money_probability": self.itm_probability,
                "delta": self.delta,
                "gamma": self.gamma,
                "theta": self.theta,
                "vega": self.vega,
            }
        )

    def get_pl(self, leg=-1):
        """
        get_pl -> returns the profit/loss profile of either a leg or the whole
        strategy.

        Parameters
        ----------
        leg : int, optional
            Index of the leg. Default is -1 (whole strategy).

        Returns
        -------
        stock prices : numpy array
            Sequence of stock prices within the bounds of the stock price domain.
        P/L profile : numpy array
            Profit/loss profile of either a leg or the whole strategy.
        """
        if self.profit.size > 0 and leg >= 0 and leg < self.profit.shape[0]:
            return self.__s, self.profit[leg]
        else:
            return self.__s, self.strategyprofit

    def pl_to_csv(self, filename="pl.csv", leg=-1):
        """
        pl_to_csv -> saves the profit/loss data to a .csv file.

        Parameters
        ----------
        filename : string, optional
            Name of the .csv file. Default is 'pl.csv'.
        leg : int, optional
            Index of the leg. Default is -1 (whole strategy).

        Returns
        -------
        None.
        """
        if self.profit.size > 0 and leg >= 0 and leg < self.profit.shape[0]:
            arr = stack((self.__s, self.profit[leg]))
        else:
            arr = stack((self.__s, self.strategyprofit))

        savetxt(
            filename, arr.transpose(), delimiter=",", header="StockPrice,Profit/Loss"
        )

    def plot_pl(self):
        """
        plot_pl -> displays the strategy's profit/loss profile diagram.

        Returns
        -------
        None.
        """
        if len(self.strategyprofit) == 0:
            raise RuntimeError(
                "Before plotting the profit/loss profile diagram, you must run a calculation!"
            )

        rcParams.update({"figure.autolayout": True})

        zeroline = zeros(self.__s.shape[0])
        strikecallbuy = []
        strikeputbuy = []
        zerocallbuy = []
        zeroputbuy = []
        strikecallsell = []
        strikeputsell = []
        zerocallsell = []
        zeroputsell = []
        comment = "P/L profile diagram:\n--------------------\n"
        comment += "The vertical green dashed line corresponds to the position "
        comment += "of the stock's spot price. The right and left arrow "
        comment += "markers indicate the strike prices of calls and puts, "
        comment += "respectively, with blue representing long and red representing "
        comment += "short positions."

        plt.axvline(self.__stockprice, ls="--", color="green")
        plt.xlabel("Stock price")
        plt.ylabel("Profit/Loss")
        plt.xlim(self.__s.min(), self.__s.max())

        for i in range(len(self.__strike)):
            if self.__strike[i] > 0.0:
                if self.__type[i] == "call":
                    if self.__action[i] == "buy":
                        strikecallbuy.append(self.__strike[i])
                        zerocallbuy.append(0.0)
                    elif self.__action[i] == "sell":
                        strikecallsell.append(self.__strike[i])
                        zerocallsell.append(0.0)
                elif self.__type[i] == "put":
                    if self.__action[i] == "buy":
                        strikeputbuy.append(self.__strike[i])
                        zeroputbuy.append(0.0)
                    elif self.__action[i] == "sell":
                        strikeputsell.append(self.__strike[i])
                        zeroputsell.append(0.0)

        if self.__profittarg is not None:
            comment += " The blue dashed line represents the profit target level."
            targetline = full(self.__s.shape[0], self.__profittarg)

        if self.__losslimit is not None:
            comment += " The red dashed line represents the loss limit level."
            lossline = full(self.__s.shape[0], self.__losslimit)

        print(comment)

        if self.__losslimit is not None and self.__profittarg is not None:
            plt.plot(
                self.__s,
                zeroline,
                "m--",
                self.__s,
                lossline,
                "r--",
                self.__s,
                targetline,
                "b--",
                self.__s,
                self.strategyprofit,
                "k-",
                strikecallbuy,
                zerocallbuy,
                "b>",
                strikeputbuy,
                zeroputbuy,
                "b<",
                strikecallsell,
                zerocallsell,
                "r>",
                strikeputsell,
                zeroputsell,
                "r<",
                markersize=10,
            )
        elif self.__losslimit is not None:
            plt.plot(
                self.__s,
                zeroline,
                "m--",
                self.__s,
                lossline,
                "r--",
                self.__s,
                self.strategyprofit,
                "k-",
                strikecallbuy,
                zerocallbuy,
                "b>",
                strikeputbuy,
                zeroputbuy,
                "b<",
                strikecallsell,
                zerocallsell,
                "r>",
                strikeputsell,
                zeroputsell,
                "r<",
                markersize=10,
            )
        elif self.__profittarg is not None:
            plt.plot(
                self.__s,
                zeroline,
                "m--",
                self.__s,
                targetline,
                "b--",
                self.__s,
                self.strategyprofit,
                "k-",
                strikecallbuy,
                zerocallbuy,
                "b>",
                strikeputbuy,
                zeroputbuy,
                "b<",
                strikecallsell,
                zerocallsell,
                "r>",
                strikeputsell,
                zeroputsell,
                "r<",
                markersize=10,
            )
        else:
            plt.plot(
                self.__s,
                zeroline,
                "m--",
                self.__s,
                self.strategyprofit,
                "k-",
                strikecallbuy,
                zerocallbuy,
                "b>",
                strikeputbuy,
                zeroputbuy,
                "b<",
                strikecallsell,
                zerocallsell,
                "r>",
                strikeputsell,
                zeroputsell,
                "r<",
                markersize=10,
            )

    """
    Properties
    ----------
    days2target : int, readonly
        Number of days remaining to the target date from the start date.
    stockpricearray : array
        A Numpy array of consecutive stock prices, from the minimum price up to 
        the maximum price in the stock price domain. It is used to compute the 
        strategy's P/L profile.
    terminalstockprices : array
        A Numpy array or terminal stock prices typically generated by Monte Carlo 
        simulations. It is used to compute strategy's expected profit and loss. 
    """

    @property
    def days2target(self):
        return self.__days2target

    @property
    def stockpricearray(self):
        return self.__s

    @stockpricearray.setter
    def stockpricearray(self, s):
        if isinstance(s, ndarray):
            if s.shape[0] > 0:
                self.__s = s
            else:
                raise ValueError("Empty stock price array is not allowed!")
        else:
            raise TypeError("A numpy array is expected!")

    @property
    def terminalstockprices(self):
        return self.__s_mc

    @terminalstockprices.setter
    def terminalstockprices(self, s):
        if isinstance(s, ndarray):
            if s.shape[0] > 0:
                self.__s_mc = s
            else:
                raise ValueError("Empty terminal stock price array is not allowed!")
        else:
            raise TypeError("A numpy array is expected!")
