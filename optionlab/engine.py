from __future__ import division
from __future__ import print_function

import datetime as dt
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy import array, ndarray, zeros, full, stack, savetxt

from optionlab.black_scholes import get_bs_info, _get_implied_vol
from optionlab.models import (
    Inputs,
    Action,
    StrategyType,
    Range,
    OptionStrategy,
    StockStrategy,
    ClosedPosition,
    Outputs,
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
        self.__s_mc = array([])
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
        self.impvol: list[float] = []
        self.itmprob: list[float] = []
        self.delta: list[float] = []
        self.gamma: list[float] = []
        self.vega: list[float] = []
        self.theta: list[float] = []
        self.cost: list[float] = []
        self.profitprob = 0.0
        self.profittargprob = 0.0
        self.losslimitprob = 0.0
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
        self.__nmcprices = inputs.nmc_prices
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
        if self.__distribution == "array" and self.__s_mc.shape[0] == 0:
            raise RuntimeError(
                "No terminal stock prices from Monte Carlo simulations! Nothing to do!"
            )

        time2target = self.__days2target / self.__daysinyear
        self.cost = [0.0 for _ in range(len(self.__type))]
        self.impvol = []
        self.itmprob = []
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
                    time2maturity = self.__days2maturity[i] / self.__daysinyear
                    bs = get_bs_info(
                        self.__stockprice,
                        self.__strike[i],
                        self.__r,
                        self.__volatility,
                        time2maturity,
                        self.__y,
                    )

                    self.gamma.append(bs.gamma)
                    self.vega.append(bs.vega)

                    if type == "call":
                        self.impvol.append(
                            _get_implied_vol(
                                "call",
                                self.__premium[i],
                                self.__stockprice,
                                self.__strike[i],
                                self.__r,
                                time2maturity,
                                self.__y,
                            )
                        )
                        self.itmprob.append(bs.call_itm_prob)

                        if self.__action[i] == "buy":
                            self.delta.append(bs.call_delta)
                            self.theta.append(bs.call_theta / self.__daysinyear)
                        else:
                            self.delta.append(-bs.call_delta)
                            self.theta.append(-bs.call_theta / self.__daysinyear)
                    else:
                        self.impvol.append(
                            _get_implied_vol(
                                "put",
                                self.__premium[i],
                                self.__stockprice,
                                self.__strike[i],
                                self.__r,
                                time2maturity,
                                self.__y,
                            )
                        )
                        self.itmprob.append(bs.put_itm_prob)

                        if self.__action[i] == "buy":
                            self.delta.append(bs.put_delta)
                            self.theta.append(bs.put_theta / self.__daysinyear)
                        else:
                            self.delta.append(-bs.put_delta)
                            self.theta.append(-bs.put_theta / self.__daysinyear)
                else:
                    self.impvol.append(0.0)
                    self.itmprob.append(0.0)
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
                self.impvol.append(0.0)
                self.itmprob.append(1.0)
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
                self.impvol.append(0.0)
                self.itmprob.append(0.0)
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

        if self.__profitranges:
            if self.__distribution in ("normal", "laplace", "black-scholes"):
                self.profitprob = get_pop(
                    self.__profitranges,
                    self.__distribution,
                    stockprice=self.__stockprice,
                    volatility=self.__volatility,
                    time2maturity=time2target,
                    interestrate=self.__r,
                    dividendyield=self.__y,
                )
            elif self.__distribution == "array":
                self.profitprob = get_pop(
                    self.__profitranges, self.__distribution, array=self.__s_mc
                )

        if self.__profittarg is not None:
            self.__profittargrange = get_profit_range(
                self.__s, self.strategyprofit, self.__profittarg
            )

            if self.__profittargrange:
                if self.__distribution in ("normal", "laplace", "black-scholes"):
                    self.profittargprob = get_pop(
                        self.__profittargrange,
                        self.__distribution,
                        stockprice=self.__stockprice,
                        volatility=self.__volatility,
                        time2maturity=time2target,
                        interestrate=self.__r,
                        dividendyield=self.__y,
                    )
                elif self.__distribution == "array":
                    self.profittargprob = get_pop(
                        self.__profittargrange, self.__distribution, array=self.__s_mc
                    )

        if self.__losslimit is not None:
            self.__losslimitranges = get_profit_range(
                self.__s, self.strategyprofit, self.__losslimit + 0.01
            )

            if self.__losslimitranges:
                if self.__distribution in ("normal", "laplace", "black-scholes"):
                    self.losslimitprob = 1.0 - get_pop(
                        self.__losslimitranges,
                        self.__distribution,
                        stockprice=self.__stockprice,
                        volatility=self.__volatility,
                        time2maturity=time2target,
                        interestrate=self.__r,
                        dividendyield=self.__y,
                    )
                elif self.__distribution == "array":
                    self.losslimitprob = 1.0 - get_pop(
                        self.__losslimitranges, self.__distribution, array=self.__s_mc
                    )

        optional_outputs = {}

        if self.__profittarg is not None:
            optional_outputs["probability_of_profit_target"] = self.profittargprob
            optional_outputs["project_target_ranges"] = self.__profittargrange

        if self.__losslimit is not None:
            optional_outputs["probability_of_loss_limit"] = self.losslimitprob

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
                optional_outputs["AverageProfitFromMC"] = tmpprof.mean()

            if tmploss.shape[0] > 0:
                optional_outputs["AverageLossFromMC"] = tmploss.mean()

            optional_outputs["ProbabilityOfProfitFromMC"] = (
                self.strategyprofit_mc >= 0.01
            ).sum() / self.strategyprofit_mc.shape[0]

        return Outputs.model_validate(
            optional_outputs
            | {
                "probability_of_profit": self.profitprob,
                "strategy_cost": sum(self.cost),
                "per_leg_cost": self.cost,
                "profit_ranges": self.__profitranges,
                "minimum_return_in_the_domain": self.strategyprofit.min(),
                "maximum_return_in_the_domain": self.strategyprofit.max(),
                "implied_volatility": self.impvol,
                "in_the_money_probability": self.itmprob,
                "delta": self.delta,
                "gamma": self.gamma,
                "theta": self.theta,
                "vega": self.vega,
            }
        )

    def getPL(self, leg=-1):
        """
        getPL -> returns the profit/loss profile of either a leg or the whole
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

    def PL2csv(self, filename="pl.csv", leg=-1):
        """
        PL2csv -> saves the profit/loss data to a .csv file.

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

    def plotPL(self):
        """
        plotPL -> displays the strategy's profit/loss profile diagram.

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
