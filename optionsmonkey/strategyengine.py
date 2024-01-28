from __future__ import print_function
from __future__ import division
from numpy import array, ndarray, zeros, full, stack, savetxt
import json
from matplotlib import rcParams
import matplotlib.pyplot as plt
from datetime import date, datetime
from optionsmonkey.black_scholes import get_bs_info, get_implied_vol
from optionsmonkey.models import Inputs, StockStrategy, Strategy, Outputs
from optionsmonkey.support import (
    getPLprofile,
    getPLprofilestock,
    getPLprofileBS,
    getprofitrange,
    getnonbusinessdays,
    createpriceseq,
    createpricesamples,
    getPoP,
)


class StrategyEngine:
    def __init__(self):
        """
        __init__ -> initializes class variables.

        Returns
        -------
        None.
        """
        self.__s = array([])
        self.__s_mc = array([])
        self.__strike = []
        self.__premium = []
        self.__n = []
        self.__action = []
        self.__type = []
        self.__expiration = []
        self.__prevpos = []
        self.__usebs = []
        self.__profitranges = []
        self.__profittargrange = []
        self.__losslimitranges = []
        self.__days2maturity = []
        self.__stockprice = None
        self.__volatility = None
        self.__startdate = date.today()
        self.__targetdate = self.__startdate
        self.__r = None
        self.__y = 0.0
        self.__profittarg = None
        self.__losslimit = None
        self.__optcommission = 0.0
        self.__stockcommission = 0.0
        self.__minstock = None
        self.__maxstock = None
        self.__distribution = "black-scholes"
        self.__country = "US"
        self.__days2target = 30
        self.__nmcprices = 100000
        self.__compute_expectation = False
        self.__use_dates = True
        self.__discard_nonbusinessdays = True
        self.__daysinyear = 252
        self.impvol = []
        self.itmprob = []
        self.delta = []
        self.gamma = []
        self.vega = []
        self.theta = []
        self.cost = []
        self.profitprob = 0.0
        self.profittargprob = 0.0
        self.losslimitprob = 0.0

    def getdata(self, inputs: Inputs):
        """
        getdata -> provides input data to performs calculations for a strategy.

        Parameters
        ----------
        inputs: Inputs

        Returns
        -------
        None.
        """
        if len(inputs.strategy) == 0:
            raise ValueError("No strategy provided!")

        self.__type = []
        self.__strike = []
        self.__premium = []
        self.__n = []
        self.__action = []
        self.__prevpos = []
        self.__expiration = []
        self.__days2maturity = []
        self.__usebs = []

        self.__discard_nonbusinessdays = inputs.discard_nonbusinessdays

        if self.__discard_nonbusinessdays:
            self.__daysinyear = 252
        else:
            self.__daysinyear = 365

        self.__country = inputs.country

        if inputs.use_dates:
            startdatetmp = datetime.strptime(inputs.startdate, "%Y-%m-%d").date()
            targetdatetmp = datetime.strptime(inputs.targetdate, "%Y-%m-%d").date()

            if targetdatetmp > startdatetmp:
                self.__startdate = startdatetmp
                self.__targetdate = targetdatetmp

                if self.__discard_nonbusinessdays:
                    ndiscardeddays = getnonbusinessdays(
                        self.__startdate, self.__targetdate, self.__country
                    )
                else:
                    ndiscardeddays = 0

                self.__days2target = (
                    self.__targetdate - self.__startdate
                ).days - ndiscardeddays
            else:
                raise ValueError("Start date cannot be after the target date!")
        else:
            self.__days2target = inputs.days2targetdate

        for i, strat in enumerate(inputs.strategy):
            strategy: Strategy = strat
            self.__type.append(strategy.type)

            if strategy.type in ("call", "put"):
                self.__strike.append(strategy.strike)  # type: ignore
                self.__premium.append(strategy.premium)  # type: ignore
                self.__n.append(strategy.n)  # type: ignore
                self.__action.append(strategy.action)  # type: ignore
                self.__prevpos.append(strategy.prevpos or 0.0)

                if strategy.expiration:  # type: ignore
                    if inputs.use_dates:
                        expirationtmp = datetime.strptime(
                            strategy.expiration, "%Y-%m-%d"  # type: ignore
                        ).date()
                    else:
                        days2maturitytmp = strategy.expiration  # type: ignore
                else:
                    if inputs.use_dates:
                        expirationtmp = self.__targetdate
                    else:
                        days2maturitytmp = self.__days2target

                if inputs.use_dates:
                    if expirationtmp >= self.__targetdate:  # FIXME
                        self.__expiration.append(expirationtmp)

                        if self.__discard_nonbusinessdays:
                            ndiscardeddays = getnonbusinessdays(
                                self.__startdate, expirationtmp, self.__country
                            )
                        else:
                            ndiscardeddays = 0

                        self.__days2maturity.append(
                            (expirationtmp - self.__startdate).days - ndiscardeddays
                        )

                        if expirationtmp == self.__targetdate:
                            self.__usebs.append(False)
                        else:
                            self.__usebs.append(True)
                    else:
                        raise ValueError(
                            "Expiration date must be after or equal to the target date!"
                        )
                else:
                    if days2maturitytmp >= self.__days2target:  # FIXME
                        self.__days2maturity.append(days2maturitytmp)

                        if days2maturitytmp == self.__days2target:
                            self.__usebs.append(False)
                        else:
                            self.__usebs.append(True)
                    else:
                        raise ValueError(
                            "Days remaining to maturity must be greater than or equal to the number of days remaining to the target date!"
                        )
            elif strategy.type == "stock":
                self.__n.append(strategy.n)
                self.__action.append(strategy.action)
                self.__prevpos.append(strategy.prevpos or 0.0)

                self.__strike.append(0.0)
                self.__premium.append(0.0)
                self.__usebs.append(False)
                self.__days2maturity.append(-1)
                self.__expiration.append(self.__targetdate if inputs.use_dates else -1)

            elif strategy.type == "closed":
                self.__prevpos.append(strategy.prevpos)
                self.__strike.append(0.0)
                self.__n.append(0)
                self.__premium.append(0.0)
                self.__action.append("n/a")
                self.__usebs.append(False)
                self.__days2maturity.append(-1)
                self.__expiration.append(self.__targetdate if inputs.use_dates else -1)
            else:
                raise ValueError("Type must be 'call', 'put', 'stock' or 'closed'!")

        self.__distribution = inputs.distribution
        self.__stockprice = inputs.stockprice
        self.__volatility = inputs.volatility
        self.__r = inputs.interestrate
        self.__y = inputs.dividendyield
        self.__minstock = inputs.minstock
        self.__maxstock = inputs.maxstock
        self.__profittarg = inputs.profittarg
        self.__losslimit = inputs.losslimit
        self.__optcommission = inputs.optcommission
        self.__stockcommission = inputs.stockcommission
        self.__nmcprices = inputs.nmcprices
        self.__compute_expectation = inputs.compute_expectation
        self.__use_dates = inputs.use_dates

    def run(self):
        """
        run -> runs calculations for an options strategy.

        Returns
        -------
        outputs : Outputs
        """
        if len(self.__type) == 0:
            raise RuntimeError("No legs in the strategy! Nothing to do!")
        elif self.__type.count("closed") > 1:
            raise RuntimeError("Only one position of type 'closed' is allowed!")
        elif self.__distribution == "array" and self.__s_mc.shape[0] == 0:
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
            self.__s = createpriceseq(self.__minstock, self.__maxstock)

        self.profit = zeros((len(self.__type), self.__s.shape[0]))
        self.strategyprofit = zeros(self.__s.shape[0])

        if self.__compute_expectation and self.__s_mc.shape[0] == 0:
            self.__s_mc = createpricesamples(
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
                            get_implied_vol(
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
                            get_implied_vol(
                                "put",
                                self.__premium[i],
                                self.__stockprice,
                                self.__strike[i],
                                self.__r,
                                time2maturity,
                                self.__y,
                            )
                        )
                        self.itmprob.append(bs.putitmprob)

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
                        self.profit[i], self.cost[i] = getPLprofileBS(
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
                            self.profit_mc[i] = getPLprofileBS(
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
                        self.profit[i], self.cost[i] = getPLprofile(
                            type,
                            self.__action[i],
                            self.__strike[i],
                            opval,
                            self.__n[i],
                            self.__s,
                            self.__optcommission,
                        )

                        if self.__compute_expectation or self.__distribution == "array":
                            self.profit_mc[i] = getPLprofile(
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

                    self.profit[i], self.cost[i] = getPLprofilestock(
                        stockpos,
                        self.__action[i],
                        self.__n[i],
                        self.__s,
                        self.__stockcommission,
                    )

                    if self.__compute_expectation or self.__distribution == "array":
                        self.profit_mc[i] = getPLprofilestock(
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

        self.__profitranges = getprofitrange(self.__s, self.strategyprofit)

        if self.__profitranges:
            if self.__distribution in ("normal", "laplace", "black-scholes"):
                self.profitprob = getPoP(
                    self.__profitranges,
                    self.__distribution,
                    stockprice=self.__stockprice,
                    volatility=self.__volatility,
                    time2maturity=time2target,
                    interestrate=self.__r,
                    dividendyield=self.__y,
                )
            elif self.__distribution == "array":
                self.profitprob = getPoP(
                    self.__profitranges, self.__distribution, array=self.__s_mc
                )

        if self.__profittarg is not None:
            self.__profittargrange = getprofitrange(
                self.__s, self.strategyprofit, self.__profittarg
            )

            if self.__profittargrange:
                if self.__distribution in ("normal", "laplace", "black-scholes"):
                    self.profittargprob = getPoP(
                        self.__profittargrange,
                        self.__distribution,
                        stockprice=self.__stockprice,
                        volatility=self.__volatility,
                        time2maturity=time2target,
                        interestrate=self.__r,
                        dividendyield=self.__y,
                    )
                elif self.__distribution == "array":
                    self.profittargprob = getPoP(
                        self.__profittargrange, self.__distribution, array=self.__s_mc
                    )

        if self.__losslimit is not None:
            self.__losslimitranges = getprofitrange(
                self.__s, self.strategyprofit, self.__losslimit + 0.01
            )

            if self.__losslimitranges:
                if self.__distribution in ("normal", "laplace", "black-scholes"):
                    self.losslimitprob = 1.0 - getPoP(
                        self.__losslimitranges,
                        self.__distribution,
                        stockprice=self.__stockprice,
                        volatility=self.__volatility,
                        time2maturity=time2target,
                        interestrate=self.__r,
                        dividendyield=self.__y,
                    )
                elif self.__distribution == "array":
                    self.losslimitprob = 1.0 - getPoP(
                        self.__losslimitranges, self.__distribution, array=self.__s_mc
                    )

        opt_outputs = {}

        if self.__profittarg is not None:
            opt_outputs["probability_of_profit_target"] = self.profittargprob
            opt_outputs["project_target_ranges"] = self.__profittargrange

        if self.__losslimit is not None:
            opt_outputs["probability_of_loss_limit"] = self.losslimitprob

        if (
            self.__compute_expectation or self.__distribution == "array"
        ) and self.__s_mc.shape[0] > 0:
            tmpprof = self.strategyprofit_mc[self.strategyprofit_mc >= 0.01]
            tmploss = self.strategyprofit_mc[self.strategyprofit_mc < 0.0]
            opt_outputs["average_profit_from_mc"] = (
                tmpprof.mean() if tmpprof.shape[0] > 0 else 0.0
            )
            opt_outputs["average_loss_from_mc"] = (
                tmploss.mean() if tmploss.shape[0] > 0 else 0.0
            )

            opt_outputs["probability_of_profit_from_mc"] = (
                self.strategyprofit_mc >= 0.01
            ).sum() / self.strategyprofit_mc.shape[0]

        return Outputs.model_validate(
            opt_outputs
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
