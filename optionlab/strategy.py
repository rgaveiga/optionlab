from __future__ import print_function
from __future__ import division
from numpy import array, ndarray, zeros, full, stack, savetxt
import json
from matplotlib import rcParams
import matplotlib.pyplot as plt
from datetime import date, datetime
from optionlab.black_scholes import getBSinfo, getimpliedvol
from optionlab.support import (
    getPLprofile,
    getPLprofilestock,
    getPLprofileBS,
    getprofitrange,
    getnonbusinessdays,
    createpriceseq,
    createpricesamples,
    getPoP,
)


class Strategy:
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
        self.__compute_the_greeks = False
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

    def getdata(
        self,
        stockprice,
        volatility,
        interestrate,
        minstock,
        maxstock,
        strategy,
        dividendyield=0.0,
        profittarg=None,
        losslimit=None,
        optcommission=0.0,
        stockcommission=0.0,
        compute_the_greeks=False,
        compute_expectation=False,
        use_dates=True,
        discard_nonbusinessdays=True,
        country="US",
        startdate="",
        targetdate="",
        days2targetdate=30,
        distribution="black-scholes",
        nmcprices=100000,
    ):
        """
        getdata -> provides input data to performs calculations for a strategy.

        Parameters
        ----------
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

        Returns
        -------
        None.
        """
        if len(strategy) == 0:
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

        self.__discard_nonbusinessdays = discard_nonbusinessdays

        if self.__discard_nonbusinessdays:
            self.__daysinyear = 252
        else:
            self.__daysinyear = 365

        self.__country = country

        if self.__country in ("United States", "USA", ""):
            self.__country = "US"

        if use_dates:
            startdatetmp = datetime.strptime(startdate, "%Y-%m-%d").date()
            targetdatetmp = datetime.strptime(targetdate, "%Y-%m-%d").date()

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
            self.__days2target = days2targetdate

        for i in range(len(strategy)):
            if "type" in strategy[i].keys():
                self.__type.append(strategy[i]["type"])
            else:
                raise KeyError("Key 'type' is missing!")

            if strategy[i]["type"] in ("call", "put"):
                if "strike" in strategy[i].keys():
                    self.__strike.append(float(strategy[i]["strike"]))
                else:
                    raise KeyError("Key 'strike' is missing!")

                if "premium" in strategy[i].keys():
                    self.__premium.append(float(strategy[i]["premium"]))
                else:
                    raise KeyError("Key 'premium' is missing!")

                if "n" in strategy[i].keys():
                    self.__n.append(int(strategy[i]["n"]))
                else:
                    raise KeyError("Key 'n' is missing!")

                if "action" in strategy[i].keys():
                    self.__action.append(strategy[i]["action"])
                else:
                    raise KeyError("Key 'action' is missing!")

                if "prevpos" in strategy[i].keys():
                    self.__prevpos.append(float(strategy[i]["prevpos"]))
                else:
                    self.__prevpos.append(0.0)

                if "expiration" in strategy[i].keys():
                    if use_dates:
                        expirationtmp = datetime.strptime(
                            strategy[i]["expiration"], "%Y-%m-%d"
                        ).date()
                    else:
                        days2maturitytmp = int(strategy[i]["expiration"])
                else:
                    if use_dates:
                        expirationtmp = self.__targetdate
                    else:
                        days2maturitytmp = self.__days2target

                if use_dates:
                    if expirationtmp >= self.__targetdate:
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
                    if days2maturitytmp >= self.__days2target:
                        self.__days2maturity.append(days2maturitytmp)

                        if days2maturitytmp == self.__days2target:
                            self.__usebs.append(False)
                        else:
                            self.__usebs.append(True)
                    else:
                        raise ValueError(
                            "Days remaining to maturity must be greater than or equal to the number of days remaining to the target date!"
                        )
            elif strategy[i]["type"] == "stock":
                if "n" in strategy[i].keys():
                    self.__n.append(int(strategy[i]["n"]))
                else:
                    raise KeyError("Key 'n' is missing!")

                if "action" in strategy[i].keys():
                    self.__action.append(strategy[i]["action"])
                else:
                    raise KeyError("Key 'action' is missing!")

                if "prevpos" in strategy[i].keys():
                    self.__prevpos.append(float(strategy[i]["prevpos"]))
                else:
                    self.__prevpos.append(0.0)

                self.__strike.append(0.0)
                self.__premium.append(0.0)
                self.__usebs.append(False)
                self.__days2maturity.append(-1)

                if use_dates:
                    self.__expiration.append(self.__targetdate)
                else:
                    self.__expiration.append(-1)
            elif strategy[i]["type"] == "closed":
                if "prevpos" in strategy[i].keys():
                    self.__prevpos.append(float(strategy[i]["prevpos"]))
                else:
                    raise KeyError("Key 'prevpos' is missing!")

                self.__strike.append(0.0)
                self.__n.append(0)
                self.__premium.append(0.0)
                self.__action.append("n/a")
                self.__usebs.append(False)
                self.__days2maturity.append(-1)

                if use_dates:
                    self.__expiration.append(self.__targetdate)
                else:
                    self.__expiration.append(-1)
            else:
                raise ValueError("Type must be 'call', 'put', 'stock' or 'closed'!")

        if distribution in ("black-scholes", "normal", "laplace", "array"):
            self.__distribution = distribution
        else:
            raise ValueError("Distribution not supported yet!")

        self.__stockprice = stockprice
        self.__volatility = volatility
        self.__r = interestrate
        self.__y = dividendyield
        self.__minstock = minstock
        self.__maxstock = maxstock
        self.__profittarg = profittarg
        self.__losslimit = losslimit
        self.__optcommission = optcommission
        self.__stockcommission = stockcommission
        self.__nmcprices = nmcprices
        self.__compute_the_greeks = compute_the_greeks
        self.__compute_expectation = compute_expectation
        self.__use_dates = use_dates

    def getdatafromdict(self, d):
        """
        getdatafromdict -> provides input data in a Python dictionary.

        Parameters
        ----------
        d : dictionary
            A Python dictionary of input data provided as key-value pairs:
                "StockPrice" : float
                    Spot price of the underlying. It is mandatory.
                "Volatility" : float
                    Annualized volatility. It is mandatory.
                "InterestRate" : float
                    Annualized risk-free interest rate. It is mandatory.
                "DividendYield" : float, optional
                    Annualized dividend yield. Default is 0.0.
                "StockPriceDomain" : list
                    A Python list of two prices defining the lower and upper bounds
                    of the stock price domain. It is mandatory.
                "StartDate": string
                    Start date in the calculations, in 'YYYY-MM-DD' format. It
                    is mandatory if 'UseDatesInCalculations' is True (see below).
                "TargetDate" : string
                    Target date in the calculations, in 'YYYY-MM-DD' format. It
                    is mandatory if 'UseDatesInCalculations' is True (see below).
                "DaysToTargetDate" : int
                    Number of days remaining until the target date. It is
                    mandatory if 'UseDatesInCalculations' is False (see below).
                "OptionCommission" : float, optional
                    Broker commission for options transactions. Default is 0.0.
                "StockCommission" : float, optional
                    Broker commission for stocks transactions. Default is 0.0.
                "ProfitTarget" : float, optional
                    Target profit level. Default is None, which means it is not
                    calculated.
                "LossLimit" : float, optional
                    Loss limit level. Default is None, which means it is not
                    calculated.
                "Distribution" : string, optional
                    Statistical distribution used to compute probabilities. It
                    can be 'black-scholes', 'normal' or 'laplace'. Default is
                    'black-scholes'.
                "NumberOfMCPrices" : int, optional
                    Number of random terminal prices to be generated when calculationg
                    the average profit and loss of a strategy. Default is 100,000.
                "ComputeTheGreeks" : logical, optional
                    Whether or not Black-Scholes formulas should be used to compute
                    the Greeks. Default is False.
                "ComputeExpectedProfitLoss" : logical, optional
                    Whether or not the strategy's average profit and loss must be
                    computed from a numpy array of random terminal prices generated
                    from the chosen distribution. Default is False.
                "UseDatesInCalculations" : logical, optional
                    Whether the target and maturity dates are provided. If False,
                    the number of days remaining to the target date and maturity
                    are provided. Default is True.
                "DiscardNonBusinessDays" : logical, optional
                    Whether to discard Saturdays and Sundays (and maybe holidays)
                    when counting the number of days between two dates. Default
                    is True.
                "Country" : string, optional
                    Country for which the holidays will be considered if non-
                    business days are discarded from the calculations. Default
                    is 'US'.
                "Strategy" : list
                    A Python list containing the strategy legs as Python dictionaries.
                    It is mandatory.
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
                            Premium effectively paid or received in a previously
                            opened position. If positive, it means that the position
                            remains open and the payoff calculation takes this
                            price into account, not the current price of the option.
                            If negative, it means that the position is closed and
                            the difference between this price and the current
                            price is considered in the payoff calculation.
                        "expiration" : string | int
                            Expiration date in 'YYYY-MM-DD' format or number of days
                            remaining before maturity, depending on the value in
                            'UseDatesInCalculations'.
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
                            remains open and the payoff calculation takes this
                            price into account, not the current price of the stock.
                            If negative, it means that the position is closed
                            and the difference between this price and the current
                            price is considered in the payoff calculation.
                    For a non-determined previously opened position to be closed,
                    which might consist of any combination of calls, puts and
                    stocks, the dictionary must contain two keys:
                        "type" : string
                            It must be 'closed'. It is mandatory.
                        "prevpos" : float
                            The total value of the position to be closed, which
                            can be positive if it made a profit or negative if
                            it is a loss. It is mandatory.
        """
        if not isinstance(d, dict):
            raise TypeError("A dictionary of input data must be provided!")

        self.__expiration = []
        self.__days2maturity = []
        self.__usebs = []

        if "StockPrice" in d.keys():
            self.__stockprice = float(d["StockPrice"])
        else:
            raise KeyError("Key 'StockPrice' is missing!")

        if "Volatility" in d.keys():
            self.__volatility = float(d["Volatility"])
        else:
            raise KeyError("Key 'Volatility' is missing!")

        if "InterestRate" in d.keys():
            self.__r = float(d["InterestRate"])
        else:
            raise KeyError("Key 'InterestRate' is missing!")

        if "DividendYield" in d.keys():
            self.__y = float(d["DividendYield"])
        else:
            self.__y = 0.0

        if "StockPriceDomain" in d.keys():
            self.__minstock = float(d["StockPriceDomain"][0])
            self.__maxstock = float(d["StockPriceDomain"][1])
        else:
            raise KeyError("Key 'StockPriceDomain' is missing!")

        if "UseDatesInCalculations" in d.keys():
            self.__use_dates = d["UseDatesInCalculations"]
        else:
            self.__use_dates = True

        if "DiscardNonBusinessDays" in d.keys():
            self.__discard_nonbusinessdays = d["DiscardNonBusinessDays"]
        else:
            self.__discard_nonbusinessdays = True

        if self.__discard_nonbusinessdays:
            self.__daysinyear = 252
        else:
            self.__daysinyear = 365

        if "Country" in d.keys():
            self.__country = d["Country"]

            if self.__country in ("United States", "USA", ""):
                self.__country = "US"
        else:
            self.__country = "US"

        if self.__use_dates:
            if "StartDate" in d.keys():
                self.__startdate = datetime.strptime(d["StartDate"], "%Y-%m-%d").date()
            else:
                raise KeyError("Key 'StartDate' is missing!")

            if "TargetDate" in d.keys():
                self.__targetdate = datetime.strptime(
                    d["TargetDate"], "%Y-%m-%d"
                ).date()
            else:
                raise KeyError("Key 'TargetDate' is missing!")

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
            if "DaysToTargetDate" in d.keys():
                self.__days2target = int(d["DaysToTargetDate"])
            else:
                raise KeyError("Key 'DaysToTargetDate' is missing!")

        if "OptionCommission" in d.keys():
            self.__optcommission = float(d["OptionCommission"])
        else:
            self.__optcommission = 0.0

        if "StockCommission" in d.keys():
            self.__stockcommission = float(d["StockCommission"])
        else:
            self.__stockcommission = 0.0

        if "ProfitTarget" in d.keys():
            self.__profittarg = float(d["ProfitTarget"])
        else:
            self.__profittarg = None

        if "LossLimit" in d.keys():
            self.__losslimit = float(d["LossLimit"])
        else:
            self.__losslimit = None

        if "ComputeTheGreeks" in d.keys():
            self.__compute_the_greeks = d["ComputeTheGreeks"]
        else:
            self.__compute_the_greeks = False

        if "ComputeExpectedProfitLoss" in d.keys():
            self.__compute_expectation = d["ComputeExpectedProfitLoss"]
        else:
            self.__compute_expectation = False

        if "Distribution" in d.keys():
            if d["Distribution"] in ("black-scholes", "normal", "laplace", "array"):
                self.__distribution = d["Distribution"]
            else:
                raise ValueError("Distribution not supported yet!")
        else:
            self.__distribution = "black-scholes"

        if "NumberOfMCPrices" in d.keys():
            self.__nmcprices = int(d["NumberOfMCPrices"])
        else:
            self.__nmcprices = 100000

        if "Strategy" in d.keys():
            for strategy in d["Strategy"]:
                if "type" in strategy.keys():
                    self.__type.append(strategy["type"])
                else:
                    raise KeyError("Key 'type' is missing!")

                if strategy["type"] in ("call", "put"):
                    if "strike" in strategy.keys():
                        self.__strike.append(float(strategy["strike"]))
                    else:
                        raise KeyError("Key 'strike' is missing!")

                    if "premium" in strategy.keys():
                        self.__premium.append(float(strategy["premium"]))
                    else:
                        raise KeyError("Key 'premium' is missing!")

                    if "n" in strategy.keys():
                        self.__n.append(int(strategy["n"]))
                    else:
                        raise KeyError("Key 'n' is missing!")

                    if "action" in strategy.keys():
                        self.__action.append(strategy["action"])
                    else:
                        raise KeyError("Key 'action' is missing!")

                    if "prevpos" in strategy.keys():
                        self.__prevpos.append(float(strategy["prevpos"]))
                    else:
                        self.__prevpos.append(0.0)

                    if "expiration" in strategy.keys():
                        if self.__use_dates:
                            expirationtmp = datetime.strptime(
                                strategy["expiration"], "%Y-%m-%d"
                            ).date()
                        else:
                            days2maturitytmp = int(strategy["expiration"])
                    else:
                        if self.__use_dates:
                            expirationtmp = self.__targetdate
                        else:
                            days2maturitytmp = self.__days2target

                    if self.__use_dates:
                        if expirationtmp >= self.__targetdate:
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
                        if days2maturitytmp >= self.__days2target:
                            self.__days2maturity.append(days2maturitytmp)

                            if days2maturitytmp == self.__days2target:
                                self.__usebs.append(False)
                            else:
                                self.__usebs.append(True)
                        else:
                            raise ValueError(
                                "Days remaining to maturity must be greater than or equal to the number of days remaining to the target date!"
                            )
                elif strategy["type"] == "stock":
                    if "n" in strategy.keys():
                        self.__n.append(int(strategy["n"]))
                    else:
                        raise KeyError("Key 'n' is missing!")

                    if "action" in strategy.keys():
                        self.__action.append(strategy["action"])
                    else:
                        raise KeyError("Key 'action' is missing!")

                    if "prevpos" in strategy.keys():
                        self.__prevpos.append(float(strategy["prevpos"]))
                    else:
                        self.__prevpos.append(0.0)

                    self.__strike.append(0.0)
                    self.__premium.append(0.0)
                    self.__usebs.append(False)
                    self.__days2maturity.append(-1)

                    if self.__use_dates:
                        self.__expiration.append(self.__targetdate)
                    else:
                        self.__expiration.append(-1)
                elif strategy["type"] == "closed":
                    if "prevpos" in strategy.keys():
                        self.__prevpos.append(float(strategy["prevpos"]))
                    else:
                        raise KeyError("Key 'prevpos' is missing!")

                    self.__strike.append(0.0)
                    self.__n.append(0)
                    self.__premium.append(0.0)
                    self.__action.append("n/a")
                    self.__usebs.append(False)
                    self.__days2maturity.append(-1)

                    if self.__use_dates:
                        self.__expiration.append(self.__targetdate)
                    else:
                        self.__expiration.append(-1)
                else:
                    raise ValueError("Type must be 'call', 'put', 'stock' or 'closed'!")
        else:
            raise KeyError("Key 'Strategy' is missing!")

    def getdatafromjson(self, jsonstring):
        """
        getdatafromjson -> reads the stock and options data from a JSON string.

        Parameters
        ----------
        jsonstring : string
            Input in JSON format (see 'getdatafromdict()' for more information).

        Returns
        -------
        None.
        """
        self.getdatafromdict(json.loads(jsonstring))

    def run(self):
        """
        run -> runs calculations for an options strategy.

        Returns
        -------
        output : dictionary
            A Python dictionary containing the output of a calculation as
            key-value pairs:
                "ProbabilityOfProfit" : float
                    Probability of the strategy yielding at least $0.01.
                "ProfitRanges" : list
                    A Python list of minimum and maximum stock prices defining
                    ranges in which the strategy makes at least $0.01.
                "StrategyCost" : float
                    Total strategy cost.
                "PerLegCost" : list
                    A Python list of costs, one per strategy leg.
                "ImpliedVolatility" : list
                    A Python list of implied volatilities, one per strategy leg.
                "InTheMoneyProbability" : list
                    A Python list of ITM probabilities, one per strategy leg.
                "Delta" : list
                    A Python list of Delta values, one per strategy leg.
                "Gamma" : list
                    A Python list of Gamma values, one per strategy leg.
                "Theta" : list
                    A Python list of Theta values, one per strategy leg.
                "Vega" : list
                    A Python list of Vega values, one per strategy leg.
                "MinimumReturnInTheDomain" : float
                    Minimum return of the strategy within the stock price domain.
                "MaximumReturnInTheDomain" : float
                    Maximum return of the strategy within the stock price domain.
                "ProbabilityOfProfitTarget" : float
                    Probability of the strategy yielding at least the profit target.
                "ProfitTargetRanges" : list
                    A Python list of minimum and maximum stock prices defining
                    ranges in which the strategy makes at least the profit
                    target.
                "ProbabilityOfLossLimit" : float
                    Probability of the strategy losing at least the loss limit.
                "AverageProfitFromMC" : float
                    Average profit as calculated from Monte Carlo-created terminal
                    stock prices for which the strategy is profitable.
                "AverageLossFromMC" : float
                    Average loss as calculated from Monte Carlo-created terminal
                    stock prices for which the strategy ends in loss.
                "ProbabilityOfProfitFromMC" : float
                    Probability of the strategy yielding at least $0.01 as calculated
                    from Monte Carlo-created terminal stock prices.
        """
        if len(self.__type) == 0:
            raise RuntimeError("No legs in the strategy! Nothing to do!")
        elif self.__type.count("closed") > 1:
            raise RuntimeError("Only one position of type 'closed' is allowed!")
        elif not self.__distribution in ("black-scholes", "normal", "laplace", "array"):
            raise ValueError("Distribution not implemented yet! Nothing to do!")
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

        for i in range(len(self.__type)):
            if self.__type[i] in ("call", "put"):
                if self.__compute_the_greeks and self.__prevpos[i] >= 0.0:
                    time2maturity = self.__days2maturity[i] / self.__daysinyear
                    (
                        calldelta,
                        putdelta,
                        calltheta,
                        puttheta,
                        gamma,
                        vega,
                        callitmprob,
                        putitmprob,
                    ) = getBSinfo(
                        self.__stockprice,
                        self.__strike[i],
                        self.__r,
                        self.__volatility,
                        time2maturity,
                        self.__y,
                    )[
                        2:
                    ]

                    self.gamma.append(gamma)
                    self.vega.append(vega)

                    if self.__type[i] == "call":
                        self.impvol.append(
                            getimpliedvol(
                                "call",
                                self.__premium[i],
                                self.__stockprice,
                                self.__strike[i],
                                self.__r,
                                time2maturity,
                                self.__y,
                            )
                        )
                        self.itmprob.append(callitmprob)

                        if self.__action[i] == "buy":
                            self.delta.append(calldelta)
                            self.theta.append(calltheta / self.__daysinyear)
                        else:
                            self.delta.append(-calldelta)
                            self.theta.append(-calltheta / self.__daysinyear)
                    else:
                        self.impvol.append(
                            getimpliedvol(
                                "put",
                                self.__premium[i],
                                self.__stockprice,
                                self.__strike[i],
                                self.__r,
                                time2maturity,
                                self.__y,
                            )
                        )
                        self.itmprob.append(putitmprob)

                        if self.__action[i] == "buy":
                            self.delta.append(putdelta)
                            self.theta.append(puttheta / self.__daysinyear)
                        else:
                            self.delta.append(-putdelta)
                            self.theta.append(-puttheta / self.__daysinyear)
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
                            self.__type[i],
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
                                self.__type[i],
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
                            self.__type[i],
                            self.__action[i],
                            self.__strike[i],
                            opval,
                            self.__n[i],
                            self.__s,
                            self.__optcommission,
                        )

                        if self.__compute_expectation or self.__distribution == "array":
                            self.profit_mc[i] = getPLprofile(
                                self.__type[i],
                                self.__action[i],
                                self.__strike[i],
                                opval,
                                self.__n[i],
                                self.__s_mc,
                                self.__optcommission,
                            )[0]
            elif self.__type[i] == "stock":
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
            elif self.__type[i] == "closed":
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

        output = {
            "ProbabilityOfProfit": self.profitprob,
            "StrategyCost": sum(self.cost),
            "PerLegCost": self.cost,
            "ProfitRanges": self.__profitranges,
            "MinimumReturnInTheDomain": self.strategyprofit.min(),
            "MaximumReturnInTheDomain": self.strategyprofit.max(),
        }

        if self.__compute_the_greeks:
            output["ImpliedVolatility"] = self.impvol
            output["InTheMoneyProbability"] = self.itmprob
            output["Delta"] = self.delta
            output["Gamma"] = self.gamma
            output["Theta"] = self.theta
            output["Vega"] = self.vega

        if self.__profittarg is not None:
            output["ProbabilityOfProfitTarget"] = self.profittargprob
            output["ProfitTargetRanges"] = self.__profittargrange

        if self.__losslimit is not None:
            output["ProbabilityOfLossLimit"] = self.losslimitprob

        if (
            self.__compute_expectation or self.__distribution == "array"
        ) and self.__s_mc.shape[0] > 0:
            tmpprof = self.strategyprofit_mc[self.strategyprofit_mc >= 0.01]
            tmploss = self.strategyprofit_mc[self.strategyprofit_mc < 0.0]
            output["AverageProfitFromMC"] = 0.0
            output["AverageLossFromMC"] = 0.0

            if tmpprof.shape[0] > 0:
                output["AverageProfitFromMC"] = tmpprof.mean()

            if tmploss.shape[0] > 0:
                output["AverageLossFromMC"] = tmploss.mean()

            output["ProbabilityOfProfitFromMC"] = (
                self.strategyprofit_mc >= 0.01
            ).sum() / self.strategyprofit_mc.shape[0]

        return output

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
