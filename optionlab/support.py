from __future__ import division
from scipy import stats
from numpy import asarray, ndarray, exp, abs, round, diff, flatnonzero, arange, inf
from numpy.random import normal, laplace
from numpy.lib.scimath import log, sqrt
from datetime import date, timedelta
from optionlab.black_scholes import get_d1_d2, getoptionprice
from optionlab.__holidays__ import getholidays


def getpayoff(optype, s, x):
    """
    getpayoff(optype,s,x) -> returns the payoff of an option trade at expiration.

    Arguments:
    ----------
    optype: option type (either 'call' or 'put').
    s: a numpy array of stock prices.
    x: strike price.
    """
    if not isinstance(s, ndarray):
        raise TypeError("'s' must be a numpy array!")

    if optype == "call":
        return (s - x + abs(s - x)) / 2.0
    elif optype == "put":
        return (x - s + abs(x - s)) / 2.0
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def getPLoption(optype, opvalue, action, s, x):
    """
    getPLoption(optype,opvalue,action,s,x) -> returns the profit (P) or loss
    (L) per option of an option trade at expiration.

    Arguments:
    ----------
    optype: option type (either 'call' or 'put').
    opvalue: option price.
    action: either 'buy' or 'sell' the option.
    s: a numpy array of stock prices.
    x: strike price.
    """
    if not isinstance(s, ndarray):
        raise TypeError("'s' must be a numpy array!")

    if action == "sell":
        return opvalue - getpayoff(optype, s, x)
    elif action == "buy":
        return getpayoff(optype, s, x) - opvalue
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")


def getPLstock(s0, action, s):
    """
    getPLstock(s0,action,s) -> returns the profit (P) or loss (L) of a stock
    position.

    Arguments:
    ----------
    s0: initial stock price.
    action: either 'buy' or 'sell' the stock.
    s: a numpy array of stock prices.
    """
    if not isinstance(s, ndarray):
        raise TypeError("'s' must be a numpy array!")

    if action == "sell":
        return s0 - s
    elif action == "buy":
        return s - s0
    else:
        raise ValueError("Action must be either 'sell' or 'buy'!")


def getPLprofile(optype, action, x, val, n, s, commission=0.0):
    """
    getPLprofile(optype,action,x,val,n,s,commision) -> returns the profit/loss
    profile and cost of an option trade at expiration.

    Arguments:
    ----------
    optype: option type ('call' or 'put').
    action: either 'buy' or 'sell' the option.
    x: strike price.
    val: option price.
    n: number of options.
    s: a numpy array of stock prices.
    comission: commission charged by the broker (default is zero).
    """
    if not isinstance(s, ndarray):
        raise TypeError("'s' must be a numpy array!")

    if action == "buy":
        cost = -val
    elif action == "sell":
        cost = val
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    if optype in ("call", "put"):
        return (
            n * getPLoption(optype, val, action, s, x) - commission,
            n * cost - commission,
        )
    else:
        raise ValueError("Option type must be either 'call' or 'put'!")


def getPLprofilestock(s0, action, n, s, commission=0.0):
    """
    getPLprofilestock(s0,action,n,s,commission) -> returns the profit/loss
    profile and cost of a stock position.

    Arguments:
    ----------
    s0: initial stock price.
    action: either 'buy' or 'sell' the shares.
    n: number of shares.
    s: a numpy array of stock prices.
    comission: commission charged by the broker (default is zero).
    """
    if not isinstance(s, ndarray):
        raise TypeError("'s' must be a numpy array!")

    if action == "buy":
        cost = -s0
    elif action == "sell":
        cost = s0
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    return n * getPLstock(s0, action, s) - commission, n * cost - commission


def getPLprofileBS(
    optype, action, x, val, r, targ2maturity, volatility, n, s, y=0.0, commission=0.0
):
    """
    getPLprofileBS(optype,action,x,val,r,targ2maturity,volatility,n,s,y,
    commission) -> returns the profit/loss profile and cost of an option trade
    on a target date before maturity using the Black-Scholes model for option
    pricing.

    Arguments:
    ----------
    optype: option type (either 'call' or 'put').
    action: either 'buy' or 'sell' the option.
    x: strike.
    val: option price when the trade was open.
    r: risk-free interest rate.
    targ2maturity: time remaining to maturity from the target date.
    volatility: annualized volatility of the underlying asset.
    n: number of options.
    s: a numpy array of stock prices.
    y: annualized dividend yield (default is zero)
    comission: commission charged by the broker (default is zero).
    """
    if not isinstance(s, ndarray):
        raise TypeError("'s' must be a numpy array!")

    if action == "buy":
        cost = -val
        fac = 1
    elif action == "sell":
        cost = val
        fac = -1
    else:
        raise ValueError("Action must be either 'buy' or 'sell'!")

    d1, d2 = get_d1_d2(s, x, r, volatility, targ2maturity, y)
    calcprice = getoptionprice(optype, s, x, r, targ2maturity, d1, d2, y)

    return fac * n * (calcprice - val) - commission, n * cost - commission


def getnonbusinessdays(startdate, enddate, country="US"):
    """
    getnonbusinessdays -> returns the number of non-business days between
    the start and end date.

    Arguments
    ---------
    startdate: Start date, provided as a 'datetime.date' object.
    enddate: End date, provided as a 'datetime.date' object.
    country: Country for which the holidays will be counted as non-business days
             (default is "US").
    """
    if not (isinstance(startdate, date) and isinstance(enddate, date)):
        raise TypeError("'startdate' and 'enddate' must be 'datetime.date' objects!")

    if enddate > startdate:
        ndays = (enddate - startdate).days
    else:
        raise ValueError("End date must be after start date!")

    nonbusinessdays = 0
    holidays = getholidays(country)

    for i in range(ndays):
        currdate = startdate + timedelta(days=i)

        if currdate.weekday() >= 5 or currdate.strftime("%Y-%m-%d") in holidays:
            nonbusinessdays += 1

    return nonbusinessdays


def createpriceseq(minprice, maxprice):
    """
    createpriceseq(minprice,maxprice) -> generates a sequence of stock prices
    from 'minprice' to 'maxprice' with increment $0.01.

    Arguments:
    ----------
    minprice: minimum stock price in the range.
    maxprice: maximum stock price in the range.
    """
    if maxprice > minprice:
        return round((arange(int(maxprice - minprice) * 100 + 1) * 0.01 + minprice), 2)
    else:
        raise ValueError("Maximum price cannot be less than minimum price!")


def createpricesamples(
    s0, volatility, time2maturity, r=0.01, distribution="black-scholes", y=0.0, n=100000
):
    """
    createpricesamples(s0,volatility,time2maturity,r,distribution,y,n) -> generates
    random stock prices at maturity according to a statistical distribution.

    Arguments:
    ----------
    s0: spot price of the stock.
    volatility: annualized volatility.
    time2maturity: time left to maturity in units of year.
    r: annualized risk-free interest rate (default is 0.01). Used only if
       distribution is 'black-scholes'.
    distribution: statistical distribution used to generate random stock prices
                  at maturity. It can be 'black-scholes' (default), 'normal' or
                  'laplace'.
    y: annualized dividend yield (default is zero).
    n: number of randomly generated terminal prices.
    """
    if distribution == "normal":
        return exp(normal(log(s0), volatility * sqrt(time2maturity), n))
    elif distribution == "black-scholes":
        drift = (r - y - 0.5 * volatility * volatility) * time2maturity

        return exp(normal((log(s0) + drift), volatility * sqrt(time2maturity), n))
    elif distribution == "laplace":
        return exp(laplace(log(s0), (volatility * sqrt(time2maturity)) / sqrt(2.0), n))
    else:
        raise ValueError("Distribution not implemented yet!")


def getprofitrange(s, profit, target=0.01):
    """
    getprofitrange(s,profit,target) -> returns pairs of stock prices, as a list,
    for which an option trade is expected to get the desired profit in between.

    Arguments:
    ----------
    s: a numpy array of stock prices.
    profit: a numpy array containing the profit (or loss) of the trade for each
            stock price in the stock price array.
    target: profit target (0.01 is the default).
    """
    if not (isinstance(s, ndarray) and isinstance(profit, ndarray)):
        raise TypeError("'s' and 'profit' must be numpy arrays!")

    profitrange = []

    t = s[profit >= target]

    if t.shape[0] == 0:
        return profitrange

    mask1 = diff(t) <= target + 0.001
    mask2 = diff(t) > target + 0.001
    maxi = flatnonzero(mask1[:-1] & mask2[1:]) + 1

    for i in range(maxi.shape[0] + 1):
        profitrange.append([])

        if i == 0:
            if t[0] == s[0]:
                profitrange[0].append(0.0)
            else:
                profitrange[0].append(t[0])
        else:
            profitrange[i].append(t[maxi[i - 1] + 1])

        if i == maxi.shape[0]:
            if t[t.shape[0] - 1] == s[s.shape[0] - 1]:
                profitrange[maxi.shape[0]].append(inf)
            else:
                profitrange[maxi.shape[0]].append(t[t.shape[0] - 1])
        else:
            profitrange[i].append(t[maxi[i]])

    return profitrange


def getPoP(profitranges, source="black-scholes", **kwargs):
    """
    getPoP(profitranges,source,kwargs) -> estimates the probability of profit
    (PoP) of an option trade.

    Arguments:
    ----------
    profitranges: a Python list containing the stock price ranges, as given by
                  'getprofitrange()', for which a trade results in profit.
    source: a string. It determines how the probability of profit is estimated
            (see next).
    **kwargs: a Python dictionary. The input that has to be provided depends on
              the value of the 'source' argument:

              * For 'source="normal"' or 'source="laplace"': the probability of
              profit is calculated assuming either a (log)normal or a (log)Laplace
              distribution of terminal stock prices at maturity.
              The keywords 'stockprice', 'volatility' and 'time2maturity' must be
              set.

              * For 'source="black-scholes"' (default): the probability of profit
              is calculated assuming a (log)normal distribution with risk neutrality
              as implemented in the Black-Scholes model.
              The keywords 'stockprice', 'volatility', 'interestrate' and
              'time2maturity' must be set. The keyword 'dividendyield' is optional.

              * For 'source="array"': the probability of profit is calculated
              from a 1D numpy array of stock prices typically at maturity generated
              by a Monte Carlo simulation (or another user-defined data generation
              process); this numpy array must be assigned to the 'array' keyword.
    """
    if not bool(kwargs):
        raise ValueError("'kwargs' is empty, nothing to do!")

    pop = 0.0
    drift = 0.0

    if len(profitranges) == 0:
        return pop

    if source in ("normal", "laplace", "black-scholes"):
        if "stockprice" in kwargs.keys():
            stockprice = float(kwargs["stockprice"])

            if stockprice <= 0.0:
                raise ValueError("Stock price must be greater than zero!")
        else:
            raise ValueError("Stock price must be provided!")

        if "volatility" in kwargs.keys():
            volatility = float(kwargs["volatility"])

            if volatility <= 0.0:
                raise ValueError("Volatility must be greater than zero!")
        else:
            raise ValueError("Volatility must be provided!")

        if "time2maturity" in kwargs.keys():
            time2maturity = float(kwargs["time2maturity"])

            if time2maturity < 0.0:
                raise ValueError("Time left to expiration must be a positive number!")
        else:
            raise ValueError("Time left to expiration must be provided!")

        if source == "black-scholes":
            if "interestrate" in kwargs.keys():
                r = float(kwargs["interestrate"])

                if r < 0.0:
                    raise ValueError(
                        "Risk-free interest rate must be a positive number!"
                    )
            else:
                raise ValueError("Risk-free interest rate must be provided!")

            if "dividendyield" in kwargs.keys():
                y = float(kwargs["dividendyield"])

                if y < 0.0:
                    raise ValueError("Dividend yield must be a positive number!")
            else:
                y = 0.0

            drift = (r - y - 0.5 * volatility * volatility) * time2maturity

        sigma = volatility * sqrt(time2maturity)

        if sigma == 0.0:
            sigma = 1e-10

        if source == "laplace":
            beta = sigma / sqrt(2.0)

        for i in range(len(profitranges)):
            lval = profitranges[i][0]

            if lval <= 0.0:
                lval = 1e-10

            hval = profitranges[i][1]

            if source in ["normal", "black-scholes"]:
                pop += stats.norm.cdf(
                    (log(hval / stockprice) - drift) / sigma
                ) - stats.norm.cdf((log(lval / stockprice) - drift) / sigma)
            else:
                pop += stats.laplace.cdf(
                    log(hval / stockprice) / beta
                ) - stats.laplace.cdf(log(lval / stockprice) / beta)

    elif source == "array":
        if "array" in kwargs.keys():
            stocks = asarray(kwargs["array"])

            if stocks.shape[0] > 0:
                for i in range(len(profitranges)):
                    tmp1 = stocks[stocks >= profitranges[i][0]]
                    tmp2 = tmp1[tmp1 <= profitranges[i][1]]
                    pop += tmp2.shape[0]

                pop = pop / stocks.shape[0]
            else:
                raise ValueError("The array of stock prices is empty!")
        else:
            raise ValueError("An array of stock prices must be provided!")
    else:
        raise ValueError("Source not supported yet!")

    return pop
