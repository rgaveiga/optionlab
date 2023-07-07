# optionlab

This package is a lightweight library written entirely in Python, designed to provide quick evaluation of option strategies.

The code produces various outputs, including the profit/loss profile of the strategy on a user-defined target date, the range of stock prices for which the strategy is profitable (i.e., generating a return greater than \$0.01), the Greeks associated with each leg of the strategy, the resulting debit or credit on the trading account, the maximum and minimum returns within a specified lower and higher price range of the underlying asset, and an estimate of the strategy's probability of profit.

The probability of profit (PoP) for the strategy is calculated based on the distribution of estimated prices of the underlying asset on the user-defined target date. Specifically, for the price range in the payoff where the strategy generates profit, the PoP represents the probability that the stock price will fall within that range. This distribution of underlying asset prices on the target date can be lognormal, log-Laplace, or derived from the Black-Scholes model. Additionally, the distribution can be obtained through simulations (e.g., Monte Carlo) or machine learning models.

Despite the code having been developed with option strategies in mind, it can also be used for strategies that combine options with stocks and/or take into account the profits or losses of closed trades.

## Installation

The easiest way to install *optionlab* is using **pip**:

```
pip install optionlab 
```
## Basic usage

Usage examples for several strategies can be found in the **examples** directory.

To evaluate an option strategy, you need to import the *Strategy* class from the *strategy* module and instantiate it:

```python
from optionlab.strategy import Strategy
st=Strategy()
```

Next, you need to pass the input data to the *getdata()* method of the newly created *Strategy* object. This method accepts the following parameters:

```
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
                Premium effectively paid or received in a previously opened position. If positive, it means
                that the position remains open and the payoff calculation takes this price into account, not
                the current price of the option. If negative, it means that the position is closed and the
                difference between this price and the current price is considered in the payoff calculation.
            "expiration" : string | int
                Expiration date in 'YYYY-MM-DD' format or number of days left before maturity, depending on
                the value in 'use_dates' (see below).
        For stocks, the dictionary should contain up to 4 keys:
            "type" : string
                It must be 'stock'. It is mandatory.
            "n" : int
                Number of shares. It is mandatory.
            "action" : string
                Either 'buy' or 'sell'. It is mandatory.
            "prevpos" : float
                Stock price effectively paid or received in a previously opened position. If positive, it
                means that the position remains open and the payoff calculation takes this price into account,
                not thecurrent price of the stock. If negative, it means that the position is closed and the
                difference between this price and the current price is considered in the payoff calculation.
        For a non-determined previously opened position to be closed, which might consist of any combination
        of calls, puts and stocks, the dictionary must contain two keys:
            "type" : string
                It must be 'closed'. It is mandatory.
            "prevpos" : float
                The total value of the position to be closed, which can be positive if it made a profit or
                negative if it is a loss. It is mandatory.
profittarg : float, optional
    Target profit level. Default is None, which means it is not calculated.
losslimit : float, optional
    Limit loss level. Default is None, which means it is not calculated.
optcommission : float
    Broker commission for options transactions. Default is 0.0.
stockcommission : float
    Broker commission for stocks transactions. Default is 0.0.
compute_the_greeks : logical, optional
    Whether or not Black-Scholes formulas should be used to compute the Greeks. Default is False.
compute_expectation : logical, optional
    Whether or not the strategy's average profit and loss must be computed from a numpy array of random
    terminal prices generated from the chosen distribution. Default is False.
use_dates : logical, optional
    Whether the target and maturity dates are provided or not. If False, the number of days remaining to
    the target date and maturity are provided. Default is True.
discard_nonbusinessdays : logical, optional
    Whether to discard Saturdays and Sundays (and maybe holidays) when counting the number of days between two
    dates. Default is True.
country : string, optional
    Country for which the holidays will be considered if 'discard_nonbusinessdyas' is True. Default is 'US'.
startdate : string, optional
    Start date in the calculations, in 'YYYY-MM-DD' format. Default is "". Mandatory if 'use_dates' is True.
targetdate : string, optional
    Target date in the calculations, in 'YYYY-MM-DD' format. Default is "". Mandatory if 'use_dates' is True.
days2targetdate : int, optional
    Number of days remaining until the target date. Not considered if 'use_dates' is True. Default is 30 days.
distribution : string, optional
    Statistical distribution used to compute probabilities. It can be 'black-scholes', 'normal', 'laplace' or
    'array'. Default is 'black-scholes'.
nmcprices : int, optional
    Number of random terminal prices to be generated when calculationg the average profit and loss of a strategy.
    Default is 100,000.
```

The calculations are performed by calling the *run()* method of the *Strategy* object:

```python
out=st.run()
```

This method returns a Python dictionary with the calculation results stored under the following keys:

```
"ProbabilityOfProfit" : float
    Probability of the strategy yielding at least $0.01.
"ProfitRanges" : list
    A Python list of minimum and maximum stock prices defining ranges in which the strategy makes at least $0.01.
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
    A Python list of minimum and maximum stock prices defining ranges in which the strategy makes at least the
    profit target.                    
"ProbabilityOfLossLimit" : float
    Probability of the strategy losing at least the loss limit.
"AverageProfitFromMC" : float
    Average profit as calculated from Monte Carlo-created terminal stock prices for which the strategy is
    profitable.
"AverageLossFromMC" : float
    Average loss as calculated from Monte Carlo-created terminal stock prices for which the strategy ends
    in loss. 
"ProbabilityOfProfitFromMC" : float
    Probability of the strategy yielding at least $0.01 as calculated from Monte Carlo-created terminal
    stock prices.
```

## Contributions

Although functional, **optionlab** is still in its early stages of development. The author has limited time available to work on this library, which is why contributions from individuals with expertise in options and Python programming are greatly appreciated.

If you have any questions, corrections, comments or suggestions, just [drop a message](mailto:roberto.veiga@ufabc.edu.br).

## Disclaimer

This is free software and is provided as is. The author makes no guarantee that its results are accurate and is not responsible for any losses caused by the use of the code. Bugs can be reported as issues.
