# OptionLab

This package is a lightweight library written entirely in Python, designed to provide 
quick evaluation of option strategies.

The code produces various outputs, including the profit/loss profile of the strategy on 
a user-defined target date, the range of stock prices for which the strategy is 
profitable (i.e., generating a return greater than \$0.01), the Greeks associated with 
each leg of the strategy, the resulting debit or credit on the trading account, the 
maximum and minimum returns within a specified lower and higher price range of the 
underlying asset, and an estimate of the strategy's probability of profit.

The probability of profit (PoP) for the strategy is calculated based on the distribution 
of estimated prices of the underlying asset on the user-defined target date. 
Specifically, for the price range in the payoff where the strategy generates profit, the 
PoP represents the probability that the stock price will fall within that range. This 
distribution of underlying asset prices on the target date can be lognormal, 
log-Laplace, or derived from the Black-Scholes model. Additionally, the distribution can 
be obtained through simulations (e.g., Monte Carlo) or machine learning models.

Despite the code having been developed with option strategies in mind, it can also be 
used for strategies that combine options with stocks and/or take into account the 
profits or losses of closed trades.

If you have any questions, corrections, comments or suggestions, just 
[drop a message](mailto:roberto.veiga@ufabc.edu.br).

You can also reach me on [Linkedin](https://www.linkedin.com/in/roberto-gomes-phd-8a718317b/).

## Installation

The easiest way to install **OptionLab** is using **pip**:

```
pip install optionlab
```

## Basic usage

Usage examples for several strategies can be found in the **examples** directory.

To evaluate an option strategy, an `Inputs` model needs to be created:

```python
from optionlab import Inputs
inputs = Inputs.model_validate(inputs_data)
```

The input data passed to `model_validate` above needs to be of the following structure: 

---

- `stock_price` : float
  - Spot price of the underlying.

- `volatility` : float
  - Annualized volatility.

- `interest_rate` : float
  - Annualized risk-free interest rate.

- `min_stock` : float
  - Minimum value of the stock in the stock price domain.

- `max_stock` : float
  - Maximum value of the stock in the stock price domain.

- `strategy` : list
  - A list of `Strategy`.

- `dividend_yield` : float, optional
  - Annualized dividend yield. Default is 0.0.

- `profit_target` : float, optional
  - Target profit level. Default is None, which means it is not calculated.

- `loss_limit` : float, optional
  - Limit loss level. Default is None, which means it is not calculated.

- `opt_commission` : float
  - Broker commission for options transactions. Default is 0.0.

- `stock_commission` : float
  - Broker commission for stocks transactions. Default is 0.0.

- `compute_expectation` : logical, optional
  - Whether or not the strategy's average profit and loss must be computed from a numpy 
  array of random terminal prices generated from the chosen distribution. Default is 
  False.

- `discard_nonbusinessdays` : logical, optional
  - Whether to discard Saturdays and Sundays (and maybe holidays) when counting the 
  number of days between two dates. Default is True.

- `country` : string, optional
  - Country for which the holidays will be considered if 'discard_nonbusinessdays' is 
  True. Default is 'US'.

- `start_date` : dt.date, optional
  - Start date in the calculations. If not provided, days_to_target_date must be 
  provided.

- `target_date` : dt.date, optional
  - Target date in the calculations. If not provided, days_to_target_date must be 
  provided.

- `days_to_target_date` : int, optional
  - Number of days until the target date, typically the maturity date of the options. 
  If not provided, start_date and end_date must be provided.

- `distribution` : string, optional
  - Statistical distribution used to compute probabilities. It can be 'black-scholes', 
  'normal', 'laplace' or 'array'. Default is 'black-scholes'.

- `mc_prices_number` : int, optional
  - Number of random terminal prices to be generated when calculating the average 
  profit and loss of a strategy. Default is 100,000.

---

The `strategy` attribute can be either of type `OptionStrategy`, `StockStrategy`, or 
`ClosedPosition`.

The `OptionStrategy` structure:

---

- `type` : string
  - Either 'call' or 'put'. It is mandatory.

- `strike` : float
  - Option strike price. It is mandatory.

- `premium` : float
  - Option premium. It is mandatory.

- `n` : int
  - Number of options. It is mandatory.

- `action` : string
  - Either 'buy' or 'sell'. It is mandatory.

- `prev_pos` : float
  - Premium effectively paid or received in a previously opened position. If positive, 
  it means that the position remains open and the payoff calculation takes this price 
  into account, not the current price of the option. If negative, it means that the 
  position is closed and the difference between this price and the current price is 
  considered in the payoff calculation.

- `expiration` : string | int
  - Expiration date or days to maturity.

---

`StockStrategy`:

---

- `type` : string
  - It must be 'stock'. It is mandatory.

- `n` : int
  - Number of shares. It is mandatory.

- `action` : string
  - Either 'buy' or 'sell'. It is mandatory.

- `prev_pos` : float
  - Stock price effectively paid or received in a previously opened position. If 
  positive, it means that the position remains open and the payoff calculation 
  takes this price into account, not the current price of the stock. If negative, it 
  means that the position is closed and the difference between this price and the 
  current price is considered in the payoff calculation.

---

For a non-determined previously opened position to be closed, which might consist 
of any combination of calls, puts and stocks, the `ClosedPosition` must contain two 
keys:

---

- `type` : string
  - It must be 'closed'. It is mandatory.

- `prev_pos` : float
  - The total value of the position to be closed, which can be positive if it made 
  a profit or negative if it is a loss. It is mandatory.

---

For example, let's say we wanted to calculate the probability of profit for naked 
calls on Apple stocks with maturity on December 17, 2021. The strategy setup consisted 
of selling 100 175.00 strike calls for 1.15 each on November 22, 2021.

```python
inputs_data = {
    "stock_price": 164.04,
    "start_date": "2021-11-22",
    "target_date": "2021-12-17",
    "volatility": 0.272,
    "interest_rate": 0.0002,
    "min_stock": 120,
    "max_stock": 200,
    "strategy": [
        {
            "type": "call",
            "strike": 175.0,
            "premium": 1.15,
            "n": 100,
            "action":"sell"
        }
    ],
}
```

The simplest way to perform the calculations is by calling the `run_strategy` function 
as follows:

```python
from optionlab import run_strategy

out = run_strategy(inputs_data)
```

Alternatively, an `Inputs` object can be passed to the `StrategyEngine` object and 
the calculations are performed by calling the `run` method of the `StrategyEngine` 
object:

```python
from optionlab import StrategyEngine

st = StrategyEngine(Inputs.model_validate(inputs_data))
out = st.run()
```

In both cases, `out` contains an `Outputs` object with the following structure:

---

- `probability_of_profit` : float
  - Probability of the strategy yielding at least $0.01.

- `profit_ranges` : list
  - A list of minimum and maximum stock prices defining ranges in which the strategy 
  makes at least $0.01.

- `strategy_cost` : float
  - Total strategy cost.

- `per_leg_cost` : list
  - A list of costs, one per strategy leg.

- `implied_volatility` : list
  - A Python list of implied volatilities, one per strategy leg.

- `in_the_money_probability` : list
  - A list of ITM probabilities, one per strategy leg.

- `delta` : list
  - A list of Delta values, one per strategy leg.

- `gamma` : list
  - A list of Gamma values, one per strategy leg.

- `theta` : list
  - A list of Theta values, one per strategy leg.

- `vega` : list
  - A list of Vega values, one per strategy leg.

- `minimum_return_in_the_domain` : float
  - Minimum return of the strategy within the stock price domain.

- `maximum_return_in_the_domain` : float
  - Maximum return of the strategy within the stock price domain.

- `probability_of_profit_target` : float, optional
  - Probability of the strategy yielding at least the profit target.

- `profit_target_ranges` : list, optional
  - A list of minimum and maximum stock prices defining ranges in which the strategy 
  makes at least the profit target.

- `probability_of_loss_limit` : float, optional
  - Probability of the strategy losing at least the loss limit.

- `average_profit_from_mc` : float, optional
  - Average profit as calculated from Monte Carlo-created terminal stock prices for 
  which the strategy is profitable.

- `average_loss_from_mc` : float, optional
  - Average loss as calculated from Monte Carlo-created terminal stock prices for 
  which the strategy ends in loss.

- `probability_of_profit_from_mc` : float, optional
  - Probability of the strategy yielding at least $0.01 as calculated from Monte 
  Carlo-created terminal stock prices.
---

To obtain the probability of profit of the naked call example above:

```python
print("Probability of Profit (PoP): %.1f%%" % (out.probability_of_profit * 100.0)) # 84.5%, according to the calculations
```

## Contributions

### Dev setup

This repository uses `poetry` as a package manager. Install `poetry` as per the 
[poetry docs](https://python-poetry.org/docs/#installing-with-pipx). It is 
recommended to install `poetry` version 1.4.0 if there are issues with the latest 
versions.

Once `poetry` is installed, set up your virtual environment for the repository with 
the following:

```
cd optionlab/
python3.10 venv venv
source venv/bin/activate
poetry install
```

That should install all your dependencies and make you ready to contribute. Please 
add tests for all new features and bug fixes and make sure you are formatting 
with [black](https://github.com/psf/black).

Optionally, to use Jupyter, you can install it with: `pip install juypter`.

#### Git Hooks

This repo uses git hooks. Git hooks are scripts that run automatically every time 
a particular event occurs in a Git repository. These events can include committing, 
merging, and pushing, among others. Git hooks allow developers to enforce certain 
standards or checks before actions are completed in the repository, enhancing the 
workflow and code quality.

The pre-commit framework is a tool that leverages Git hooks to run checks on the 
code before it is committed to the repository. By using pre-commit, developers can 
configure various plugins or hooks that automatically check for syntax errors, 
formatting issues, or even run tests on the code being committed. This ensures that 
only code that passes all the defined checks can be added to the repository, helping 
to maintain code quality and prevent issues from being introduced. 

To install the pre-commit framework on a system with Homebrew, follow these steps:

```
brew install pre-commit
```

Once pre-commit is installed, navigate to the root directory of your Git repository 
where you want to enable pre-commit hooks. Then, run the following command to set up 
pre-commit for that repository. This command installs the Git hook scripts that the 
pre-commit framework will use to run checks before commits.

```
pre-commit install
```

Now, before each commit, the pre-commit hooks you've configured will automatically 
run. If any hook fails, the commit will be aborted, allowing you to fix the issues 
before successfully committing your changes. This process helps maintain a high code 
quality and ensures that common issues are addressed early in 
the development process.

To check all files in a repository with pre-commit, use:

```
pre-commit run --all-files
```

## Disclaimer

This is free software and is provided as is. The author makes no guarantee that its 
results are accurate and is not responsible for any losses caused by the use of the 
code. Bugs can be reported as issues.
