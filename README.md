# OptionLab

This package is a lightweight library written entirely in Python, designed to provide 
quick evaluation of option strategies.

The code produces various outputs, including the profit/loss profile of the strategy on 
a user-defined target date, the range of stock prices for which the strategy is 
profitable (i.e., generating a return greater than \$0.01), the Greeks associated with 
each leg of the strategy using the Black-Sholes model, the resulting debit or credit on the 
trading account, the maximum and minimum returns within a specified lower and higher price 
range of the underlying asset, and an estimate of the strategy's probability of profit, 
expected profit and expected loss.

The probability of profit (PoP), expected profit and expected loss at the user-defined target 
date for the strategy are calculated by default using the Black-Scholes model. Alternatively,
the user can provide an array of underlying asset prices following a distribution other than 
the normal (e.g. Laplace) or model other than the Black-Scholes model (e.g. Heston model) that 
will be used in the calculations.

Despite the code having been developed with option strategies in mind, it can also be 
used for strategies that combine options with stocks and/or take into account the 
profits or losses of closed trades.

If you have any questions, corrections, comments or suggestions, just 
[drop a message](mailto:roberto.veiga@ufabc.edu.br).

You can also reach me on [Linkedin](https://www.linkedin.com/in/roberto-gomes-phd-8a718317b/) or 
follow me on [X](https://x.com/rgaveiga). When I have some free time, which is rare, I publish articles 
on [Medium](https://medium.com/@rgaveiga).

## Installation

The easiest way to install **OptionLab** is using **pip**:

```
pip install optionlab
```

## Basic usage

The evaluation of a strategy is done by calling the `run_strategy` function provided by 
the library. This function receives the input data either as a Python dictionary or an 
`Inputs` object. For example, let's say we wanted to calculate the probability of profit 
for naked calls on Apple stocks with maturity on December 17, 2021. The strategy setup 
consisted of selling 100 175.00 strike calls for 1.15 each on November 22, 2021.

```python
input_data = {
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

After defining the input data as a Python dictionary, we pass it to the `run_strategy` 
function as shown below:

```python
from optionlab import run_strategy, plot_pl

out = run_strategy(input_data)

print(out)

plot_pl(out)
```

The variable `out` is an `Outputs` object that contains the results from the 
calculations. By calling `print` with `out` as an argument, these results are 
displayed on screen. The `plot_pl` function, in turn, takes an `Outputs` object as 
its argument and plots the profit/loss diagram for the strategy.

Usage examples for a number of popular options trading strategies can be found in the 
Jupyter notebooks in the **examples** directory.

## Contributions

Contributions are definitely welcome. However, it should be mentioned that this 
repository uses [poetry](https://python-poetry.org/) as a package manager and 
[git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) with 
[pre-commit](https://pre-commit.com/) to customize actions on the repository.

## Disclaimer

This is free software and is provided as is. The author makes no guarantee that its 
results are accurate and is not responsible for any losses caused by the use of the 
code.

Options are very risky derivatives and, like any other type of financial vehicle, 
trading options requires due diligence. This code is provided for educational and 
research purposes only.

Bugs can be reported as issues.
