"""
## OptionLab is...

... a Python library designed as a research tool for quickly evaluating options 
strategy ideas. It is intended for a wide range of users, from individuals learning 
about options trading to developers of quantitative strategies.

**OptionLab** calculations can produce a number of useful outputs:

- the profit/loss profile of the strategy on a user-defined target date

- the range of stock prices for which the strategy is profitable

- the Greeks associated with each leg of the strategy

- the resulting debit or credit on the trading account

- the maximum and minimum returns within a specified lower and higher price range 
of the underlying asset 

- the expected profit and expected loss of the strategy

- the probability of profit

The probability of profit (PoP) of the strategy on the user-defined target date 
is calculated analytically by using the Black-Scholes model. Alternatively, 
the user can provide an array of terminal underlying asset prices obtained from 
other sources (e.g., the Heston model, a Laplace distribution, or a Machine 
Learning/Deep Learning model) to be used in the calculations instead of the 
Black-Scholes model. This allows **OptionLab** to function as a calculator that 
supports a variety of pricing models.

An advanced feature of **OptionLab** that provides great flexibility in building 
complex dynamic strategies is the ability to include previously created positions 
as legs in a new strategy. Popular strategies that can benefit from this feature 
include the Wheel and Covered Call strategies.

## OptionLab is not...

... a platform for direct order execution. This capability has not been and 
probably will not be implemented.

Backtesting and trade simulation using Monte Carlo have also not (yet) been 
implemented in the API.

That being said, nothing prevents **OptionLab** from being integrated into an 
options quant trader's workflow alongside other tools.

## Installation

The easiest way to install **OptionLab** is using **pip**:

```
pip install optionlab
```

## Quickstart

**OptionLab** is designed with ease of use in mind. An options strategy can be 
defined and evaluated with just a few lines of Python code. The API is streamlined, 
and the learning curve is minimal.

The evaluation of a strategy is done by calling the `optionlab.engine.run_strategy` 
function provided by the library. This function receives the input data either 
as a dictionary or an `optionlab.models.Inputs` object.

For example, let's say we wanted to calculate the probability of profit for naked 
calls on Apple stocks expiring on December 17, 2021. The strategy setup consisted 
of selling 100 175.00 strike calls for 1.15 each on November 22, 2021.

The input data for this strategy can be provided in a dictionary as follows:

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

Alternatively, the input data could be defined as the `optionlab.models.Inputs` 
object below:
    
```python
from optionlab import Inputs

input_data = Inputs(
    stock_price = 164.04,
    start_date = "2021-11-22",
    target_date = "2021-12-17",
    volatility = 0.272,
    interest_rate = 0.0002,
    min_stock = 120,
    max_stock = 200,
    strategy = [
        {
            "type": "call",
            "strike": 175.0,
            "premium": 1.15,
            "n": 100,
            "action":"sell"
        }
    ],
)
```

In both cases, the strategy itself is a list of dictionaries, where each dictionary
defines a leg in the strategy. The fields in a leg, depending on the type of the
leg, are described in `optionlab.models.Stock`, `optionlab.models.Option`, and
`optionlab.models.ClosedPosition`.

After defining the input data, we pass it to the `run_strategy` function as shown 
below:

```python
from optionlab import run_strategy, plot_pl

out = run_strategy(input_data)

print(out)

plot_pl(out)
```

The variable `out` is an `optionlab.models.Outputs` object that contains the 
results from the calculations. By calling `print` with `out` as an argument, 
these results are displayed on screen. 

The `optionlab.plot.plot_pl` function, in turn, takes an `optionlab.models.Outputs` 
object as its argument and plots the profit/loss diagram for the strategy.

## Examples

Examples for a number of popular options trading strategies can be found as 
Jupyter notebooks in the [examples](https://github.com/rgaveiga/optionlab/tree/main/examples) 
directory.
"""

from .models import Inputs
from .engine import run_strategy
from .plot import plot_pl
from .utils import get_pl, pl_to_csv

__docformat__ = "markdown"
__version__ = "1.5.1"
