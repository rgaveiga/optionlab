# Runnable strategy examples

The historical values below are examples from the repository, not current quotes.
Each block is self-contained except the final plotting block, which uses `out`
from any preceding strategy block. Plain Python replaces notebook magics.

## Dictionary and Inputs: covered call

Adapted from `examples/covered_call.ipynb`. Both representations use the same
schema and produce `Outputs`. `n=100` is 100 shares/options, not 100 contracts.

```python
from optionlab import Inputs, run_strategy
from optionlab.models import Outputs

payload = {
    "stock_price": 164.04,
    "volatility": 0.272,
    "interest_rate": 0.0002,
    "start_date": "2021-11-22",
    "target_date": "2021-12-17",
    "min_stock": 82.02,
    "max_stock": 246.06,
    "strategy": [
        {"type": "stock", "n": 100, "action": "buy"},
        {"type": "call", "strike": 175.0, "premium": 1.15,
         "n": 100, "action": "sell"},
    ],
}
out = run_strategy(payload)
inputs = Inputs(**payload)  # Also: Inputs.model_validate(payload)
out_from_model = run_strategy(inputs)
assert isinstance(out, Outputs) and isinstance(out_from_model, Outputs)
print(f"PoP: {out.probability_of_profit:.2%}")
print(out.model_dump(exclude={"inputs", "data"}))
```

## Inputs with typed legs: naked call, thresholds, commissions and day count

Adapted from `examples/naked_call.ipynb`; uses an explicit illustrative 30-calendar-day
horizon instead of the notebook's dates. Other optional fields are shown explicitly.

```python
from optionlab import Inputs, run_strategy
from optionlab.models import Option

inputs = Inputs(
    stock_price=164.04,
    volatility=0.272,
    interest_rate=0.0002,
    dividend_yield=0.01,
    min_stock=82.02,
    max_stock=246.06,
    price_step=0.10,
    days_to_target_date=30,
    discard_nonbusiness_days=False,
    business_days_in_year=252,  # Used only with discard_nonbusiness_days=True.
    country="US",  # Holidays are used only for date-based business-day counts.
    opt_commission=1.0,
    stock_commission=0.0,
    profit_target=100.0,
    loss_limit=-100.0,
    model="black-scholes",
    calculations=["pop", "expectation", "impvol", "greeks"],
    strategy=[Option(type="call", strike=175.0, premium=1.15,
                     n=100, action="sell", expiration=30)],
)
out = run_strategy(inputs)
print(out.probability_of_profit_target, out.probability_of_loss_limit)
```

## Calendar spread: later expiration

Adapted from `examples/calendar_spread.ipynb`. The short expires at target; the
long is valued with its remaining time. For count mode, replace dates by positive
day counts from the same start, not days measured from the target.

```python
import datetime as dt
from optionlab import run_strategy

payload = {
    "stock_price": 127.14,
    "volatility": 0.427,
    "interest_rate": 0.0009,
    "start_date": dt.date(2021, 1, 18),
    "target_date": dt.date(2021, 1, 29),
    "min_stock": 63.57,
    "max_stock": 190.71,
    "strategy": [
        {"type": "call", "strike": 127.0, "premium": 4.60,
         "n": 1000, "action": "sell"},
        {"type": "call", "strike": 127.0, "premium": 5.90,
         "n": 1000, "action": "buy", "expiration": dt.date(2021, 2, 12)},
    ],
}
out = run_strategy(payload)
print(out)
```

## Previous positions and realized P/L

The first strategy comes from `examples/nonsimultaneous_call_spread.ipynb`.
The alternative illustrates the two different meanings of `prev_pos`. Negative
per-unit `prev_pos` uses the engine's cash-flow convention; to record a known
realized gain of 515 instead, use `{"type": "closed", "prev_pos": 515.0}` and
combine it with any other realized totals in that one closed leg.

```python
from optionlab import run_strategy

base = dict(stock_price=168.99, volatility=0.483, interest_rate=0.045,
            start_date="2023-01-16", target_date="2023-02-17",
            min_stock=68.99, max_stock=268.99)
held_legs = [
    {"type": "call", "strike": 165.0, "premium": 12.65, "n": 100,
     "action": "buy", "prev_pos": 7.5},  # Still held; historical entry premium.
    {"type": "call", "strike": 170.0, "premium": 9.9, "n": 100,
     "action": "sell"},
]
out = run_strategy(base | {"strategy": held_legs})

closed_legs = [
    {"type": "call", "strike": 165.0, "premium": 12.65, "n": 100,
     "action": "buy", "prev_pos": -7.5},  # Engine convention contributes -515; see api.md.
    {"type": "closed", "prev_pos": -50.0},  # Another realized total loss.
    {"type": "stock", "n": 100, "action": "buy", "prev_pos": 158.99},
]
out_closed = run_strategy(base | {"strategy": closed_legs})
```

## Calls and puts: short straddle with probability-only screening

Adapted from `examples/short_straddle.ipynb`. Calculation selection follows
`examples/call_spread.ipynb`, which uses `["pop"]` while screening combinations.

```python
from optionlab import Inputs, run_strategy

inputs = Inputs(
    stock_price=168.99, volatility=0.483, interest_rate=0.045,
    start_date="2023-01-16", target_date="2023-02-17",
    min_stock=84.50, max_stock=253.48,
    calculations=["PoP"],  # Alias for "pop".
    strategy=[
        {"type": "call", "strike": 170.0, "premium": 9.9,
         "n": 100, "action": "sell"},
        {"type": "put", "strike": 170.0, "premium": 10.2,
         "n": 100, "action": "sell"},
    ],
)
out = run_strategy(inputs)
print(out.probability_of_profit)
# Expectations remain zero and Greeks/IV remain empty because they were skipped.
```

## Array model and profile-only mode

Based on `Inputs` and engine docstrings; the notebooks do not illustrate array
mode. The small deterministic array demonstrates the interface, not a calibrated
distribution. Replace it with terminal underlying-price samples for the chosen horizon.

```python
import numpy as np
from optionlab import run_strategy

payload = dict(
    stock_price=100.0, volatility=0.2, interest_rate=0.05,
    min_stock=50.0, max_stock=150.0, price_step=0.5,
    days_to_target_date=30, discard_nonbusiness_days=False,
    model="array", array=np.asarray([80, 95, 100, 105, 120], dtype=float),
    calculations=["pop", "expectation"],
    strategy=[{"type": "put", "strike": 100.0, "premium": 5.0,
               "n": 100, "action": "buy"}],
)
out = run_strategy(payload)
print(out.probability_of_profit, out.expected_profit_if_profitable)
profile_only = run_strategy(payload | {"calculations": []})
# profile_only still contains the grid P/L, costs and domain extrema for plotting.
```

## Plot, save, and export the profile

Uses `out` from one of the preceding blocks. Notebook plotting examples are in
`calendar_spread.ipynb` and `naked_call.ipynb`; per-leg plots are in
`covered_call.ipynb` and `short_straddle.ipynb`. Saving uses Matplotlib directly.
For headless execution, call `matplotlib.use("Agg")` before the first OptionLab
or pyplot import in the process, including imports in preceding blocks.

```python
import matplotlib.pyplot as plt
from optionlab import plot_pl, get_pl, pl_to_csv

fig = plt.figure()
plot_pl(out)
fig.savefig("strategy-pl.png", dpi=150)
plt.show()  # Optional in notebooks; omit in headless scripts.
plt.close(fig)
stock_prices, strategy_pl = get_pl(out)
stock_prices, first_leg_pl = get_pl(out, leg=0)
pl_to_csv(out, filename="strategy-pl.csv")
```

`examples/black_scholes_calculator.ipynb` illustrates separate option pricing
with `get_bs_info`, annualized decimal inputs, and probability display as
percentages. Its `BlackScholesInfo` result is not the `Outputs` accepted by `plot_pl`.
