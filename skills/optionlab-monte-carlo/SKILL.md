---
name: optionlab-monte-carlo
description: Generate and plot underlying asset price paths with OptionLab's Monte Carlo API using risk-neutral geometric Brownian motion. Use for create_price_path_bs, plot_price_paths, reproducible path simulations, and terminal-price samples; not for direct option valuation or multi-leg strategy P/L.
---

# OptionLab Monte Carlo price paths

Produce runnable Python using `optionlab.mc`. Answer in the user's language.
These instructions describe `optionlab/mc.py` and the example in
`examples/monte_carlo_paths.ipynb`. They are self-contained when installed
elsewhere. If the installed API differs or the module is unavailable, inspect
the installed version, signatures, and docstrings before adapting the code.

## Inputs and API

Import from the module, not the package root:

```python
from optionlab.mc import create_price_path_bs, plot_price_paths

# create_price_path_bs(
#     s0, days_to_maturity, volatility, interest_rate,
#     n_intervals, n_paths, seed,
# ) -> np.ndarray
# plot_price_paths(paths) -> None
```

Use keyword arguments and supplied inputs. Ask for material missing inputs
when needed; label demonstration values as illustrative rather than market data.
All seven generator arguments are required; there are no defaults.

| Argument | Meaning and constraints |
| --- | --- |
| `s0` | Finite, positive initial underlying price. |
| `days_to_maturity` | Finite, nonnegative number of trading days; fractional days are accepted. |
| `volatility` | Finite, nonnegative annualized volatility as a decimal. |
| `interest_rate` | Finite continuously compounded annual risk-free rate as a decimal; negative rates are accepted. |
| `n_intervals` | Positive Python or NumPy integer; number of time intervals, excluding the initial point. Booleans are rejected. |
| `n_paths` | Nonnegative Python or NumPy integer; number of independent paths. Booleans are rejected. Use a positive count for summaries or useful plots. |
| `seed` | Pass an explicit nonnegative integer seed for reproducibility. The function passes it to `numpy.random.default_rng`. |

Convert percentages once: 20% becomes `0.20`. The function fixes the annual
basis at **252 trading days**: `T = days_to_maturity / 252.0` and
`dt = T / n_intervals`. Do not pass years or calendar-day counts as trading
days. There is no annual-basis, date, or holiday-calendar argument. Time-step
count is independent of day count; 2,520 intervals over 252 trading days means
ten intervals per trading day.

The model uses risk-neutral geometric Brownian motion, with log increments
`(interest_rate - 0.5 * volatility**2) * dt + volatility * sqrt(dt) * Z`.
There is no dividend-yield or historical-drift argument. Simulated outcomes
describe this model, not real-world forecasts or strategy probability of profit.

## Notebook example

These illustrative inputs reproduce the notebook's simulation and plot:

```python
import matplotlib.pyplot as plt
import numpy as np

from optionlab.mc import create_price_path_bs, plot_price_paths

s0 = 100.0
volatility = 0.2
interest_rate = 0.05
days_to_maturity = 252.0
n_intervals = 2520
n_paths = 1000
seed = 42

paths = create_price_path_bs(
    s0=s0,
    days_to_maturity=days_to_maturity,
    volatility=volatility,
    interest_rate=interest_rate,
    n_intervals=n_intervals,
    n_paths=n_paths,
    seed=seed,
)

assert paths.shape == (1000, 2521)
np.testing.assert_array_equal(paths[:, 0], s0)
plot_price_paths(paths)
plt.show()
```

## Interpret and reuse paths

The return value is a `float64` NumPy array with shape
`(n_paths, n_intervals + 1)`. Each row is one complete path; each column is
one time point, including the initial price at column zero. Extract terminal
prices with `paths[:, -1]`, not `paths[-1]`. A terminal-price summary using
the example above is:

```python
terminal_prices = paths[:, -1]
print(f"Mean terminal price: {terminal_prices.mean():.4f}")
print("Terminal price quantiles:", np.quantile(terminal_prices, [0.05, 0.5, 0.95]))
trading_day_grid = np.linspace(0.0, days_to_maturity, n_intervals + 1)
```

Terminal prices are underlying prices, not option premiums, discounted payoffs,
or strategy P/L. If another API needs terminal-price samples, pass the terminal
column after checking that API's input contract rather than the full path matrix.

Zero maturity produces constant paths at `s0`; zero volatility produces the
deterministic path `s0 * exp(interest_rate * t)` on the year grid. Zero paths
returns an empty array with shape `(0, n_intervals + 1)`; do not compute
quantiles or means on it.

Each call creates a local NumPy generator. Repeating all inputs and seed in
the same environment reproduces the array without resetting global Python or
NumPy random state. Changing the grid or path count is not a guarantee of
preserving the same simulated trajectories. Memory for the returned array is
approximately `8 * n_paths * (n_intervals + 1)` bytes; choose counts that fit
the user's environment.

## Plot and export

`plot_price_paths(paths)` creates a new Matplotlib figure and axes, plots
`paths.T`, and returns `None`. Do not transpose the input yourself or create a
separate figure first. It labels the horizontal axis "Monte Carlo steps" and
the vertical axis "Asset prices ($)"; it does not call `show()` or save a file.
For a different currency or trading-day axis, adapt the axes or plot against
`trading_day_grid` directly.

For a requested image export, call `plt.savefig(user_selected_path)` after
`plot_price_paths(paths)` and before `plt.show()`. In a headless process select
Matplotlib's `Agg` backend before importing pyplot or OptionLab, and close the
figure after saving. For large simulations, plot an explicitly labeled subset
such as `plot_price_paths(paths[:100])` while retaining all paths for statistics.

## Verify generated code

When execution is available, run the relevant example and check shape, dtype,
initial prices, and finite positive values for ordinary parameter values. Check
repeatability with the same seed, or verify zero-volatility paths against
`s0 * np.exp(interest_rate * np.linspace(0, days_to_maturity / 252, n_intervals + 1))`
with floating-point tolerance. Stochastic sample means need sampling tolerance;
they will not equal `s0 * exp(interest_rate * T)` exactly. For a headless plot,
check that a saved image is nonempty. Report whether execution succeeded and
omit notebook magics from ordinary Python scripts.
