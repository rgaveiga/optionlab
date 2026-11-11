---
name: optionlab-option-chain
description: Create synthetic OptionLab option chains with user-defined strike bounds and strike count, returning NumPy arrays of Black-Scholes prices and Greeks. Use for generating, tabulating, or exporting theoretical option chains; not for fetching live market chains or evaluating multi-leg strategy P/L.
---

# OptionLab option chains

Produce runnable Python using `create_bs_option_chain`. Answer in the user's
language. This skill describes the OptionLab 1.9.7 implementation in
`optionlab/option_chain.py` and is self-contained when installed elsewhere.
If the installed API differs, inspect its signature and docstring before adapting.

## Inputs and API

Import from the module, not the package root:

```python
from optionlab.option_chain import create_bs_option_chain

# Signature: create_bs_option_chain(
#     s0, min_strike, max_strike, vol, r, years_to_maturity, n, y=0.0
# )
```

Use keyword arguments to avoid confusing the positions of volatility and rate.
Use supplied values; ask for material missing inputs. Label demonstration values
as illustrative, rather than presenting them as market quotes.

| Argument | Meaning and constraints |
| --- | --- |
| `s0` | Finite, positive scalar underlying price. |
| `min_strike`, `max_strike` | Finite positive bounds, with `min_strike < max_strike`. Both are included. |
| `n` | Python integer at least 2, excluding booleans; total strike count including both bounds, not an interval count or contract quantity. |
| `vol` | Annualized volatility as a decimal, finite and nonnegative. |
| `r` | Annualized risk-free rate as a finite decimal. |
| `years_to_maturity` | Finite, nonnegative time remaining in years. |
| `y` | Annualized dividend yield as a finite decimal; defaults to zero. |

Convert percentage inputs once: 20% becomes `0.20`. Convert a supplied day count
using the user's annual basis; the example below uses calendar days divided by
365. The function does not accept dates or count holidays. Ensure finite rates
and nonnegative finite time in generated code; the library does not validate all
of these inputs itself.

The function generates `n` equally spaced strikes with `numpy.linspace`, with
spacing `(max_strike - min_strike) / (n - 1)`. It does not round strikes to cents
or align them with exchange-listed strike increments. Preserve full precision
for calculations and round only for display. It uses one scalar volatility and
one maturity for the whole chain; it does not model a volatility smile. For
multiple maturities, create a separate chain per maturity. For explicitly chosen
irregular strikes, use `get_bs_info` directly instead of changing this contract.

## Runnable example

These inputs are illustrative. The DataFrame is an optional presentation layer;
`chain` remains the original dictionary of arrays.

```python
import numpy as np
import pandas as pd

from optionlab.option_chain import create_bs_option_chain

chain = create_bs_option_chain(
    s0=100.0,
    min_strike=80.0,
    max_strike=120.0,
    vol=0.20,
    r=0.05,
    years_to_maturity=60 / 365,
    n=9,
    y=0.02,
)

table = pd.DataFrame({
    "strike": chain["strikes"],
    "call_price": chain["calls"]["price"],
    "call_delta": chain["calls"]["delta"],
    "call_theta": chain["calls"]["theta"],
    "put_price": chain["puts"]["price"],
    "put_delta": chain["puts"]["delta"],
    "put_theta": chain["puts"]["theta"],
    "gamma": chain["gamma"],
    "vega": chain["vega"],
})
print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

np.testing.assert_allclose(chain["strikes"], [80, 85, 90, 95, 100, 105, 110, 115, 120])
np.testing.assert_allclose(
    chain["calls"]["price"] - chain["puts"]["price"],
    100.0 * np.exp(-0.02 * 60 / 365)
    - chain["strikes"] * np.exp(-0.05 * 60 / 365),
)
```

For a requested CSV export, use `table.to_csv(user_selected_path, index=False)`.
Keep calculations at full precision. For JSON, explicitly convert each array
to a list at the serialization boundary; do not change the function's return type.

## Output and interpretation

The dictionary has exactly these fields; every numeric leaf is a NumPy array of
shape `(n,)`, with index `i` corresponding to `chain["strikes"][i]`:

| Dictionary path | Meaning |
| --- | --- |
| `strikes` | Increasing strike grid including the supplied bounds. |
| `calls.price`, `puts.price` | Theoretical prices per option unit. |
| `calls.delta`, `puts.delta` | Price sensitivity per unit change in spot. |
| `calls.theta`, `puts.theta` | Theta per year; divide by the selected annual day basis for daily theta. |
| `gamma` | Gamma common to calls and puts. |
| `vega` | Vega common to calls and puts, per one volatility percentage point; already divided by 100. |

The dotted paths above denote nested dictionary keys, not object attributes.
These are long-option, per-unit values without contract multipliers. Apply signed
units only when the user asks for position exposures. Rho, ITM probabilities,
touch probabilities, and implied volatility are not included in this dictionary.
For additional Black-Scholes fields, call `get_bs_info` with the same strikes and
parameters and label the separate results. Do not pass this dictionary to `plot_pl`,
which expects strategy `Outputs`.

The chain delegates prices and Greeks to `get_bs_info`. When volatility or time
is zero, a strike at `s0*exp(-y*t) == strike*exp(-r*t)` causes the entire call to
raise `ValueError` because Greeks are undefined there. Report this limitation;
do not silently remove the strike, perturb the inputs, or replace Greeks with zero.

## Verify generated code

When execution is available, run the relevant example and check count, inclusive
bounds, spacing, array shapes, and put-call parity with floating-point tolerance.
When adapting the output mapping, compare individual rows against `get_bs_info`
for the same scalar strike and parameters. Report whether execution succeeded.
