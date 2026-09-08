---
name: optionlab-strategy
description: Build and validate OptionLab run_strategy inputs as Python dictionaries or Inputs objects, interpret Outputs, and plot strategy profit/loss with plot_pl. Use for coding OptionLab options and stock strategies, including prior positions, different expirations, and terminal-price samples.
---

# OptionLab strategy construction

Create runnable Python for the user's strategy using `run_strategy(inputs_data)`.
It accepts an `Inputs` object or a Python dictionary and returns `Outputs` in both
cases. Prefer the user's requested representation. Answer in the user's language.

This skill describes the repository's OptionLab 1.9.0 API, based on the docstrings
and validation in `optionlab/models.py`, `engine.py`, `support.py`, `plot.py`,
`utils.py`, and the notebooks in `examples/`. If the installed version differs,
check its model fields and docstrings before adapting version-sensitive options.
The bundled references are self-contained when installed outside this repository.

## Build the input

1. Read [the API reference](references/api.md) for field names, defaults, units,
   date rules, calculation selection, and output semantics. It covers every
   `Inputs` field and each strategy-leg type.
2. Establish the underlying price, annualized volatility and interest rate,
   evaluation horizon, price domain, and each leg's side, quantity, strike and
   premium where applicable. Use supplied data; identify illustrative values as
   examples. Ask for material missing trade data rather than inventing quotes.
3. Choose dates or an explicit day count. Ensure no option expires before the
   evaluation horizon. For later expirations, OptionLab values the remaining
   option at the target date using Black-Scholes.
4. Express rates as decimals and quantities as units: `0.20` means 20%; `n=100`
   means 100 options, with no automatic contract multiplier. Translate contracts
   to units using the actual contract specification.
5. Select calculations for the requested outputs. Use `calculations=[]` for only
   the P/L profile and costs, `["pop"]` for probability screening, or omit the
   field for all calculations. Required market fields remain required in every mode.
6. Validate with `Inputs.model_validate(payload)` or construct `Inputs(...)`, then
   call `run_strategy`. Validation of a model alone does not cover every runtime
   condition, such as price-domain ordering or integer expirations.

Use [the examples](references/examples.md) for a complete dictionary/object pair,
calendar spreads, previous positions, puts, explicit day counts, sample arrays,
and plot/export code. Adapt the smallest relevant example; do not execute the
entire example catalog for an ordinary user request.

## Interpret and present

Read the `Outputs` table in [the API reference](references/api.md) before reporting
results. Probabilities are fractions; monetary outputs are total strategy amounts.
Costs are signed cash flows (debit negative, credit positive). Domain extrema are
limited to the sampled plotting domain; they do not establish a global maximum
loss or profit. Black-Scholes PoP and conditional expectations cover the full
terminal-price domain, including tails outside the plotted interval.

Keep full floating-point values in calculations; format to two decimals only for
display. Expected profit and loss are conditional means, not an unconditional
expected return. Disabled calculations retain defaults, so zero or an empty list
can mean "not calculated". Use `pytest.approx` for numerical example checks.

## Plot

```python
from optionlab import plot_pl
import matplotlib.pyplot as plt

# out is the complete Outputs returned by run_strategy.
plt.figure()
plot_pl(out)
plt.show()
```

`plot_pl(out)` draws on Matplotlib's current axes, prints a diagram explanation,
and returns `None`; it does not call `show()` or return a Figure. To save, use
`plt.savefig(...)` before `plt.show()`. In a headless process select Matplotlib's
`Agg` backend before importing OptionLab or pyplot. Keep `out.inputs` and
`out.data` intact because plotting needs them.

## Check the generated code

When execution is available, run the chosen example in the user's Python
environment, verify it returns `Outputs`, and inspect the requested fields. For
plots, verify a nonempty saved image when running headlessly. Report actual
execution results; otherwise state that the code was not run. Do not add notebook
magics such as `%matplotlib inline` or `%%time` to ordinary Python scripts.
