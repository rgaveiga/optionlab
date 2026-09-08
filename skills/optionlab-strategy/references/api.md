# OptionLab API reference

Contents: input fields; legs and prior positions; dates; models and calculations;
outputs; plotting. Names and constraints below come from the repository's Python
models and docstrings, with engine behavior noted where it is more specific.

## Entry points

```python
from optionlab import Inputs, run_strategy, plot_pl
from optionlab.models import Outputs, Stock, Option, ClosedPosition
```

`run_strategy(inputs_data: Inputs | dict) -> Outputs` validates dictionaries via
`Inputs.model_validate`. `Inputs(**payload)` is equivalent for construction.
Nested legs may be dictionaries or the corresponding Pydantic models. Use the
exact documented names: unknown keys can be silently ignored by Pydantic, so a
misspelled optional field may leave its default active.

## All Inputs fields

| Field | Type; default | Meaning and constraints |
| --- | --- | --- |
| `stock_price` | float; required | Spot price, > 0. |
| `volatility` | float; required | Annualized underlying volatility, >= 0; decimal fraction. |
| `interest_rate` | float; required | Annualized risk-free rate, >= 0; decimal fraction. |
| `min_stock` | float; required | Price-domain lower bound, >= 0. |
| `max_stock` | float; required | Price-domain upper bound, >= 0; must be strictly greater than `min_stock` at execution. |
| `price_step` | float; `0.01` | Positive grid spacing; controls plot resolution and domain extrema. Endpoint rounding may put the last sample slightly beyond the requested bound. |
| `strategy` | list of legs; required | At least one leg; at most one `closed` leg. All legs share the underlying and market inputs. |
| `dividend_yield` | float; `0.0` | Annualized yield as a decimal, >= 0. The Black-Scholes probability model additionally requires <= 1. |
| `profit_target` | float or None; `None` | Total strategy profit threshold; its probability/ranges run only with `pop` and a value > 0.01. |
| `loss_limit` | float or None; `None` | Total strategy loss threshold; supply a negative amount, e.g. -100. Runs only with `pop` and a value < 0. |
| `opt_commission` | float; `0.0` | Fixed amount subtracted once per active option leg, not multiplied by `n`. No sign constraint in the schema. |
| `stock_commission` | float; `0.0` | Fixed amount subtracted once per active stock leg, not multiplied by `n`. No sign constraint in the schema. |
| `discard_nonbusiness_days` | bool; `True` | Exclude weekends/holidays in date-based horizons; selects business-day year basis. |
| `business_days_in_year` | int; `252` | Year denominator when discarding nonbusiness days. Supply a positive count; the schema itself does not impose positivity. |
| `country` | str; `"US"` | Holiday country accepted by the `holidays` library for date-based counting. |
| `start_date` | date or None; `None` | Start date; use together with `target_date`. ISO `YYYY-MM-DD` strings are accepted by Pydantic. |
| `target_date` | date or None; `None` | Evaluation date, strictly after `start_date`. |
| `days_to_target_date` | int; `0` | Explicit horizon when not using dates; must be > 0 to satisfy the combined date validator, despite the field allowing zero. |
| `model` | `"black-scholes"` or `"array"`; `"black-scholes"` | Distribution used for strategy probabilities and expectations. |
| `array` | numpy.ndarray; empty array | Terminal underlying-price samples for `model="array"`; must be nonempty in that mode. |
| `calculations` | list of strings; `["pop", "expectation", "impvol", "greeks"]` | Allowed names: `pop`, `PoP`, `expectation`, `impvol`, `greeks`. An empty list is valid. |

`min_stock`, `max_stock`, volatility and interest rate are required even for array
mode or profile-only calculations. Choose a useful plot window around the spot,
strikes and relevant break-even levels without treating its extremes as risk bounds.

## Every leg field

| Leg | Required fields | Optional fields |
| --- | --- | --- |
| Stock | `n`: positive int shares; `action`: `"buy"` or `"sell"` | `type="stock"`; `prev_pos`: None or float |
| Option | `type`: `"call"` or `"put"`; `strike`: positive float; `premium`: positive float; `action`: `"buy"` or `"sell"`; `n`: positive int options | `prev_pos`: None or float; `expiration`: None, date, or positive integer |
| ClosedPosition | `prev_pos`: float total realized P/L | `type="closed"` |

Include `type` explicitly in dictionary legs to make their intent unambiguous.
Premiums and strikes are per unit. There is no `contracts`, `multiplier`, `ticker`,
`option_price`, `days_to_maturity`, or per-leg `volatility` input field. Do not use
negative `n` for shorts; set `action="sell"`. Zero premiums are rejected.

### Previous positions

For a stock/option leg, `prev_pos` is a **per-unit historical price**, not P/L:

- `None` or `0`: use the current spot/premium as the entry basis.
- Positive: previously opened and still held; use this basis for P/L. Keep the
  current `premium` on option legs for implied-volatility calculation.
- Negative: that position is now closed. Use the negative historical entry price,
  retain the original `action`, and supply the current closing spot/premium.
  The current engine contributes `(abs(prev_pos) - current_price) * n` for `buy`,
  with the opposite sign for `sell`. This is the implemented cash-flow convention,
  which reverses the usual realized trade P/L sign. If the user supplies a realized
  gain/loss total, represent it with `type="closed"` to preserve that intended sign.

For `type="closed"`, `prev_pos` instead means **total realized P/L**: positive for
profit and negative for loss. Combine multiple realized amounts into one such leg.
The engine does not apply commissions to these closed branches; incorporate any
required realized fees in the supplied total rather than assuming another charge.

## Dates and expiration

Choose one coherent horizon representation:

- Dates: set both `start_date` and `target_date`, with start < target. Each explicit
  date expiration must be on or after target. Counting uses calendar difference
  plus one, then removes nonbusiness days from the start-inclusive, target-exclusive
  interval when enabled. Thus the target day contributes one day even if nonbusiness.
- Day count: omit both dates and set `days_to_target_date > 0`. The count is used
  directly: business-day units with `discard_nonbusiness_days=True`, calendar-day
  units with `False`. The respective year denominator is `business_days_in_year`
  or 365. The engine does not remove holidays again from an explicit count.

If both dates and a day count are supplied, the dates take precedence; avoid the
redundancy. In count mode, an option expiration cannot be a date. In either mode,
an integer expiration counts from the same start as the horizon, uses the same
units, and must be >= the computed days to target (checked by the engine).

Omitted/None expiration means expiration at the target. Equal expiration uses
intrinsic payoff. Later expiration uses Black-Scholes value for the remaining
time at the target; this is how to construct calendar spreads. Expirations before
target are invalid, including on legs represented with negative `prev_pos`.

## Models and calculation selection

`black-scholes` uses risk-neutral lognormal terminal prices, drift determined by
interest and dividend rates, and the shared underlying volatility. Strategy PoP
and conditional means integrate the complete profile over nonnegative prices;
changing the plotting range or `price_step` does not truncate their tails. Before
expiration, numerical interpolation targets a conditional-return error of 0.0005
and probability error of 1e-9; nonconvergence raises a runtime error.

Zero volatility is supported as a deterministic terminal price. At a deterministic
option payoff kink, requesting `greeks` can raise `ValueError`; select only the
calculations needed rather than perturbing the supplied market data.

`array` uses **terminal stock prices**, not option prices, log returns, a price
history, or strategy P/L. Convert lists with `np.asarray(samples, dtype=float)`;
`Inputs.array` does not have the list conversion that the lower-level `ArrayInputs`
model has. Supply a finite, one-dimensional, nonempty array of nonnegative prices.
Each sample has equal weight; repetitions represent empirical frequency. Samples
may extend beyond the plot window. PoP/expectations use sample P/L, while the
plotted profile still uses the regular price grid. Pre-expiry option valuation,
implied volatility and Greeks still use Black-Scholes. Array-mode ranges describe
the grid profile and are limited by that grid; they are not sample quantiles.

`"laplace"` is not an `Inputs.model` choice. The separate `LaplaceInputs` is for
lower-level helpers, not a `run_strategy` input. If a user supplies samples from
another distribution, pass those terminal prices through `model="array"`.

| Calculation | What it populates |
| --- | --- |
| `pop` or `PoP` | PoP and profit ranges; target/limit probabilities and ranges when thresholds meet the conditions above. |
| `expectation` | Both conditional expected-return fields, independently of `pop`. |
| `impvol` | Per-leg implied volatility from current option premiums. |
| `greeks` | Delta, gamma, theta, vega, rho, ITM probability and probability of touch. |
| Always computed | P/L profile, per-leg and total cost, domain minimum and maximum. |

## All Outputs fields

`run_strategy` returns an `Outputs` Pydantic object, never a plain dictionary.
Lists associated with legs follow `inputs.strategy` order when calculated.

| Field | Type | Interpretation |
| --- | --- | --- |
| `probability_of_profit` | float | Probability of total P/L >= 0.01. Fraction, not percent. |
| `profit_ranges` | list of (float, float) | Underlying-price intervals with P/L >= 0.01; may be disconnected or have infinite bounds in Black-Scholes mode. |
| `expected_profit_if_profitable` | float | Conditional mean P/L given P/L >= 0.01. |
| `expected_loss_if_unprofitable` | float | Conditional mean P/L given P/L < 0.01; can include zero or a tiny positive gain below one cent. |
| `per_leg_cost` | list of float | Signed leg cash flows, including applicable commissions and realized contributions. |
| `strategy_cost` | float | Sum of leg costs; negative debit, positive credit. |
| `minimum_return_in_the_domain` | float | Minimum P/L sampled on the price grid; not necessarily global maximum loss. |
| `maximum_return_in_the_domain` | float | Maximum P/L sampled on the price grid; not necessarily global maximum profit. |
| `implied_volatility` | list of float | Per-leg implied volatility, in decimal annualized units; zero placeholders for stock/closed legs. |
| `in_the_money_probability` | list of float | Per-leg probability of expiring ITM. |
| `probability_of_touch` | list of float | Per-leg probability of touching the ITM region before expiration. |
| `delta` | list of float | Per-unit delta, with buy/sell sign applied. |
| `gamma` | list of float | Per-unit gamma; current engine does not invert its sign for short options. |
| `theta` | list of float | Per-unit theta per day under the selected year basis, with buy/sell sign applied. |
| `vega` | list of float | Per-unit sensitivity per one percentage-point volatility change; current engine does not invert its sign for shorts. |
| `rho` | list of float | Per-unit sensitivity per one percentage-point interest-rate change, with buy/sell sign applied. |
| `probability_of_profit_target` | float | Probability of P/L >= supplied valid profit target; otherwise default 0.0. |
| `profit_target_ranges` | list of (float, float) | Price intervals meeting that profit target; otherwise empty. |
| `probability_of_loss_limit` | float | Probability below `loss_limit + 0.01`, the engine's cent-offset convention for losing at least the limit; otherwise default 0.0. |
| `loss_limit_ranges` | list of (float, float) | Price intervals for that loss event; otherwise empty. |
| `inputs` | Inputs | Original validated input; marked private in the docstrings, retained for plotting. |
| `data` | EngineDataResults | Internal profile arrays and bookkeeping; marked private, retained for plotting. |

Expected returns are nominal currency amounts, not percentages or discounted
option prices. They are not rounded to cents. Empty conditioning events return
0.0. Skipped probability/expectation calculations remain 0.0 with empty ranges;
skipped `impvol`/`greeks` produce empty lists. Do not interpret those defaults as
computed results. Greeks are not multiplied by `n`; do not sum the lists as total
portfolio Greeks without accounting for quantities and the sign conventions.

`print(out)` displays nondefault public fields. For a dictionary, use
`out.model_dump(exclude={"inputs", "data"})`; avoid `exclude_defaults=True` when a
stable set of keys is needed. Internal NumPy arrays need explicit conversion if
serializing the full object to JSON.

## Plotting and profile access

`plot_pl(outputs: Outputs) -> None` draws the strategy P/L at the target on current
Matplotlib axes. The plot uses the price domain, marks spot and option strikes,
and includes supplied target/limit lines. It prints a legend explanation but does
not call `plt.show()`. It raises `RuntimeError` for an empty strategy profile.

Use `get_pl(out)` to retrieve `(stock_prices, strategy_pl)`, or `get_pl(out, leg=0)`
for a specific valid zero-based leg. `pl_to_csv(out, filename="pl.csv", leg=None)`
saves these arrays. Both are exported from `optionlab`. Prefer these helpers for
custom plots over depending on internal arrays; the notebook equivalents are
`out.data.stock_price_array`, `out.data.strategy_profit`, and `out.data.profit[i]`.
