---
name: optionlab-black-scholes
description: Calculate theoretical call and put prices and Greeks (delta, gamma, theta, vega, rho), probability of touch, and implied volatility with OptionLab's Black-Scholes API. Use for standalone option valuation, strike comparisons, and Black-Scholes calculator code.
---

# OptionLab Black-Scholes calculations

Use `optionlab.black_scholes` to produce runnable Python for theoretical option
prices, Greeks, probability of touch, and implied volatility. Answer in the user's language. For strategy P/L and multi-leg
evaluation, use `run_strategy`; this skill addresses direct option valuation.

These instructions are based on the docstrings and implementation in
`optionlab/black_scholes.py`, the `BlackScholesInfo` model in `optionlab/models.py`,
and `examples/black_scholes_calculator.ipynb` in the repository. They are
self-contained when installed elsewhere. If the installed API differs, inspect
its signatures and docstrings before adapting the code.

## Inputs and API

Import `get_bs_info` from `optionlab.black_scholes`, not the package root.
It returns a `BlackScholesInfo` object containing both calls and puts:

```python
get_bs_info(s, x, r, vol, years_to_maturity, y=0.0)
```

| Argument | Meaning and units |
| --- | --- |
| `s` | Scalar spot price, positive. |
| `x` | Positive strike price, or a NumPy array of positive strikes. |
| `r` | Annualized risk-free rate as a decimal. |
| `vol` | Annualized volatility as a decimal. |
| `years_to_maturity` | Remaining time in years. |
| `y` | Annualized dividend yield as a decimal; defaults to zero. |

Use supplied inputs; ask for material missing data instead of inventing quotes.
Label demonstration values as illustrative. Convert percentages exactly once:
20% volatility becomes `0.20`. Keep percentage input variables separate from
decimal variables so repeated execution does not divide rates again.

The notebook uses calendar days divided by 365. If the user specifies a trading
day convention, use a matching day count and annual basis consistently for time,
volatility, and daily theta. These direct functions do not accept dates or count
holidays. Use finite inputs, positive spot/strikes, and nonnegative volatility
and maturity; the functions do not provide `Inputs` model validation.

## Calculator example

This reproduces the notebook inputs and displays all five Greeks for each type.
Values are illustrative, not live market data.

```python
from optionlab.black_scholes import get_bs_info

stock_price = 100.0
strike = 105.0
interest_rate_pct = 1.0
dividend_yield_pct = 0.0
volatility_pct = 20.0
days_to_maturity = 60
days_in_year = 365

r = interest_rate_pct / 100.0
y = dividend_yield_pct / 100.0
vol = volatility_pct / 100.0
t = days_to_maturity / days_in_year
bs = get_bs_info(s=stock_price, x=strike, r=r, vol=vol,
                 years_to_maturity=t, y=y)

for option_type in ("call", "put"):
    print(option_type.upper())
    print(f"Price: {getattr(bs, option_type + '_price'):.6f}")
    print(f"Delta: {getattr(bs, option_type + '_delta'):.6f}")
    print(f"Gamma: {bs.gamma:.6f}")
    theta = getattr(bs, option_type + "_theta")
    print(f"Theta per year: {theta:.6f}")
    print(f"Theta per day: {theta / days_in_year:.6f}")
    print(f"Vega per volatility percentage point: {bs.vega:.6f}")
    print(f"Rho per rate percentage point: {getattr(bs, option_type + '_rho'):.6f}")
```

For a strike comparison, pass `x=np.array([95.0, 100.0, 105.0])` after importing
NumPy as `np`, with the other inputs scalar. Output fields become arrays; build
rows by strike instead of applying scalar formatting to a whole array. Convert
Python lists to arrays before calling the API.

## Interpret results

| Fields | Interpretation |
| --- | --- |
| `call_price`, `put_price` | Theoretical price per option unit, in the same currency as spot and strike. |
| `call_delta`, `put_delta` | Price sensitivity to a one-unit change in spot. |
| `gamma` | Delta sensitivity to a one-unit change in spot; common to calls and puts. |
| `call_theta`, `put_theta` | Price sensitivity to elapsed time, per year. Divide by the selected annual day basis for daily theta. |
| `vega` | Price sensitivity per one percentage point of volatility; common to calls and puts. Already divided by 100. |
| `call_rho`, `put_rho` | Price sensitivity per one percentage point of the interest rate. Already divided by 100. |

These are long-option, per-unit outputs. For position Greeks, multiply by signed
option units (negative for short positions), applying the actual contract
multiplier when converting contracts to units. Keep full precision internally
and round only for display. Do not substitute the daily theta convention of
`run_strategy` for the annual theta returned here.

The object also includes `call_itm_prob`, `put_itm_prob`,
`call_prob_of_touch`, and `put_prob_of_touch`. Display fractions as percentages
only when requested. The implementation's ITM fields include `exp(-y*t)`;
with nonzero dividend yield they are dividend-discounted values, not simply
`N(d2)` and `N(-d2)`. Do not equate ITM or touch values with strategy probability
of profit. `BlackScholesInfo` is not the `Outputs` object accepted by `plot_pl`.

## Probability of touch

Use `bs.call_prob_of_touch` and `bs.put_prob_of_touch` for the notebook's
probability-of-touch outputs, or call the dedicated function for one option:

```python
from optionlab.black_scholes import get_probability_of_touch

# Uses the inputs from the calculator example.
call_touch = get_probability_of_touch("call", stock_price, strike, r, vol, t, y)
put_touch = get_probability_of_touch("put", stock_price, strike, r, vol, t, y)
print(f"Call probability of touch: {call_touch:.2%}")
print(f"Put probability of touch: {put_touch:.2%}")
```

Results are fractions. The function returns 1 for a call when spot is already
at or above strike, and for a put when spot is at or below strike. Touching the
strike during the remaining life differs from finishing ITM at expiration.
Use the library's implemented formula rather than approximating touch as twice
delta or twice ITM probability.

Despite its array type annotation, the current standalone
`get_probability_of_touch` uses scalar `if` comparisons: call it once per scalar
strike or use the vectorized touch fields from `get_bs_info` for strike arrays.
The standalone function requires positive volatility and maturity; for
deterministic cases use the handling described below for `get_bs_info`.

## Implied volatility

Import `get_implied_vol` from `optionlab.black_scholes`. It is a separate
calculation; `BlackScholesInfo` does not contain implied volatility.

`get_implied_vol(option_type, oprice, s0, x, r, years_to_maturity, y=0.0)`
takes `"call"` or `"put"`, a market premium per option unit, scalar spot and
strike, annual decimal risk-free rate, positive time in years, and optional
annual decimal dividend yield. Volatility is the output, not an input. Use
finite inputs, positive spot/strike, and a nonnegative premium. Ask for the
observed premium when the user wants market IV; do not replace it with a
theoretical price. Iterate over options for multiple premiums or strikes.

```python
from optionlab.black_scholes import get_implied_vol

# Synthetic premiums for a round-trip demonstration, not market quotes.
# Uses bs and the inputs from the calculator example.
for option_type in ("call", "put"):
    premium = float(getattr(bs, option_type + "_price"))
    implied_vol = get_implied_vol(
        option_type=option_type, oprice=premium, s0=stock_price,
        x=strike, r=r, years_to_maturity=t, y=y,
    )
    repriced = get_bs_info(stock_price, strike, r, implied_vol, t, y)
    recovered_price = float(getattr(repriced, option_type + "_price"))
    print(f"{option_type} implied volatility: {implied_vol:.2%}")
    print(f"Repricing residual: {recovered_price - premium:.8f}")
```

The returned volatility is annualized and decimal: `0.20` means 20%.
The current implementation searches only `[0.001, 1.0]` (0.1% to 100%).
It returns an endpoint when the premium is outside the prices attainable in
that interval, including invalid premiums; it does not necessarily raise an
error. Reprice with the returned volatility and compare against the supplied
premium using an appropriate price tolerance. Identify endpoint results as
potentially limited by the solver bounds rather than claiming a recovered IV.
A valid premium can still require volatility outside this interval.
Do not infer meaningful IV at expiration from this solver.

## Individual functions and boundary cases

When only selected quantities are needed, import these public functions from
`optionlab.black_scholes`. With strictly positive volatility and maturity,
compute `d1` and `d2` with the same inputs used by the pricing/Greek calls:

```python
from optionlab.black_scholes import (
    get_d1, get_d2, get_option_price, get_delta, get_gamma,
    get_theta, get_vega, get_rho,
)

# Uses the inputs defined in the calculator example above.
d1 = get_d1(stock_price, strike, r, vol, t, y)
d2 = get_d2(stock_price, strike, r, vol, t, y)
option_type = "call"  # Or "put".
price = get_option_price(option_type, stock_price, strike, r, t, d1, d2, y)
delta = get_delta(option_type, d1, t, y)
gamma = get_gamma(stock_price, vol, t, d1, y)
theta = get_theta(option_type, stock_price, strike, r, vol, t, d1, d2, y)
vega = get_vega(stock_price, t, d1, y)
rho = get_rho(option_type, strike, r, t, d2)
```

For zero volatility or zero maturity, avoid the `d1`/`d2` helpers, which divide
by zero. The current `get_bs_info` implementation handles deterministic cases
away from `s*exp(-y*t) == x*exp(-r*t)`. At that kink it raises `ValueError`
because Greeks are undefined. It has no switch to disable Greeks: for a
price-only request in this case, calculate the deterministic call price as
`max(s*exp(-y*t) - x*exp(-r*t), 0)` and the put price as the reverse difference
clipped at zero. Report undefined Greeks rather than inventing finite values.

## Verify generated calculations

When execution is available, run the relevant example and inspect the requested
fields. A useful numerical check is put-call parity:
`call_price - put_price == s*exp(-y*t) - x*exp(-r*t)`, with floating-point
tolerance. For arrays, compare each row with its scalar calculation. If using
individual functions, compare their outputs with `get_bs_info`. Report whether
the code was actually executed. Omit notebook magics such as `%%time` from scripts.
