![OptionLab](https://raw.githubusercontent.com/rgaveiga/optionlab/refs/heads/main/optionlab.png)

# OptionLab

This package is a lightweight library designed to provide quick evaluation of options trading 
strategies. It produces various outputs:

- the profit/loss profile of the strategy on a user-defined target date

- the range of stock prices for which the strategy is profitable (i.e., generating a return of 
at least \$0.01)

- the Greeks (delta, theta, rho, vega and gamma) associated with each leg of the strategy 

- the resulting debit or credit on the trading account 

- the maximum and minimum returns within a specified lower and higher price 
range of the underlying asset 

- The expected profit when the strategy is profitable and the expected loss if it proves unprofitable 

- the strategy's probability of profit.

## Contact

If you have any questions, corrections, comments or suggestions, just 
[drop a message](mailto:roberto.veiga@ufabc.edu.br).

You can also reach me on [Linkedin](https://www.linkedin.com/in/roberto-gomes-phd-8a718317b/) or 
follow me on [X](https://x.com/rgaveiga).

> [!NOTE]
> If you want to support this and other open source projects that I maintain, become a 
>[sponsor on Github](https://github.com/sponsors/rgaveiga).

## Installation

The easiest way to install **OptionLab** is using **pip**:

```
pip install optionlab
```

## Documentation

Black-Scholes profit probabilities and conditional expected returns use the
risk-neutral measure; they do not estimate real-world profit frequencies.
Success means nominal P/L >= $0.01 by default. The complementary event includes
break-even and gains below one cent. Premiums and costs are not compounded or
discounted in the P/L, and stock dividend cash flows are not included. Dividend
yield affects the ex-dividend price distribution and option valuation.

At expiration, global metrics use all strategy strikes, independently of
`min_stock`, `max_stock`, and `price_step`. These settings still determine the
displayed profile and the minimum/maximum return in that domain. Before
expiration, an internal adaptive linear approximation bounds probability error
by 1e-9 and conditional-return error by $0.0005 before rounding to cents. It uses
option curvature bounds and exact asymptotic slopes with bounded tail residuals;
no first moment is discarded. Event bounds also cover narrow regions and
tangencies. The bounds exclude floating-point roundoff. Refinement is limited
to 60 passes and 250,000 nodes; inability to establish convergence raises
`RuntimeError`, including when rare-event conditioning exhausts numerical
precision. Returned pre-expiration range boundaries are approximations subject
to the probability tolerance, not a fixed stock-price tolerance.

Zero time or volatility is handled as a deterministic terminal price. An event
with zero probability returns a conditional value of `0.0` for compatibility;
mathematically its conditional expectation is undefined. Zero-volatility pricing
and Greeks are supported away from payoff kinks. At a deterministic kink,
requesting Greeks raises `ValueError`; omit `greeks` from `calculations` to
calculate P/L and probabilities there.

The array-only `get_pop` helper assumes linear interpolation and extrapolates
the first/last secants to zero/infinity. A one-point array means a constant
profile. It cannot reconstruct missing strikes or curvature. Empirical
`ArrayInputs` probabilities and means continue to classify each sample directly.

You can access the API documentation for **OptionLab** on the [project's GitHub Pages site](https://rgaveiga.github.io/optionlab).

## Contributions

Contributions are definitely welcome. However, it should be mentioned that this 
repository uses [poetry](https://python-poetry.org/) as a package manager and 
[git hooks](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) with 
[pre-commit](https://pre-commit.com/) to customize actions on the repository. Source 
code must be formatted using [black](https://github.com/psf/black).

## Disclaimer

This is free software and is provided as is. The author makes no guarantee that its 
results are accurate and is not responsible for any losses caused by the use of the 
code.

Bugs can be reported as issues.

> [!CAUTION]
> Options are very risky derivatives and, like any other type of financial vehicle, 
> trading options requires due diligence. This code is provided for educational and 
> research purposes only.
