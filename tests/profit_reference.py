"""Independent scalar BS pricing, root solving and normal-space quadrature.

Fixtures here have ordinary tails: truncation at 14 standard deviations loses
less than 1e-35 of their first moments. Production profit helpers are not used.
"""

import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm


def reference(outputs, target=0.01):
    inputs, data = outputs.inputs, outputs.data
    time = data.days_to_target / data.days_in_year
    v = inputs.volatility * math.sqrt(time)
    m = (
        math.log(inputs.stock_price)
        + (inputs.interest_rate - inputs.dividend_yield - inputs.volatility**2 / 2)
        * time
    )

    def payoff(s):
        total = 0.0
        for i, leg in enumerate(inputs.strategy):
            if leg.type == "closed":
                total += leg.prev_pos
                continue
            sign = 1 if leg.action == "buy" else -1
            previous = leg.prev_pos or 0.0
            current = inputs.stock_price if leg.type == "stock" else leg.premium
            if previous < 0:
                total -= sign * leg.n * (current + previous)
                continue
            basis = previous if previous > 0 else current
            if leg.type == "stock":
                total += sign * leg.n * (s - basis) - inputs.stock_commission
                continue
            tau = (data.days_to_maturity[i] - data.days_to_target) / data.days_in_year
            if tau > 0:
                w = inputs.volatility * math.sqrt(tau)
                d = (
                    math.log(s / leg.strike)
                    + (
                        inputs.interest_rate
                        - inputs.dividend_yield
                        + inputs.volatility**2 / 2
                    )
                    * tau
                ) / w
                dy, dr = math.exp(-inputs.dividend_yield * tau), math.exp(
                    -inputs.interest_rate * tau
                )
                value = (
                    s * dy * norm.cdf(d) - leg.strike * dr * norm.cdf(d - w)
                    if leg.type == "call"
                    else leg.strike * dr * norm.cdf(w - d) - s * dy * norm.cdf(-d)
                )
            else:
                value = (
                    max(s - leg.strike, 0.0)
                    if leg.type == "call"
                    else max(leg.strike - s, 0.0)
                )
            total += sign * leg.n * (value - basis) - inputs.opt_commission
        return total

    strikes = [
        (math.log(leg.strike) - m) / v
        for leg in inputs.strategy
        if leg.type in ("call", "put")
    ]
    # Known fixtures have broad, simple roots; include all payoff kinks exactly.
    grid = sorted(
        set([*np.linspace(-14, 14, 401), *[z for z in strikes if -14 < z < 14]])
    )

    def f(z):
        return payoff(math.exp(m + v * z)) - target

    values = [f(z) for z in grid]
    roots = [
        brentq(f, lower, upper, xtol=1e-13)
        for lower, upper, fl, fu in zip(grid[:-1], grid[1:], values[:-1], values[1:])
        if fl * fu < 0
    ]
    roots += [z for z, value in zip(grid, values) if value == 0]
    boundaries = [-14, *sorted(set(roots)), 14]
    probabilities, moments, groups = [0.0, 0.0], [0.0, 0.0], ([], [])
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        group = 0 if f((lower + upper) / 2) >= 0 else 1
        probabilities[group] += (
            norm.sf(lower) - norm.sf(upper)
            if lower > 0
            else norm.cdf(upper) - norm.cdf(lower)
        )
        cuts = [lower, *sorted(z for z in strikes if lower < z < upper), upper]
        moments[group] += sum(
            quad(
                lambda z: payoff(math.exp(m + v * z)) * norm.pdf(z),
                a,
                b,
                epsabs=1e-8,
                epsrel=1e-11,
            )[0]
            for a, b in zip(cuts[:-1], cuts[1:])
        )
        interval = (
            0.0 if lower == -14 else math.exp(m + v * lower),
            math.inf if upper == 14 else math.exp(m + v * upper),
        )
        if groups[group] and groups[group][-1][1] == interval[0]:
            groups[group][-1] = (groups[group][-1][0], interval[1])
        else:
            groups[group].append(interval)
    means = [moment / p if p else 0.0 for moment, p in zip(moments, probabilities)]
    return probabilities, means, groups


def assert_reference(outputs):
    p, means, groups = reference(outputs)
    assert outputs.probability_of_profit == pytest.approx(p[0], abs=1e-9, rel=0)
    assert outputs.expected_profit_if_profitable == pytest.approx(
        means[0], abs=0.0055, rel=0
    )
    assert outputs.expected_loss_if_unprofitable == pytest.approx(
        means[1], abs=0.0055, rel=0
    )
    np.testing.assert_allclose(outputs.profit_ranges, groups[0], atol=1e-6, rtol=0)
    for target, probability, intervals, index in (
        (
            outputs.inputs.profit_target,
            outputs.probability_of_profit_target,
            outputs.profit_target_ranges,
            0,
        ),
        (
            outputs.inputs.loss_limit,
            outputs.probability_of_loss_limit,
            outputs.loss_limit_ranges,
            1,
        ),
    ):
        if target is not None:
            p, _, groups = reference(outputs, target if index == 0 else target + 0.01)
            assert probability == pytest.approx(p[index], abs=1e-9, rel=0)
            np.testing.assert_allclose(intervals, groups[index], atol=1e-6, rtol=0)
