"""Shared risk-neutral integration of nominal, conditional strategy returns."""

import numpy as np
from scipy.special import ndtr

from optionlab.models import PoPOutputs


def normal_mass(lower, upper):
    """Avoid cancellation in representable right-tail probabilities."""
    return np.where(lower > 0, ndtr(-lower) - ndtr(-upper), ndtr(upper) - ndtr(lower))


def array_segments(s, profit):
    """Linear interpolation and end secants; a single point means constant P/L."""
    s, profit = np.asarray(s, dtype=float), np.asarray(profit, dtype=float)
    if (
        s.ndim != 1
        or profit.shape != s.shape
        or not s.size
        or not np.all(np.isfinite(s))
        or not np.all(np.isfinite(profit))
        or np.any(s < 0)
        or np.any(np.diff(s) <= 0)
    ):
        raise ValueError(
            "Prices/profits must be finite matching nonempty 1-D arrays with increasing nonnegative prices"
        )
    if s.size == 1:
        return np.array([[0.0, np.inf, 0.0, profit[0]]])
    a = np.diff(profit) / np.diff(s)
    b = profit[:-1] - a * s[:-1]
    lower, upper = s[:-1].copy(), s[1:].copy()
    lower[0], upper[-1] = 0.0, np.inf
    return np.column_stack((lower, upper, a, b))


def partition(segments, target):
    lower, upper, a, b = segments.T
    root = np.divide(target - b, a, out=np.full_like(a, np.nan), where=a != 0)
    crossing = (root > lower) & (root < upper)
    first = np.column_stack((lower, np.where(crossing, root, upper), a, b))
    second = np.column_stack(
        (root[crossing], upper[crossing], a[crossing], b[crossing])
    )
    pieces = np.concatenate((first, second))
    owners = np.concatenate((np.arange(len(segments)), np.flatnonzero(crossing)))
    order = np.argsort(pieces[:, 0], kind="stable")
    pieces, owners = pieces[order], owners[order]
    lower, upper, slope, intercept = pieces.T
    roots = np.divide(
        target - intercept, slope, out=np.zeros_like(slope), where=slope != 0
    )
    # Endpoint roots belong to the adjacent open interval according to slope.
    # Constant segments exactly on target retain their entire mass.
    above = np.where(
        slope > 0,
        lower >= roots,
        np.where(slope < 0, upper <= roots, intercept >= target),
    )
    return pieces, above, owners


def ranges(segments, target):
    pieces, above, _ = partition(segments, target)
    groups = ([], [])
    for (lower, upper, _, _), success in zip(pieces, above):
        group = groups[0 if success else 1]
        if group and group[-1][1] == lower:
            group[-1] = (group[-1][0], float(upper))
        else:
            group.append((float(lower), float(upper)))
    return groups


def integrate(segments, m, v):
    lower, upper, a, b = segments.T
    with np.errstate(divide="ignore"):
        zl, zu = (np.log(lower) - m) / v, (np.log(upper) - m) / v
    p = normal_mass(zl, zu)
    first = np.exp(m + v * v / 2) * normal_mass(zl - v, zu - v)
    return p, a * first + b * p


def event_integrals(segments, inputs, target):
    m = (
        np.log(inputs.stock_price)
        + (inputs.interest_rate - inputs.dividend_yield - inputs.volatility**2 / 2)
        * inputs.years_to_target_date
    )
    v = inputs.volatility * np.sqrt(inputs.years_to_target_date)
    pieces, above, owners = partition(segments, target)
    p, moment = integrate(pieces, m, v)
    probabilities, moments = np.zeros((2, len(segments))), np.zeros((2, len(segments)))
    for i, mask in enumerate((above, ~above)):
        np.add.at(probabilities[i], owners[mask], p[mask])
        np.add.at(moments[i], owners[mask], moment[mask])
    return probabilities, moments


def profile_pop(
    segments, inputs, target=0.01, calculate_expectation=True, evaluator=None
):
    reaching, missing = ranges(segments, target)
    if inputs.years_to_target_date == 0 or inputs.volatility == 0:
        price = inputs.stock_price * np.exp(
            (inputs.interest_rate - inputs.dividend_yield) * inputs.years_to_target_date
        )
        if evaluator is None:
            row = segments[np.searchsorted(segments[:, 1], price, side="right")]
            value = row[2] * price + row[3]
        else:
            value = float(evaluator(np.array([price]))[0])
        success = value >= target
        probabilities = [float(success), float(not success)]
        expectations = [value if success else 0.0, 0.0 if success else value]
    else:
        p, moment = event_integrals(segments, inputs, target)
        probabilities = p.sum(axis=1)
        expectations = np.divide(
            moment.sum(axis=1), probabilities, out=np.zeros(2), where=probabilities > 0
        )
    return PoPOutputs(
        probability_of_reaching_target=probabilities[0],
        probability_of_missing_target=probabilities[1],
        reaching_target_range=reaching,
        missing_target_range=missing,
        expected_return_above_target=(
            float(np.round(expectations[0], 2)) if calculate_expectation else 0.0
        ),
        expected_return_below_target=(
            float(np.round(expectations[1], 2)) if calculate_expectation else 0.0
        ),
    )


def adaptive_segments(evaluate, bounds, knots, inputs, target, calculate_expectation):
    """Bound interpolation errors, including infinite tails and rare events.

    Bracket events by approximate P/L +/- the absolute per-cell error. This
    detects narrow regions and tangencies without relying on sign changes.
    Probability tolerance is 1e-9; conditional monetary tolerance is .0005.
    Bounds exclude floating-point roundoff. Nonconvergence raises explicitly.
    """
    m = (
        np.log(inputs.stock_price)
        + (inputs.interest_rate - inputs.dividend_yield - inputs.volatility**2 / 2)
        * inputs.years_to_target_date
    )
    v = inputs.volatility * np.sqrt(inputs.years_to_target_date)
    nodes = np.unique(np.concatenate((np.exp(m + v * np.linspace(-8, 8, 65)), knots)))
    nodes = nodes[(nodes > 0) & np.isfinite(nodes)]
    for _ in range(60):
        values = evaluate(nodes)
        a = np.diff(values) / np.diff(nodes)
        interior = np.column_stack(
            (nodes[:-1], nodes[1:], a, values[:-1] - a * nodes[:-1])
        )
        segments, error = bounds(nodes, interior)
        p, moment = event_integrals(segments, inputs, target)
        low, high = segments.copy(), segments.copy()
        low[:, 3] -= error
        high[:, 3] += error
        pl, _ = event_integrals(low, inputs, target)
        ph, _ = event_integrals(high, inputs, target)
        # Either event may be rare. Use both complementary differences to
        # retain uncertainty that subtracting near-one probabilities can lose.
        ambiguity = np.maximum(np.maximum(ph[0] - pl[0], pl[1] - ph[1]), 0.0)
        dp = ambiguity.sum()
        error_mass = error * p.sum(axis=0)
        # On the ambiguous set |approximate P/L| <= |target|+error.
        dm = (error_mass + ambiguity * (abs(target) + 2 * error)).sum()
        prob = p.sum(axis=1)
        means = np.divide(moment.sum(axis=1), prob, out=np.zeros(2), where=prob > 0)
        conditional_error = np.divide(
            dm + np.abs(means) * dp, prob - dp, out=np.full(2, np.inf), where=prob > dp
        )
        conditional_error[(prob == 0) & (dp == 0)] = 0
        if dp <= 1e-9 and (
            not calculate_expectation or np.max(conditional_error) <= 0.0005
        ):
            return segments
        score = ambiguity / max(dp, 1e-300)
        if calculate_expectation:
            score += error_mass / max(error_mass.sum(), 1e-300)
        refine = score > 0.25 / len(segments)
        extra = (segments[refine, 0] + segments[refine, 1]) / 2
        extra[~np.isfinite(extra)] = nodes[-1] * 2
        nodes = np.unique(np.concatenate((nodes, extra)))
        if len(nodes) > 250_000 or not np.all(np.isfinite(nodes)):
            break
    raise RuntimeError(
        "Pre-expiry profit integration did not converge to probability 1e-9 / conditional return 0.0005 tolerances"
    )
