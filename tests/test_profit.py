import numpy as np
import pytest
from scipy.stats import norm

from optionlab import run_strategy
from optionlab.models import BlackScholesModelInputs, LaplaceInputs
from optionlab.support import get_pop, _integrate_linear_profit_bs
from optionlab.profit import array_segments, event_integrals
from tests.profit_reference import assert_reference


BASE = dict(
    stock_price=100.0,
    volatility=0.2,
    interest_rate=0.05,
    days_to_target_date=365,
    discard_nonbusiness_days=False,
    min_stock=50.0,
    max_stock=150.0,
    price_step=1.0,
    calculations=["pop", "expectation"],
)
MODEL = BlackScholesModelInputs(
    stock_price=100.0, volatility=0.2, interest_rate=0.05, years_to_target_date=1.0
)


@pytest.mark.parametrize("domain", [(50, 150, 1), (110, 120, 2), (50, 90, 5)])
@pytest.mark.parametrize(
    "kind,p,above,below",
    [
        ("stock", 0.5594204551783997, 19.64, -13.30),
        ("call", 0.4623849888709348, 18.24, -4.55),
    ],
)
def test_complete_profile_analytic_reference(domain, kind, p, above, below):
    leg = dict(type=kind, action="buy", n=1)
    if kind == "call":
        leg.update(strike=100.0, premium=5.0)
    out = run_strategy(
        BASE
        | dict(
            min_stock=domain[0],
            max_stock=domain[1],
            price_step=domain[2],
            strategy=[leg],
        )
    )
    assert out.probability_of_profit == pytest.approx(p, abs=1e-12, rel=0)
    assert out.expected_profit_if_profitable == above
    assert out.expected_loss_if_unprofitable == below
    assert out.minimum_return_in_the_domain == out.data.strategy_profit.min()
    assert_reference(out)


@pytest.mark.parametrize("target", [0.01, 0.0, 1.0])
def test_partition_and_total_expectation(target):
    for s, profit, expected in (
        (np.arange(50.0, 151.0), np.arange(50.0, 151.0) - 100, 100 * np.expm1(0.05)),
        (
            np.array([0.0, 100.0, 150.0]),
            np.array([-5.0, -5.0, 45.0]),
            100 * np.exp(0.05) * norm.cdf(0.35) - 100 * norm.cdf(0.15) - 5,
        ),
    ):
        p, m = event_integrals(array_segments(s, profit), MODEL, target)
        assert p.sum() == pytest.approx(1.0, abs=1e-12, rel=0)
        assert m.sum() == pytest.approx(expected, abs=1e-11, rel=0)


@pytest.mark.parametrize(
    "vol,time,target,p",
    [(0.2, 0, 0, 1), (0.2, 0, 0.01, 0), (0, 1, 0, 1), (0, 1, 100 * np.expm1(0.05), 1)],
)
def test_degenerate(vol, time, target, p):
    model = MODEL.model_copy(update=dict(volatility=vol, years_to_target_date=time))
    # Use the same direct deterministic price expression for exact equality.
    if vol == 0 and target > 0:
        target = 100 * np.exp(0.05) - 100
    with np.errstate(all="raise"):
        out = get_pop(np.array([50.0, 150.0]), np.array([-50.0, 50.0]), model, target)
    assert out.probability_of_reaching_target == p
    assert out.probability_of_missing_target == 1 - p


@pytest.mark.parametrize("value,p", [(-1, 0), (0.01, 1), (1, 1)])
def test_single_point_constant(value, p):
    out = get_pop(np.array([100.0]), np.array([value]), MODEL)
    assert out.probability_of_reaching_target == p
    assert out.expected_return_above_target == (value if p else 0)


@pytest.mark.parametrize(
    "prices,profits",
    [
        ([], []),
        ([1, 1], [1, 2]),
        ([2, 1], [1, 2]),
        ([1, 2], [1]),
        ([1, np.nan], [1, 2]),
        ([-1, 2], [1, 2]),
    ],
)
def test_invalid_interpolation_arrays(prices, profits):
    with pytest.raises(ValueError):
        get_pop(np.array(prices), np.array(profits), MODEL)


def test_disconnected_and_isolated_events():
    s = np.array([0.0, 90.0, 100.0, 110.0, 200.0])
    out = get_pop(s, np.array([1.0, 1.0, -1.0, 1.0, 1.0]), MODEL, 0.0)
    assert out.reaching_target_range == [(0.0, 95.0), (105.0, np.inf)]
    assert (
        out.probability_of_reaching_target + out.probability_of_missing_target
        == pytest.approx(1.0, abs=1e-12)
    )
    isolated = get_pop(
        np.array([0.0, 100.0, 200.0]), np.array([-1.0, 0.0, -1.0]), MODEL, 0.0
    )
    assert isolated.probability_of_reaching_target == 0


def test_right_tail_moment_is_representable():
    lower, upper = np.exp(9.0), np.exp(10.0)
    p, moment = _integrate_linear_profit_bs(
        np.array([lower]), np.array([upper]), np.array([1.0]), np.array([0.0]), 0.0, 1.0
    )
    assert norm.cdf(9.0) == norm.cdf(10.0) == 1
    assert p[0] == pytest.approx(norm.sf(9) - norm.sf(10), rel=1e-13, abs=0)
    assert moment[0] == pytest.approx(
        np.exp(0.5) * (norm.sf(8) - norm.sf(9)), rel=1e-13, abs=0
    )


@pytest.mark.parametrize(
    "strikes,quantities", [([99.3, 105.7], [1, -1]), ([98.3, 100.7, 103.1], [1, -2, 1])]
)
def test_off_grid_spreads(strikes, quantities):
    legs = [
        dict(
            type="call",
            strike=k,
            premium=2.0,
            n=abs(n),
            action="buy" if n > 0 else "sell",
        )
        for k, n in zip(strikes, quantities)
    ]
    assert_reference(
        run_strategy(
            BASE | dict(strategy=legs, min_stock=110, max_stock=120, price_step=3)
        )
    )


@pytest.mark.parametrize("domain", [(50, 150, 1), (110, 120, 2)])
def test_pre_expiry_dividends_multiple_maturities(domain):
    legs = [
        dict(type="call", strike=99.3, premium=5.0, n=2, action="buy", expiration=500),
        dict(type="put", strike=102.7, premium=4.0, n=1, action="buy", expiration=600),
    ]
    assert_reference(
        run_strategy(
            BASE
            | dict(
                strategy=legs,
                dividend_yield=0.03,
                min_stock=domain[0],
                max_stock=domain[1],
                price_step=domain[2],
            )
        )
    )


def test_zero_volatility_public_and_laplace_validation():
    out = run_strategy(
        BASE
        | dict(
            volatility=0.0,
            calculations=["pop", "expectation", "greeks", "impvol"],
            strategy=[dict(type="call", strike=100.0, premium=5.0, n=1, action="buy")],
        )
    )
    assert out.probability_of_profit == 1
    assert out.expected_profit_if_profitable == 0.13
    assert np.all(np.isfinite(out.gamma + out.delta + out.theta + out.vega + out.rho))
    with pytest.raises(ValueError):
        LaplaceInputs(stock_price=100.0, volatility=0.0, years_to_target_date=1.0)


def test_deterministic_kink_has_explicit_greek_limitation():
    payload = BASE | dict(
        volatility=0.0,
        interest_rate=0.0,
        strategy=[dict(type="call", strike=100.0, premium=5.0, n=1, action="buy")],
    )
    with pytest.raises(ValueError, match="Greeks are undefined"):
        run_strategy(payload | dict(calculations=["pop", "greeks"]))
    out = run_strategy(payload)
    assert out.probability_of_profit == 0
    assert out.expected_loss_if_unprofitable == -5


def test_zero_volatility_before_expiration():
    payload = BASE | dict(
        volatility=0.0,
        dividend_yield=0.03,
        strategy=[
            dict(
                type="call",
                strike=100.0,
                premium=5.0,
                n=1,
                action="buy",
                expiration=730,
            )
        ],
    )
    out = run_strategy(payload)
    spot = 100 * np.exp(0.05 - 0.03)
    value = max(spot * np.exp(-0.03) - 100 * np.exp(-0.05), 0) - 5
    assert out.expected_loss_if_unprofitable == np.round(value, 2)
    assert out.probability_of_profit == 0


def test_constant_plateau_on_target_retains_mass():
    out = get_pop(np.array([0.0, 100.0, 200.0]), np.array([0.01, 0.01, -1.0]), MODEL)
    expected = norm.cdf((np.log(100) - (np.log(100) + 0.03)) / 0.2)
    assert out.probability_of_reaching_target == pytest.approx(expected, abs=1e-12)
    assert out.expected_return_above_target == 0.01


def test_pre_expiry_narrow_profit_region():
    from scipy.optimize import minimize_scalar, brentq

    # Independent BS butterfly: target just below its interior maximum.
    def payoff(s):
        result = 0.0
        for strike, quantity in ((95.0, 1), (100.0, -2), (105.0, 1)):
            d1 = (np.log(s / strike) + 0.07) / 0.2
            result += quantity * (
                s * norm.cdf(d1) - strike * np.exp(-0.05) * norm.cdf(d1 - 0.2)
            )
        return result

    peak = minimize_scalar(
        lambda s: -payoff(s), bounds=(70.0, 130.0), method="bounded"
    ).x
    target = payoff(peak) - 1e-5
    lower = brentq(lambda s: payoff(s) - target, 70.0, peak)
    upper = brentq(lambda s: payoff(s) - target, peak, 130.0)
    expected = norm.cdf((np.log(upper / 100) - 0.03) / 0.2) - norm.cdf(
        (np.log(lower / 100) - 0.03) / 0.2
    )
    legs = [
        dict(
            type="call",
            strike=k,
            premium=1.0,
            n=abs(n),
            action="buy" if n > 0 else "sell",
            expiration=730,
        )
        for k, n in ((95.0, 1), (100.0, -2), (105.0, 1))
    ]
    out = run_strategy(
        BASE | dict(strategy=legs, profit_target=target, calculations=["pop"])
    )
    assert out.probability_of_profit_target == pytest.approx(expected, abs=1e-9, rel=0)
