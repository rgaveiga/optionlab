import datetime as dt

import numpy as np
import pytest

from optionlab import run_strategy
from optionlab.black_scholes import get_bs_info, get_implied_vol
from optionlab.utils import get_nonbusiness_days

COVERED_CALL_LEGS = [
    {"type": "stock", "n": 100, "action": "buy"},
    {"type": "call", "strike": 185.0, "premium": 4.1, "n": 100, "action": "sell"},
]


@pytest.mark.benchmark(group="engine")
def test_benchmark_run_strategy_full(benchmark, nvidia):
    payload = nvidia | {"strategy": COVERED_CALL_LEGS}

    outputs = benchmark(run_strategy, payload)

    assert outputs.probability_of_profit > 0.0


@pytest.mark.benchmark(group="engine")
def test_benchmark_run_strategy_pop_only(benchmark, nvidia):
    payload = nvidia | {"strategy": COVERED_CALL_LEGS, "calculations": ["pop"]}

    outputs = benchmark(run_strategy, payload)

    assert outputs.probability_of_profit > 0.0


@pytest.mark.benchmark(group="engine-wide")
def test_benchmark_run_strategy_wide_domain(benchmark, nvidia):
    payload = nvidia | {
        "strategy": COVERED_CALL_LEGS,
        "min_stock": 0.01,
        "max_stock": 1000.0,
        "calculations": ["pop", "expectation"],
    }

    outputs = benchmark.pedantic(run_strategy, args=(payload,), rounds=3, iterations=1)

    assert outputs.probability_of_profit > 0.0


@pytest.mark.benchmark(group="engine-wide")
def test_benchmark_run_strategy_wide_domain_coarse_step(benchmark, nvidia):
    payload = nvidia | {
        "strategy": COVERED_CALL_LEGS,
        "min_stock": 0.01,
        "max_stock": 1000.0,
        "price_step": 0.1,
        "calculations": ["pop", "expectation"],
    }

    outputs = benchmark.pedantic(run_strategy, args=(payload,), rounds=3, iterations=1)

    assert outputs.probability_of_profit > 0.0


@pytest.mark.benchmark(group="engine-mc")
def test_benchmark_run_strategy_array_model(benchmark, nvidia):
    time_to_target = 24 / 252
    rng = np.random.default_rng(42)
    log_mean = (
        np.log(nvidia["stock_price"])
        + (nvidia["interest_rate"] - 0.5 * nvidia["volatility"] ** 2) * time_to_target
    )
    log_sigma = nvidia["volatility"] * np.sqrt(time_to_target)
    payload = nvidia | {
        "strategy": COVERED_CALL_LEGS,
        "model": "array",
        "array": rng.lognormal(log_mean, log_sigma, 1_000_000),
    }

    outputs = benchmark.pedantic(run_strategy, args=(payload,), rounds=3, iterations=1)

    assert outputs.probability_of_profit > 0.0


@pytest.mark.benchmark(group="black-scholes")
def test_benchmark_get_bs_info_vectorized(benchmark):
    strikes = np.linspace(50.0, 300.0, 100_000)

    bs = benchmark(get_bs_info, 168.99, strikes, 0.045, 0.483, 24 / 252)

    assert bs.call_price.shape == strikes.shape


@pytest.mark.benchmark(group="black-scholes")
def test_benchmark_get_implied_vol(benchmark):
    implied_vol = benchmark(
        get_implied_vol, "call", 4.1, 168.99, 185.0, 0.045, 24 / 252
    )

    assert 0.001 < implied_vol < 1.0


@pytest.mark.benchmark(group="utils")
def test_benchmark_get_nonbusiness_days(benchmark):
    def run_uncached():
        get_nonbusiness_days.cache_clear()
        return get_nonbusiness_days(dt.date(2023, 1, 1), dt.date(2024, 12, 31))

    count = benchmark(run_uncached)

    assert count > 0
