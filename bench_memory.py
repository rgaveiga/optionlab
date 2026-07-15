"""Measures peak memory allocated during an array-model run_strategy call."""

import tracemalloc

import numpy as np

from optionlab import run_strategy

rng = np.random.default_rng(42)
payload = dict(
    stock_price=168.99,
    volatility=0.483,
    interest_rate=0.045,
    min_stock=68.99,
    max_stock=268.99,
    days_to_target_date=24,
    strategy=[
        {"type": "stock", "n": 100, "action": "buy"},
        {"type": "call", "strike": 185.0, "premium": 4.1, "n": 100, "action": "sell"},
    ],
    model="array",
    array=rng.lognormal(np.log(168.99), 0.1, 1_000_000),
)

run_strategy(payload)  # warm-up (fills lru caches)

tracemalloc.start()
run_strategy(payload)
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"peak allocations during run_strategy: {peak / 1e6:.1f} MB")
