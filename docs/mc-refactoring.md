# Monte Carlo price-path refactoring

## Contract

Keep the function name and argument order. Interpret `n_intervals` as the number
of time steps and return `(n_paths, n_intervals + 1)`, including both the initial
price and maturity. Use 252 trading days per year and zero dividend yield.
The additional column is an intentional change from the original implementation.

## Implementation plan

1. Validate scalar inputs and integer dimensions before allocating the output.
2. Replace Python's global random generator with `np.random.default_rng(seed)`.
   Repeated calls remain reproducible, but values differ from the old generator.
3. Allocate one C-contiguous float64 output array and fill it directly with
   `Generator.standard_normal(out=paths)`. Discard the first draw in each row
   when setting its initial log return to zero. This small extra sampling cost
   avoids allocating a second full-sized random array.
4. Scale and shift the draws into log returns, then use
   `np.cumsum(axis=1, out=paths)` and `np.exp(out=paths)`. Multiply by the initial
   price in place. Accumulating log returns avoids a Python recurrence.
5. Document parameters, shape, units, assumptions, and compatibility changes in
   module/function docstrings, following the existing pdoc conventions.
6. Test deterministic paths through maturity, one interval, zero time, empty
   batches, seed reproducibility, global random-state preservation, invalid
   inputs, and Black-Scholes means and covariance across time.
7. Run the existing test suite and compare execution times using equal output
   dimensions and equal time horizons.

## Efficiency and validation

Time complexity remains O(n_paths * n_intervals), the minimum order needed to
produce the complete output. The main allocation occupies
`8 * n_paths * (n_intervals + 1)` bytes. Operations reuse that buffer rather than
materializing separate shock, cumulative-return, and price matrices.

A local measurement with Python's standard random generator and NumPy 2.4.6,
using 2,000 paths and 252 intervals, gave:

| Implementation | Best of five runs after warm-up |
| --- | ---: |
| Python nested loops, adjusted to include maturity | 542 ms |
| NumPy implementation | 12.1 ms |

The measured speedup was approximately 45x. Both implementations include the
initial point and maturity. Random samples differ; this comparison measures the
whole implementation, including the generator change. Results depend on hardware
and dimensions and do not establish a universal optimum.

Validation: 173 tests passed, 8 benchmark tests deselected by the repository's
default configuration; Ruff passed for the two Python files. Documentation lives
in the source docstrings and can be rendered by the existing `runpdoc.bat`.

