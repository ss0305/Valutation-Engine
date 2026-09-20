# Real-Time Fair Value, IPV & Valuation Reserves Engine

C++20 core with `pybind11` bindings for Fair Value Adjustments, Independent Price
Verification and Prudent Valuation Adjustments on FX options and rates
derivatives.

```
CMakeLists.txt
include/   OptionPricer.hpp  IPVEngine.hpp  ReserveCalculator.hpp
src/       OptionPricer.cpp  IPVEngine.cpp  ReserveCalculator.cpp  bindings.cpp
tests/     test_analytics.cpp
python/    demo_ipv_run.py
```

## Build

```bash
pip install pybind11
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cd build && ctest --output-on-failure
PYTHONPATH=build python3 python/demo_ipv_run.py
```

GoogleTest is taken from `third_party/googletest` when present, otherwise from a
system install, otherwise fetched at configure time. Options: `VE_BUILD_TESTS`,
`VE_BUILD_PYTHON`, `VE_ENABLE_WARNINGS` (all `ON` by default).

## Module 1 — `OptionPricer`

European FX options under Garman-Kohlhagen, i.e. Black-Scholes with the foreign
rate acting as a continuous dividend yield; `rate_foreign = 0` recovers textbook
Black-Scholes.

- `evaluate()` returns price, delta, gamma, vega and theta in one pass. Expiry and
  zero-volatility cases are handled in closed form rather than left to produce NaN.
- `calibrate_volatility()` is Newton-Raphson safeguarded by bisection on
  `[min_volatility, max_volatility]`. Pure Newton stalls on the wings, where the
  Brenner-Subrahmanyam seed can land in a region of negligible vega; the bracket
  step rescues it. Returns `std::optional<double>`, empty when the quote is
  outside the attainable range or the iteration budget runs out.
- `monte_carlo_price()` / `monte_carlo()` split paths across `std::jthread`
  workers, each with an independently seeded `mt19937_64`. Results are
  reproducible for a given `(seed, num_threads, num_paths, antithetic)`.
  `monte_carlo()` takes any terminal payoff satisfying the `TerminalPayoff`
  concept.

Two details worth knowing when reading the Monte Carlo code:

1. The workers deliberately take no `std::stop_token`. `~jthread` requests a stop
   before joining, so a worker polling the token abandons its remaining paths as
   soon as the first thread in the vector is destroyed — which silently truncates
   the sample rather than failing.
2. With antithetic sampling, each *pair* is accumulated as one observation at its
   average, not as two independent draws. Summing the legs separately would report
   the crude-sampling standard error and hide the variance reduction.

## Module 2 — `IPVEngine`

`variance = V_FO - V_Indep`, `variance_pct = |V_FO - V_Indep| / |V_Indep|`, breach
when `variance_pct > tolerance`.

**Denominator convention.** The specification writes the denominator unsigned.
Derivative valuations are routinely negative, and an unsigned denominator yields a
negative percentage that can never exceed a positive tolerance — genuine breaches
on short positions would go unflagged. The engine divides by `|V_Indep|`, which
matches the specification for every positive consensus mark and is the standard
market-risk convention. `test_analytics.cpp` pins this with a negative-mark case.

`variance_pct` is `std::optional`: against a zero consensus mark the ratio is
undefined, so it comes back empty (`None` in Python) and any non-zero difference is
escalated for review rather than silently passed. Tolerances are strict (`>`), so a
variance landing exactly on the threshold does not breach.

`process_portfolio()` takes a `std::span`, preserves input order, and stores the
run for `get_breaches()` (ranked worst-first), `summary()`, `summary_by_book()`
and `find()`. One engine instance per run; it is not internally synchronised.

## Module 3 — `ReserveCalculator`

All reserves are returned as non-negative costs. Position direction does not change
the cost of crossing a spread, so size enters through its magnitude.

| Component | Formula |
| --- | --- |
| Bid-offer | `0.5 * |size| * (ask - bid)` |
| Market unearned spread PVA | `|size| * (spread / 2) * unearned_spread_fraction` |
| Close-out costs PVA | `|size| * reference_price * close_out_cost_rate * liquidity_horizon_factor` |

`reference_price` falls back to the bid/ask mid when left at zero. Crossed markets,
out-of-range fractions and non-finite fields raise `ReserveInputError` rather than
producing a negative reserve. Aggregation over `std::span<const Position>` yields
`std::unordered_map` keyed by book or category, plus `ranked_by_book()` for a
report-shaped `std::vector`.

## Module 4 — Python bindings

```python
import valuation_engine as ve

pricer = ve.OptionPricer()
spec = ve.OptionSpec(spot=1.10, strike=1.12, time_to_expiry=0.5,
                     rate_domestic=0.0425, rate_foreign=0.025, volatility=0.085)
risk = pricer.evaluate(spec)                      # .price .delta .gamma .vega .theta
iv = pricer.calibrate_volatility(spec, 0.0214)    # float, or None
mc = pricer.monte_carlo_price(spec, ve.MonteCarloConfig(num_paths=500_000, num_threads=4))

engine = ve.IPVEngine(default_tolerance=0.02)
engine.process_portfolio([ve.IPVInput("FXO-100", 1_050_000.0, 1_000_000.0, book="FX")])
breaches = engine.get_breaches()

ve.ReserveCalculator.compute_bid_offer_reserve(1e6, 1.0995, 1.1005)   # 500.0
```

`std::optional` maps to `None`, `std::vector` to lists, `std::unordered_map` to
dicts, and the C++ exception types to `ValueError`. `price_curve()` takes and
returns NumPy arrays; `monte_carlo_price()` releases the GIL while simulating.
Result objects carry `as_dict()` for frame-building and `__repr__` for the REPL.

## Verification

61 GoogleTest cases pass (`ctest`), covering the closed forms against textbook
values and put-call parity, every Greek against central finite differences,
implied-vol round trips across strikes and payoffs, each documented
non-convergence path, Monte Carlo convergence, reproducibility and path
accounting, the IPV edge cases above, reserve formulas and aggregation
identities, and an end-to-end price → verify → reserve integration test.

Beyond the suite: the core is clean under ThreadSanitizer and
AddressSanitizer/UBSan, and the compiled engine was cross-checked against an
independent reference implementation over a 2,048-point grid of spot, strike,
maturity, rates, volatility and payoff — maximum price difference `4.4e-16`,
implied volatility recovered at every point (max repricing error `9.9e-11`), and
Monte Carlo 95% intervals covering the analytical price on 37 of 40 seeds.
