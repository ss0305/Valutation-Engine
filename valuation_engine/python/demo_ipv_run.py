"""End-to-end demonstration of the valuation engine through its Python bindings.

Prices an FX options book, verifies the Front Office marks against independent
consensus prices, and computes the valuation reserves for the same positions.

Run from the build directory, e.g.:
    PYTHONPATH=build python3 python/demo_ipv_run.py
"""

import numpy as np

import valuation_engine as ve

# -- Module 1: pricing and calibration ----------------------------------------
pricer = ve.OptionPricer()

spec = ve.OptionSpec(
    spot=1.1000,
    strike=1.1200,
    time_to_expiry=0.5,
    rate_domestic=0.0425,
    rate_foreign=0.0250,
    volatility=0.085,
    type=ve.OptionType.Call,
)

risk = pricer.evaluate(spec)
print("=== Module 1: pricing ===")
print(f"  price {risk.price:.8f}  delta {risk.delta:.6f}  "
      f"gamma {risk.gamma:.6f}  vega {risk.vega:.6f}  theta {risk.theta:.6f}")

implied = pricer.calibrate_volatility(
    ve.OptionSpec(spot=1.10, strike=1.12, time_to_expiry=0.5,
                  rate_domestic=0.0425, rate_foreign=0.0250),
    risk.price,
)
print(f"  implied vol from that price: {implied:.10f} (input 0.085)")

mc = pricer.monte_carlo_price(spec, ve.MonteCarloConfig(num_paths=500_000, num_threads=4, seed=99))
print(f"  monte carlo {mc.price:.8f} +/- {mc.confidence_interval_95:.8f} "
      f"({mc.paths:,} paths, {mc.threads_used} threads)")

spots = np.linspace(1.05, 1.15, 5)
print(f"  price curve {np.round(pricer.price_curve(spec, spots), 6)}")

# A market quote that no volatility can reproduce comes back as None.
print("  unreachable quote ->", pricer.calibrate_volatility(spec, 5.0))

# -- Module 2: IPV -------------------------------------------------------------
print("\n=== Module 2: IPV ===")
engine = ve.IPVEngine(default_tolerance=0.02)
portfolio = [
    ve.IPVInput("FXO-100", 1_050_000.0, 1_000_000.0, book="FX-OPTIONS"),
    ve.IPVInput("FXO-101", 495_000.0, 500_000.0, book="FX-OPTIONS"),
    ve.IPVInput("IRS-010", 2_400_000.0, 2_000_000.0, book="RATES"),
    ve.IPVInput("IRS-011", -1_100_000.0, -1_000_000.0, tolerance=0.05, book="RATES"),
]
for result in engine.process_portfolio(portfolio):
    pct = "n/a" if result.variance_pct is None else f"{result.variance_pct:7.2%}"
    print(f"  {result.trade_id:<9} variance {result.variance:>12,.0f}  {pct}  "
          f"{'BREACH' if result.breached else 'ok'}")

summary = engine.summary()
print(f"  {summary.breach_count} of {summary.trade_count} trades breached "
      f"({summary.breach_rate:.0%}); worst is {summary.largest_variance_trade_id}")
print(f"  breaches ranked: {[b.trade_id for b in engine.get_breaches()]}")

# -- Module 3: reserves --------------------------------------------------------
print("\n=== Module 3: reserves and PVA ===")
positions = [
    ve.Position("FXO-100", book="FX-OPTIONS", category="FX Options", position_size=10_000_000.0,
                bid=0.0421, ask=0.0427, unearned_spread_fraction=0.5, close_out_cost_rate=0.0001),
    ve.Position("FXO-101", book="FX-OPTIONS", category="FX Options", position_size=-5_000_000.0,
                bid=0.0180, ask=0.0184, unearned_spread_fraction=0.25, close_out_cost_rate=0.0001),
    ve.Position("IRS-010", book="RATES", category="Rates", position_size=250_000.0,
                bid=99.95, ask=100.05, unearned_spread_fraction=0.10,
                close_out_cost_rate=0.00005, liquidity_horizon_factor=2.0),
]

print(f"  single bid-offer reserve: "
      f"{ve.ReserveCalculator.compute_bid_offer_reserve(10_000_000.0, 0.0421, 0.0427):,.2f}")

for entry in ve.ReserveCalculator.ranked_by_book(positions):
    b = entry.breakdown
    print(f"  {entry.key:<11} bid-offer {b.bid_offer:>12,.2f}  MUS PVA "
          f"{b.market_unearned_spread_pva:>10,.2f}  close-out PVA "
          f"{b.close_out_cost_pva:>10,.2f}  total {b.total:>12,.2f}")

total = ve.ReserveCalculator.compute_portfolio_reserves(positions)
print(f"  portfolio total reserve: {total.total:,.2f} across {total.position_count} positions")
print(f"  as_dict -> {total.as_dict()}")

# -- Error handling surfaces as Python exceptions ------------------------------
print("\n=== Error handling ===")
for label, thunk in [
    ("crossed market", lambda: ve.ReserveCalculator.compute_bid_offer_reserve(1e6, 1.10, 1.09)),
    ("negative spot", lambda: pricer.price(ve.OptionSpec(spot=-1.0, strike=1.0,
                                                         time_to_expiry=1.0, volatility=0.2))),
    ("bad tolerance", lambda: ve.IPVEngine(-0.5)),
]:
    try:
        thunk()
    except ValueError as exc:
        print(f"  {label}: ValueError({exc})")
