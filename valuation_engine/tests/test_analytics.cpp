// =============================================================================
//  test_analytics.cpp
//  Unit and integration coverage for the fair value / IPV / reserves engine.
// =============================================================================
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "IPVEngine.hpp"
#include "OptionPricer.hpp"
#include "ReserveCalculator.hpp"

using namespace valuation;

namespace {

OptionSpec atm_call() {
    OptionSpec spec{};
    spec.spot = 100.0;
    spec.strike = 100.0;
    spec.time_to_expiry = 1.0;
    spec.rate_domestic = 0.05;
    spec.rate_foreign = 0.0;
    spec.volatility = 0.20;
    spec.type = OptionType::Call;
    return spec;
}

OptionSpec fx_call() {
    OptionSpec spec{};
    spec.spot = 1.1000;       // EURUSD
    spec.strike = 1.1200;
    spec.time_to_expiry = 0.5;
    spec.rate_domestic = 0.0425;  // USD
    spec.rate_foreign = 0.0250;   // EUR
    spec.volatility = 0.085;
    spec.type = OptionType::Call;
    return spec;
}

}  // namespace

// =============================================================================
// Module 1: maths helpers
// =============================================================================

TEST(NormalDistribution, CdfMatchesKnownValues) {
    EXPECT_NEAR(math::norm_cdf(0.0), 0.5, 1e-15);
    EXPECT_NEAR(math::norm_cdf(1.0), 0.8413447460685429, 1e-12);
    EXPECT_NEAR(math::norm_cdf(-1.96), 0.024997895148220435, 1e-12);
    EXPECT_NEAR(math::norm_cdf(8.0), 1.0, 1e-12);
    EXPECT_NEAR(math::norm_cdf(-8.0), 0.0, 1e-12);
}

TEST(NormalDistribution, PdfMatchesKnownValues) {
    EXPECT_NEAR(math::norm_pdf(0.0), 0.3989422804014327, 1e-15);
    EXPECT_NEAR(math::norm_pdf(1.0), 0.24197072451914337, 1e-15);
    EXPECT_NEAR(math::norm_pdf(-1.0), math::norm_pdf(1.0), 1e-18);
}

// =============================================================================
// Module 1: analytical pricing
// =============================================================================

TEST(OptionPricing, MatchesTextbookBlackScholesValues) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();

    EXPECT_NEAR(pricer.price(spec), 10.450583572185565, 1e-9);

    spec.type = OptionType::Put;
    EXPECT_NEAR(pricer.price(spec), 5.573526022256971, 1e-9);
}

TEST(OptionPricing, SatisfiesPutCallParity) {
    const OptionPricer pricer;
    OptionSpec call = fx_call();
    OptionSpec put = call;
    put.type = OptionType::Put;

    const double parity = call.spot * std::exp(-call.rate_foreign * call.time_to_expiry) -
                          call.strike * std::exp(-call.rate_domestic * call.time_to_expiry);
    EXPECT_NEAR(pricer.price(call) - pricer.price(put), parity, 1e-12);
}

TEST(OptionPricing, DeepInTheMoneyCallApproachesForwardValue) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();
    spec.spot = 1000.0;
    const double expected = spec.spot * std::exp(-spec.rate_foreign * spec.time_to_expiry) -
                            spec.strike * std::exp(-spec.rate_domestic * spec.time_to_expiry);
    EXPECT_NEAR(pricer.price(spec), expected, 1e-6);
}

TEST(OptionPricing, ExpiredOptionIsWorthIntrinsic) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();
    spec.time_to_expiry = 0.0;
    spec.spot = 120.0;

    const PricingResult result = pricer.evaluate(spec);
    EXPECT_NEAR(result.price, 20.0, 1e-12);
    EXPECT_NEAR(result.delta, 1.0, 1e-12);
    EXPECT_NEAR(result.gamma, 0.0, 1e-12);
    EXPECT_NEAR(result.vega, 0.0, 1e-12);

    spec.spot = 80.0;
    EXPECT_NEAR(pricer.price(spec), 0.0, 1e-12);
}

TEST(OptionPricing, ZeroVolatilityIsDiscountedIntrinsic) {
    const OptionPricer pricer;
    OptionSpec spec = fx_call();
    spec.volatility = 0.0;
    EXPECT_NEAR(pricer.price(spec), OptionPricer::discounted_intrinsic(spec), 1e-15);
}

TEST(OptionPricing, RejectsMalformedSpecifications) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();

    spec.spot = -1.0;
    EXPECT_THROW(static_cast<void>(pricer.price(spec)), InvalidOptionSpec);

    spec = atm_call();
    spec.strike = 0.0;
    EXPECT_THROW(static_cast<void>(pricer.price(spec)), InvalidOptionSpec);

    spec = atm_call();
    spec.time_to_expiry = -0.5;
    EXPECT_THROW(static_cast<void>(pricer.price(spec)), InvalidOptionSpec);

    spec = atm_call();
    spec.volatility = -0.1;
    EXPECT_THROW(static_cast<void>(pricer.price(spec)), InvalidOptionSpec);
}

// =============================================================================
// Module 1: Greeks
// =============================================================================

TEST(Greeks, DeltaMatchesFiniteDifference) {
    const OptionPricer pricer;
    const OptionSpec spec = fx_call();
    const double bump = 1e-5;

    OptionSpec up = spec;
    OptionSpec down = spec;
    up.spot += bump;
    down.spot -= bump;

    const double numeric = (pricer.price(up) - pricer.price(down)) / (2.0 * bump);
    EXPECT_NEAR(pricer.evaluate(spec).delta, numeric, 1e-7);
}

TEST(Greeks, GammaMatchesFiniteDifference) {
    const OptionPricer pricer;
    const OptionSpec spec = fx_call();
    const double bump = 1e-4;

    OptionSpec up = spec;
    OptionSpec down = spec;
    up.spot += bump;
    down.spot -= bump;

    const double numeric =
        (pricer.price(up) - 2.0 * pricer.price(spec) + pricer.price(down)) / (bump * bump);
    EXPECT_NEAR(pricer.evaluate(spec).gamma, numeric, 1e-4);
}

TEST(Greeks, VegaMatchesFiniteDifference) {
    const OptionPricer pricer;
    const OptionSpec spec = fx_call();
    const double bump = 1e-6;

    OptionSpec up = spec;
    OptionSpec down = spec;
    up.volatility += bump;
    down.volatility -= bump;

    const double numeric = (pricer.price(up) - pricer.price(down)) / (2.0 * bump);
    EXPECT_NEAR(pricer.evaluate(spec).vega, numeric, 1e-6);
}

TEST(Greeks, ThetaMatchesFiniteDifference) {
    const OptionPricer pricer;
    const OptionSpec spec = fx_call();
    const double bump = 1e-6;

    OptionSpec longer = spec;
    OptionSpec shorter = spec;
    longer.time_to_expiry += bump;
    shorter.time_to_expiry -= bump;

    // theta = dV/dt = -dV/dT.
    const double numeric = -(pricer.price(longer) - pricer.price(shorter)) / (2.0 * bump);
    EXPECT_NEAR(pricer.evaluate(spec).theta, numeric, 1e-5);
}

TEST(Greeks, GammaAndVegaAreIdenticalForCallsAndPuts) {
    const OptionPricer pricer;
    OptionSpec call = fx_call();
    OptionSpec put = call;
    put.type = OptionType::Put;

    EXPECT_NEAR(pricer.evaluate(call).gamma, pricer.evaluate(put).gamma, 1e-15);
    EXPECT_NEAR(pricer.evaluate(call).vega, pricer.evaluate(put).vega, 1e-15);
}

TEST(Greeks, CallAndPutDeltasDifferByTheForeignDiscountFactor) {
    const OptionPricer pricer;
    OptionSpec call = fx_call();
    OptionSpec put = call;
    put.type = OptionType::Put;

    const double df_foreign = std::exp(-call.rate_foreign * call.time_to_expiry);
    EXPECT_NEAR(pricer.evaluate(call).delta - pricer.evaluate(put).delta, df_foreign, 1e-12);
}

// =============================================================================
// Module 1: implied volatility calibration
// =============================================================================

TEST(ImpliedVolatility, RecoversTheVolatilityUsedToPrice) {
    const OptionPricer pricer;
    for (const double true_vol : {0.05, 0.10, 0.20, 0.45, 0.85}) {
        OptionSpec spec = fx_call();
        spec.volatility = true_vol;
        const double target = pricer.price(spec);

        OptionSpec unknown = spec;
        unknown.volatility = 0.0;  // Force the internal initial guess.
        const std::optional<double> implied = pricer.calibrate_volatility(unknown, target);

        ASSERT_TRUE(implied.has_value()) << "no convergence at vol " << true_vol;
        EXPECT_NEAR(*implied, true_vol, 1e-8);
    }
}

TEST(ImpliedVolatility, RecoversVolatilityAcrossStrikesAndPayoffs) {
    const OptionPricer pricer;
    for (const double strike : {0.95, 1.00, 1.05, 1.15, 1.30}) {
        for (const OptionType type : {OptionType::Call, OptionType::Put}) {
            OptionSpec spec = fx_call();
            spec.strike = strike;
            spec.type = type;
            spec.volatility = 0.135;
            const double target = pricer.price(spec);

            OptionSpec unknown = spec;
            unknown.volatility = 0.0;
            const std::optional<double> implied = pricer.calibrate_volatility(unknown, target);

            ASSERT_TRUE(implied.has_value())
                << "no convergence at strike " << strike << " type " << to_string(type);
            EXPECT_NEAR(*implied, 0.135, 1e-6);
        }
    }
}

TEST(ImpliedVolatility, HonoursACallerSuppliedInitialGuess) {
    const OptionPricer pricer;
    OptionSpec spec = fx_call();
    spec.volatility = 0.25;
    const double target = pricer.price(spec);

    OptionSpec seeded = spec;
    seeded.volatility = 0.05;  // Poor but usable seed.
    const std::optional<double> implied = pricer.calibrate_volatility(seeded, target);

    ASSERT_TRUE(implied.has_value());
    EXPECT_NEAR(*implied, 0.25, 1e-8);
}

TEST(ImpliedVolatility, ReturnsNulloptBelowIntrinsicValue) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();
    spec.spot = 140.0;
    const double intrinsic = OptionPricer::discounted_intrinsic(spec);

    EXPECT_FALSE(pricer.calibrate_volatility(spec, intrinsic * 0.5).has_value());
}

TEST(ImpliedVolatility, ReturnsNulloptAboveNoArbitrageCeiling) {
    const OptionPricer pricer;
    const OptionSpec spec = atm_call();
    EXPECT_FALSE(pricer.calibrate_volatility(spec, spec.spot * 2.0).has_value());
}

TEST(ImpliedVolatility, ReturnsNulloptForNegativeOrNonFiniteTargets) {
    const OptionPricer pricer;
    const OptionSpec spec = atm_call();
    EXPECT_FALSE(pricer.calibrate_volatility(spec, -1.0).has_value());
    EXPECT_FALSE(
        pricer.calibrate_volatility(spec, std::numeric_limits<double>::quiet_NaN()).has_value());
}

TEST(ImpliedVolatility, ReturnsNulloptForExpiredOptions) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();
    spec.time_to_expiry = 0.0;
    EXPECT_FALSE(pricer.calibrate_volatility(spec, 1.0).has_value());
}

TEST(ImpliedVolatility, ReturnsNulloptWhenVegaCollapses) {
    const OptionPricer pricer;
    // A far out-of-the-money, short-dated option is flat in volatility.
    OptionSpec spec = atm_call();
    spec.spot = 100.0;
    spec.strike = 1000.0;
    spec.time_to_expiry = 0.002;
    EXPECT_FALSE(pricer.calibrate_volatility(spec, 1.0).has_value());
}

TEST(ImpliedVolatility, RespectsAnIterationBudget) {
    CalibrationSettings settings{};
    settings.max_iterations = 1;
    settings.price_tolerance = 1e-14;
    const OptionPricer strict(settings);

    OptionSpec spec = fx_call();
    spec.volatility = 0.40;
    const double target = strict.price(spec);

    OptionSpec unknown = spec;
    unknown.volatility = 0.0;
    EXPECT_FALSE(strict.calibrate_volatility(unknown, target).has_value());
}

// =============================================================================
// Module 1: Monte Carlo
// =============================================================================

TEST(MonteCarlo, ConvergesToTheAnalyticalPrice) {
    const OptionPricer pricer;
    const OptionSpec spec = atm_call();

    MonteCarloConfig config{};
    config.num_paths = 400'000;
    config.num_threads = 4;
    config.seed = 20260920;
    config.antithetic = true;

    const MonteCarloResult result = pricer.monte_carlo_price(spec, config);
    const double analytic = pricer.price(spec);

    EXPECT_EQ(result.paths, config.num_paths);
    EXPECT_EQ(result.threads_used, 4U);
    EXPECT_GT(result.standard_error, 0.0);
    EXPECT_NEAR(result.price, analytic, 4.0 * result.standard_error + 1e-3);
}

TEST(MonteCarlo, ConvergesForPutsToo) {
    const OptionPricer pricer;
    OptionSpec spec = fx_call();
    spec.type = OptionType::Put;

    MonteCarloConfig config{};
    config.num_paths = 200'000;
    config.num_threads = 2;
    config.seed = 7;

    const MonteCarloResult result = pricer.monte_carlo_price(spec, config);
    EXPECT_NEAR(result.price, pricer.price(spec), 4.0 * result.standard_error + 1e-5);
}

TEST(MonteCarlo, IsReproducibleForAGivenSeedAndThreadCount) {
    const OptionPricer pricer;
    const OptionSpec spec = atm_call();

    MonteCarloConfig config{};
    config.num_paths = 50'000;
    config.num_threads = 3;
    config.seed = 123456789;

    const MonteCarloResult first = pricer.monte_carlo_price(spec, config);
    const MonteCarloResult second = pricer.monte_carlo_price(spec, config);

    EXPECT_DOUBLE_EQ(first.price, second.price);
    EXPECT_DOUBLE_EQ(first.standard_error, second.standard_error);
}

TEST(MonteCarlo, DifferentSeedsGiveDifferentDraws) {
    const OptionPricer pricer;
    const OptionSpec spec = atm_call();

    MonteCarloConfig a{};
    a.num_paths = 20'000;
    a.num_threads = 2;
    a.seed = 1;

    MonteCarloConfig b = a;
    b.seed = 2;

    EXPECT_NE(pricer.monte_carlo_price(spec, a).price, pricer.monte_carlo_price(spec, b).price);
}

TEST(MonteCarlo, AllocatesEveryRequestedPathAcrossThreads) {
    const OptionPricer pricer;
    const OptionSpec spec = atm_call();

    for (const unsigned threads : {1U, 2U, 3U, 5U, 8U}) {
        MonteCarloConfig config{};
        config.num_paths = 9'999;  // Deliberately indivisible.
        config.num_threads = threads;
        const MonteCarloResult result = pricer.monte_carlo_price(spec, config);
        EXPECT_EQ(result.paths, 9'999U) << "thread count " << threads;
        EXPECT_EQ(result.threads_used, threads);
    }
}

TEST(MonteCarlo, AntitheticSamplingReducesSamplingError) {
    const OptionPricer pricer;
    const OptionSpec spec = atm_call();

    MonteCarloConfig plain{};
    plain.num_paths = 100'000;
    plain.num_threads = 2;
    plain.seed = 99;
    plain.antithetic = false;

    MonteCarloConfig antithetic = plain;
    antithetic.antithetic = true;

    const double plain_error = pricer.monte_carlo_price(spec, plain).standard_error;
    const double antithetic_error = pricer.monte_carlo_price(spec, antithetic).standard_error;
    EXPECT_LT(antithetic_error, plain_error);
}

TEST(MonteCarlo, HandlesDegenerateDiffusionExactly) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();
    spec.volatility = 0.0;

    MonteCarloConfig config{};
    config.num_paths = 1'000;
    config.num_threads = 2;

    const MonteCarloResult result = pricer.monte_carlo_price(spec, config);
    EXPECT_NEAR(result.price, pricer.price(spec), 1e-10);
    EXPECT_DOUBLE_EQ(result.standard_error, 0.0);
}

TEST(MonteCarlo, PricesACustomTerminalPayoff) {
    const OptionPricer pricer;
    OptionSpec spec = atm_call();

    MonteCarloConfig config{};
    config.num_paths = 200'000;
    config.num_threads = 2;
    config.seed = 4242;

    // A digital (cash-or-nothing) call paying 1.0 above the strike has the
    // closed form exp(-r T) * N(d2).
    const double strike = spec.strike;
    const MonteCarloResult result = pricer.monte_carlo(
        spec, config, [strike](double terminal) { return terminal > strike ? 1.0 : 0.0; });

    const double analytic = std::exp(-spec.rate_domestic * spec.time_to_expiry) *
                            math::norm_cdf(pricer.evaluate(spec).d2);
    EXPECT_NEAR(result.price, analytic, 4.0 * result.standard_error + 1e-4);
}

TEST(MonteCarlo, RejectsAZeroPathBudget) {
    const OptionPricer pricer;
    MonteCarloConfig config{};
    config.num_paths = 0;
    EXPECT_THROW(static_cast<void>(pricer.monte_carlo_price(atm_call(), config)),
                 std::invalid_argument);
}

// =============================================================================
// Module 2: IPV
// =============================================================================

TEST(IPVEngineTest, ComputesSignedVarianceAndPercentage) {
    const IPVEngine engine(0.01);
    IPVInput input{};
    input.trade_id = "FX-001";
    input.fo_value = 1'050'000.0;
    input.independent_value = 1'000'000.0;

    const IPVResult result = engine.evaluate(input);
    EXPECT_DOUBLE_EQ(result.variance, 50'000.0);
    EXPECT_DOUBLE_EQ(result.absolute_variance, 50'000.0);
    ASSERT_TRUE(result.variance_pct.has_value());
    EXPECT_DOUBLE_EQ(*result.variance_pct, 0.05);
    EXPECT_TRUE(result.breached);
}

TEST(IPVEngineTest, DoesNotFlagWithinTolerance) {
    const IPVEngine engine(0.05);
    IPVInput input{};
    input.trade_id = "FX-002";
    input.fo_value = 1'020'000.0;
    input.independent_value = 1'000'000.0;

    const IPVResult result = engine.evaluate(input);
    ASSERT_TRUE(result.variance_pct.has_value());
    EXPECT_DOUBLE_EQ(*result.variance_pct, 0.02);
    EXPECT_FALSE(result.breached);
}

TEST(IPVEngineTest, TreatsTheToleranceAsAStrictThreshold) {
    const IPVEngine engine(0.05);
    IPVInput input{};
    input.trade_id = "FX-EDGE";
    input.fo_value = 105.0;
    input.independent_value = 100.0;  // Exactly 5%.
    EXPECT_FALSE(engine.evaluate(input).breached);
}

TEST(IPVEngineTest, PerTradeToleranceOverridesTheDefault) {
    const IPVEngine engine(0.10);
    IPVInput input{};
    input.trade_id = "FX-003";
    input.fo_value = 103.0;
    input.independent_value = 100.0;
    input.tolerance = 0.01;

    const IPVResult result = engine.evaluate(input);
    EXPECT_DOUBLE_EQ(result.tolerance_applied, 0.01);
    EXPECT_TRUE(result.breached);
}

TEST(IPVEngineTest, UsesTheAbsoluteConsensusMarkAsDenominator) {
    const IPVEngine engine(0.01);
    IPVInput input{};
    input.trade_id = "IRS-NEG";
    input.fo_value = -1'100'000.0;   // Short position marked further out.
    input.independent_value = -1'000'000.0;

    const IPVResult result = engine.evaluate(input);
    ASSERT_TRUE(result.variance_pct.has_value());
    EXPECT_DOUBLE_EQ(*result.variance_pct, 0.10);
    EXPECT_TRUE(result.breached) << "a 10% variance on a negative mark must still be flagged";
}

TEST(IPVEngineTest, EscalatesNonZeroVarianceAgainstAZeroMark) {
    const IPVEngine engine(0.01);
    IPVInput input{};
    input.trade_id = "FX-ZERO";
    input.fo_value = 250.0;
    input.independent_value = 0.0;

    const IPVResult result = engine.evaluate(input);
    EXPECT_FALSE(result.variance_pct.has_value());
    EXPECT_TRUE(result.breached);
}

TEST(IPVEngineTest, DoesNotEscalateAMatchedZeroMark) {
    const IPVEngine engine(0.01);
    IPVInput input{};
    input.trade_id = "FX-ZERO-MATCH";
    input.fo_value = 0.0;
    input.independent_value = 0.0;

    const IPVResult result = engine.evaluate(input);
    EXPECT_FALSE(result.variance_pct.has_value());
    EXPECT_FALSE(result.breached);
}

TEST(IPVEngineTest, RejectsInvalidConfiguration) {
    EXPECT_THROW(IPVEngine(-0.01), IPVConfigurationError);
    EXPECT_THROW(IPVEngine(std::numeric_limits<double>::quiet_NaN()), IPVConfigurationError);

    const IPVEngine engine(0.01);
    IPVInput input{};
    input.trade_id = "BAD";
    input.fo_value = std::numeric_limits<double>::infinity();
    input.independent_value = 100.0;
    EXPECT_THROW(engine.evaluate(input), IPVConfigurationError);
}

TEST(IPVEngineTest, ProcessesAPortfolioAndRanksBreaches) {
    IPVEngine engine(0.02);
    const std::vector<IPVInput> portfolio{
        {"FX-001", 1'050'000.0, 1'000'000.0, std::nullopt, "FX-OPTIONS"},   // 5%  breach
        {"FX-002", 500'500.0, 500'000.0, std::nullopt, "FX-OPTIONS"},       // 0.1% ok
        {"IRS-010", 2'400'000.0, 2'000'000.0, std::nullopt, "RATES"},       // 20%  breach
        {"IRS-011", -750'000.0, -745'000.0, std::nullopt, "RATES"},         // 0.67% ok
    };

    const std::vector<IPVResult> results = engine.process_portfolio(portfolio);
    ASSERT_EQ(results.size(), 4U);
    EXPECT_EQ(results[0].trade_id, "FX-001");  // Input order preserved.

    const std::vector<IPVResult> breaches = engine.get_breaches();
    ASSERT_EQ(breaches.size(), 2U);
    EXPECT_EQ(breaches[0].trade_id, "IRS-010");  // Largest absolute variance first.
    EXPECT_EQ(breaches[1].trade_id, "FX-001");

    const IPVSummary summary = engine.summary();
    EXPECT_EQ(summary.trade_count, 4U);
    EXPECT_EQ(summary.breach_count, 2U);
    EXPECT_DOUBLE_EQ(summary.net_variance, 50'000.0 + 500.0 + 400'000.0 - 5'000.0);
    EXPECT_DOUBLE_EQ(summary.gross_absolute_variance, 50'000.0 + 500.0 + 400'000.0 + 5'000.0);
    EXPECT_DOUBLE_EQ(summary.largest_absolute_variance, 400'000.0);
    EXPECT_EQ(summary.largest_variance_trade_id, "IRS-010");
    EXPECT_DOUBLE_EQ(summary.breach_rate(), 0.5);
}

TEST(IPVEngineTest, AggregatesByBook) {
    IPVEngine engine(0.02);
    const std::vector<IPVInput> portfolio{
        {"FX-001", 1'050'000.0, 1'000'000.0, std::nullopt, "FX-OPTIONS"},
        {"FX-002", 500'500.0, 500'000.0, std::nullopt, "FX-OPTIONS"},
        {"IRS-010", 2'400'000.0, 2'000'000.0, std::nullopt, "RATES"},
    };
    engine.process_portfolio(portfolio);

    const auto by_book = engine.summary_by_book();
    ASSERT_EQ(by_book.size(), 2U);
    EXPECT_EQ(by_book.at("FX-OPTIONS").trade_count, 2U);
    EXPECT_EQ(by_book.at("FX-OPTIONS").breach_count, 1U);
    EXPECT_DOUBLE_EQ(by_book.at("RATES").net_variance, 400'000.0);
}

TEST(IPVEngineTest, LooksUpAndClearsStoredResults) {
    IPVEngine engine(0.02);
    const std::vector<IPVInput> portfolio{
        {"FX-001", 1'050'000.0, 1'000'000.0, std::nullopt, "FX-OPTIONS"}};
    engine.process_portfolio(portfolio);

    ASSERT_TRUE(engine.find("FX-001").has_value());
    EXPECT_FALSE(engine.find("NOT-A-TRADE").has_value());

    engine.clear();
    EXPECT_TRUE(engine.results().empty());
    EXPECT_TRUE(engine.get_breaches().empty());
    EXPECT_EQ(engine.summary().trade_count, 0U);
}

TEST(IPVEngineTest, ProcessingAPortfolioReplacesThePreviousRun) {
    IPVEngine engine(0.02);
    const std::vector<IPVInput> first{{"A", 110.0, 100.0, std::nullopt, "BOOK"}};
    const std::vector<IPVInput> second{{"B", 101.0, 100.0, std::nullopt, "BOOK"}};

    engine.process_portfolio(first);
    engine.process_portfolio(second);

    EXPECT_EQ(engine.results().size(), 1U);
    EXPECT_EQ(engine.results().front().trade_id, "B");
    EXPECT_TRUE(engine.get_breaches().empty());
}

TEST(IPVEngineTest, AcceptsAnEmptyPortfolio) {
    IPVEngine engine(0.02);
    const std::vector<IPVInput> empty;
    EXPECT_TRUE(engine.process_portfolio(empty).empty());
    EXPECT_EQ(engine.summary().trade_count, 0U);
    EXPECT_DOUBLE_EQ(engine.summary().breach_rate(), 0.0);
}

// =============================================================================
// Module 3: reserves and PVA
// =============================================================================

TEST(ReserveCalculatorTest, ComputesTheBidOfferReserve) {
    // 0.5 * 1,000,000 * 0.0010 = 500. The tolerance absorbs the binary
    // representation of the quoted prices, not any error in the formula.
    EXPECT_NEAR(ReserveCalculator::compute_bid_offer_reserve(1'000'000.0, 1.0995, 1.1005), 500.0,
                1e-9);
    EXPECT_DOUBLE_EQ(ReserveCalculator::compute_bid_offer_reserve(2'000'000.0, 99.0, 100.0),
                     1'000'000.0);
}

TEST(ReserveCalculatorTest, ChargesShortPositionsTheSameAsLongOnes) {
    const double long_reserve =
        ReserveCalculator::compute_bid_offer_reserve(1'000'000.0, 1.0995, 1.1005);
    const double short_reserve =
        ReserveCalculator::compute_bid_offer_reserve(-1'000'000.0, 1.0995, 1.1005);
    EXPECT_DOUBLE_EQ(long_reserve, short_reserve);
}

TEST(ReserveCalculatorTest, ReturnsZeroForAZeroSpreadOrZeroPosition) {
    EXPECT_DOUBLE_EQ(ReserveCalculator::compute_bid_offer_reserve(1'000'000.0, 1.1, 1.1), 0.0);
    EXPECT_DOUBLE_EQ(ReserveCalculator::compute_bid_offer_reserve(0.0, 1.0995, 1.1005), 0.0);
}

TEST(ReserveCalculatorTest, RejectsCrossedMarketsAndNonFiniteInputs) {
    EXPECT_THROW(static_cast<void>(
                     ReserveCalculator::compute_bid_offer_reserve(1'000.0, 1.1005, 1.0995)),
                 ReserveInputError);
    EXPECT_THROW(static_cast<void>(ReserveCalculator::compute_bid_offer_reserve(
                     std::numeric_limits<double>::quiet_NaN(), 1.0, 1.1)),
                 ReserveInputError);
}

TEST(ReserveCalculatorTest, ValidatesPositionFields) {
    Position position{};
    position.trade_id = "P-1";
    position.position_size = 1'000.0;
    position.bid = 1.0;
    position.ask = 1.1;

    position.unearned_spread_fraction = 1.5;
    EXPECT_THROW(position.validate(), ReserveInputError);

    position.unearned_spread_fraction = 0.5;
    position.close_out_cost_rate = -0.01;
    EXPECT_THROW(position.validate(), ReserveInputError);

    position.close_out_cost_rate = 0.01;
    position.liquidity_horizon_factor = -1.0;
    EXPECT_THROW(position.validate(), ReserveInputError);

    position.liquidity_horizon_factor = 1.0;
    EXPECT_NO_THROW(position.validate());
}

TEST(ReserveCalculatorTest, ComputesMarketUnearnedSpreadPva) {
    Position position{};
    position.trade_id = "P-MUS";
    position.position_size = 2'000'000.0;
    position.bid = 1.0990;
    position.ask = 1.1010;              // 0.0020 spread.
    position.unearned_spread_fraction = 0.25;

    // 2,000,000 * 0.0010 * 0.25 = 500.
    EXPECT_NEAR(ReserveCalculator::compute_market_unearned_spread_pva(position), 500.0, 1e-9);
}

TEST(ReserveCalculatorTest, ComputesCloseOutCostPva) {
    Position position{};
    position.trade_id = "P-CO";
    position.position_size = 1'000'000.0;
    position.bid = 0.999;
    position.ask = 1.001;               // Mid 1.0.
    position.close_out_cost_rate = 0.0005;
    position.liquidity_horizon_factor = 2.0;

    // 1,000,000 * 1.0 * 0.0005 * 2.0 = 1,000.
    EXPECT_NEAR(ReserveCalculator::compute_close_out_cost_pva(position), 1'000.0, 1e-9);
}

TEST(ReserveCalculatorTest, PrefersAnExplicitReferencePriceOverTheMid) {
    Position position{};
    position.trade_id = "P-REF";
    position.position_size = 1'000'000.0;
    position.bid = 0.999;
    position.ask = 1.001;
    position.close_out_cost_rate = 0.001;
    position.reference_price = 2.0;

    EXPECT_NEAR(ReserveCalculator::compute_close_out_cost_pva(position), 2'000.0, 1e-9);
}

TEST(ReserveCalculatorTest, BreaksDownAllComponentsForOnePosition) {
    Position position{};
    position.trade_id = "P-ALL";
    position.book = "FX-OPTIONS";
    position.category = "FX Options";
    position.position_size = 1'000'000.0;
    position.bid = 0.998;
    position.ask = 1.002;                  // Spread 0.004, mid 1.0.
    position.unearned_spread_fraction = 0.5;
    position.close_out_cost_rate = 0.0002;
    position.liquidity_horizon_factor = 1.0;

    const ReserveBreakdown breakdown = ReserveCalculator::compute_position_reserves(position);
    EXPECT_NEAR(breakdown.bid_offer, 2'000.0, 1e-9);
    EXPECT_NEAR(breakdown.market_unearned_spread_pva, 1'000.0, 1e-9);
    EXPECT_NEAR(breakdown.close_out_cost_pva, 200.0, 1e-9);
    EXPECT_NEAR(breakdown.total_pva(), 1'200.0, 1e-9);
    EXPECT_NEAR(breakdown.total(), 3'200.0, 1e-9);
    EXPECT_EQ(breakdown.position_count, 1U);
}

namespace {

std::vector<Position> sample_portfolio() {
    Position fx1{};
    fx1.trade_id = "FX-1";
    fx1.book = "FX-OPTIONS";
    fx1.category = "FX Options";
    fx1.position_size = 1'000'000.0;
    fx1.bid = 0.998;
    fx1.ask = 1.002;
    fx1.unearned_spread_fraction = 0.5;
    fx1.close_out_cost_rate = 0.0002;

    Position fx2{};
    fx2.trade_id = "FX-2";
    fx2.book = "FX-OPTIONS";
    fx2.category = "FX Options";
    fx2.position_size = -500'000.0;
    fx2.bid = 0.999;
    fx2.ask = 1.001;
    fx2.unearned_spread_fraction = 0.25;
    fx2.close_out_cost_rate = 0.0001;

    Position rates{};
    rates.trade_id = "IRS-1";
    rates.book = "RATES";
    rates.category = "Rates";
    rates.position_size = 250'000.0;
    rates.bid = 99.95;
    rates.ask = 100.05;
    rates.unearned_spread_fraction = 0.1;
    rates.close_out_cost_rate = 0.00005;

    return {fx1, fx2, rates};
}

}  // namespace

TEST(ReserveCalculatorTest, AggregatesAcrossThePortfolio) {
    const std::vector<Position> positions = sample_portfolio();
    const ReserveBreakdown total =
        ReserveCalculator::compute_portfolio_reserves(std::span<const Position>(positions));

    double expected_bid_offer = 0.0;
    double expected_mus = 0.0;
    double expected_close_out = 0.0;
    for (const Position& position : positions) {
        expected_bid_offer += ReserveCalculator::compute_bid_offer_reserve(position);
        expected_mus += ReserveCalculator::compute_market_unearned_spread_pva(position);
        expected_close_out += ReserveCalculator::compute_close_out_cost_pva(position);
    }

    EXPECT_EQ(total.position_count, 3U);
    EXPECT_NEAR(total.bid_offer, expected_bid_offer, 1e-9);
    EXPECT_NEAR(total.market_unearned_spread_pva, expected_mus, 1e-9);
    EXPECT_NEAR(total.close_out_cost_pva, expected_close_out, 1e-9);
    EXPECT_NEAR(total.total(), expected_bid_offer + expected_mus + expected_close_out, 1e-9);
}

TEST(ReserveCalculatorTest, AggregatesByBookAndCategory) {
    const std::vector<Position> positions = sample_portfolio();

    const auto by_book = ReserveCalculator::aggregate_by_book(std::span<const Position>(positions));
    ASSERT_EQ(by_book.size(), 2U);
    EXPECT_EQ(by_book.at("FX-OPTIONS").position_count, 2U);
    EXPECT_EQ(by_book.at("RATES").position_count, 1U);
    // FX-1 2,000 + FX-2 500.
    EXPECT_NEAR(by_book.at("FX-OPTIONS").bid_offer, 2'500.0, 1e-9);
    // 0.5 * 250,000 * 0.10.
    EXPECT_NEAR(by_book.at("RATES").bid_offer, 12'500.0, 1e-9);

    const auto by_category =
        ReserveCalculator::aggregate_by_category(std::span<const Position>(positions));
    ASSERT_EQ(by_category.size(), 2U);
    EXPECT_NEAR(by_category.at("FX Options").bid_offer, 2'500.0, 1e-9);
}

TEST(ReserveCalculatorTest, RanksBooksByTotalReserve) {
    const std::vector<Position> positions = sample_portfolio();
    const std::vector<KeyedReserve> ranked =
        ReserveCalculator::ranked_by_book(std::span<const Position>(positions));

    ASSERT_EQ(ranked.size(), 2U);
    EXPECT_EQ(ranked.front().key, "RATES");
    EXPECT_GE(ranked.front().breakdown.total(), ranked.back().breakdown.total());
}

TEST(ReserveCalculatorTest, AggregatesAnEmptyPortfolioToZero) {
    const std::vector<Position> empty;
    const ReserveBreakdown total =
        ReserveCalculator::compute_portfolio_reserves(std::span<const Position>(empty));
    EXPECT_EQ(total.position_count, 0U);
    EXPECT_DOUBLE_EQ(total.total(), 0.0);
    EXPECT_TRUE(ReserveCalculator::ranked_by_book(std::span<const Position>(empty)).empty());
}

TEST(ReserveCalculatorTest, AggregationPropagatesPositionErrors) {
    std::vector<Position> positions = sample_portfolio();
    positions[1].ask = positions[1].bid - 0.01;  // Crossed market.
    EXPECT_THROW(static_cast<void>(ReserveCalculator::compute_portfolio_reserves(
                     std::span<const Position>(positions))),
                 ReserveInputError);
}

// =============================================================================
// Integration: price -> verify -> reserve
// =============================================================================

TEST(Integration, PricesVerifiesAndReservesAnFxOptionsBook) {
    const OptionPricer pricer;

    // Front Office marks three EURUSD options off its own vol surface.
    struct BookEntry {
        std::string trade_id;
        double strike;
        double fo_vol;
        double consensus_vol;
        double notional;
    };
    const std::vector<BookEntry> book{
        {"FXO-100", 1.10, 0.0850, 0.0852, 10'000'000.0},
        {"FXO-101", 1.15, 0.0910, 0.0990, 5'000'000.0},   // Consensus well above FO.
        {"FXO-102", 1.05, 0.0880, 0.0881, 7'500'000.0},
    };

    OptionSpec base = fx_call();
    std::vector<IPVInput> ipv_inputs;
    std::vector<Position> positions;
    ipv_inputs.reserve(book.size());
    positions.reserve(book.size());

    for (const BookEntry& entry : book) {
        OptionSpec fo_spec = base;
        fo_spec.strike = entry.strike;
        fo_spec.volatility = entry.fo_vol;

        OptionSpec consensus_spec = fo_spec;
        consensus_spec.volatility = entry.consensus_vol;

        const double fo_value = pricer.price(fo_spec) * entry.notional;
        const double consensus_value = pricer.price(consensus_spec) * entry.notional;

        IPVInput ipv{};
        ipv.trade_id = entry.trade_id;
        ipv.fo_value = fo_value;
        ipv.independent_value = consensus_value;
        ipv.book = "FX-OPTIONS";
        ipv_inputs.push_back(ipv);

        // The consensus mark is bracketed by a two-vol-point market.
        OptionSpec bid_spec = consensus_spec;
        OptionSpec ask_spec = consensus_spec;
        bid_spec.volatility = entry.consensus_vol - 0.001;
        ask_spec.volatility = entry.consensus_vol + 0.001;

        Position position{};
        position.trade_id = entry.trade_id;
        position.book = "FX-OPTIONS";
        position.category = "FX Options";
        position.position_size = entry.notional;
        position.bid = pricer.price(bid_spec);
        position.ask = pricer.price(ask_spec);
        position.unearned_spread_fraction = 0.5;
        position.close_out_cost_rate = 0.0001;
        position.liquidity_horizon_factor = 1.0;
        positions.push_back(position);
    }

    // --- Verification -------------------------------------------------------
    IPVEngine engine(0.02);
    const std::vector<IPVResult> results = engine.process_portfolio(ipv_inputs);
    ASSERT_EQ(results.size(), 3U);

    const std::vector<IPVResult> breaches = engine.get_breaches();
    ASSERT_EQ(breaches.size(), 1U) << "only the mispriced strike should breach";
    EXPECT_EQ(breaches.front().trade_id, "FXO-101");
    EXPECT_LT(breaches.front().variance, 0.0) << "FO is below consensus on this trade";

    // --- Calibration round-trip on the breaching trade -----------------------
    const IPVResult breach = breaches.front();
    OptionSpec implied_spec = base;
    implied_spec.strike = 1.15;
    implied_spec.volatility = 0.0;
    const std::optional<double> implied =
        pricer.calibrate_volatility(implied_spec, breach.independent_value / 5'000'000.0);
    ASSERT_TRUE(implied.has_value());
    EXPECT_NEAR(*implied, 0.0990, 1e-6) << "consensus price must imply the consensus vol";

    // --- Reserves -----------------------------------------------------------
    const ReserveBreakdown reserves =
        ReserveCalculator::compute_portfolio_reserves(std::span<const Position>(positions));
    EXPECT_EQ(reserves.position_count, 3U);
    EXPECT_GT(reserves.bid_offer, 0.0);
    EXPECT_GT(reserves.market_unearned_spread_pva, 0.0);
    EXPECT_GT(reserves.close_out_cost_pva, 0.0);
    EXPECT_NEAR(reserves.total(),
                reserves.bid_offer + reserves.market_unearned_spread_pva +
                    reserves.close_out_cost_pva,
                1e-9);

    // Half the spread is the bid-offer reserve; the unearned half of that is MUS.
    EXPECT_NEAR(reserves.market_unearned_spread_pva, 0.5 * reserves.bid_offer, 1e-9);

    const auto by_book = ReserveCalculator::aggregate_by_book(std::span<const Position>(positions));
    ASSERT_EQ(by_book.size(), 1U);
    EXPECT_NEAR(by_book.at("FX-OPTIONS").total(), reserves.total(), 1e-9);
}

TEST(Integration, MonteCarloAgreesWithTheAnalyticalMarkUsedForIpv) {
    const OptionPricer pricer;
    OptionSpec spec = fx_call();
    const double notional = 10'000'000.0;

    MonteCarloConfig config{};
    config.num_paths = 250'000;
    config.num_threads = 4;
    config.seed = 31337;

    const double analytic_value = pricer.price(spec) * notional;
    const MonteCarloResult mc = pricer.monte_carlo_price(spec, config);
    const double mc_value = mc.price * notional;

    IPVEngine engine(0.01);
    IPVInput input{};
    input.trade_id = "FXO-MC";
    input.fo_value = analytic_value;
    input.independent_value = mc_value;
    input.book = "FX-OPTIONS";

    const IPVResult result = engine.evaluate(input);
    ASSERT_TRUE(result.variance_pct.has_value());
    EXPECT_FALSE(result.breached)
        << "Monte Carlo and the closed form should agree inside a 1% tolerance; variance was "
        << *result.variance_pct;
}
