// =============================================================================
//  OptionPricer.hpp
//  Module 1: Options Pricing & Calibration Core
//
//  Analytical Black-Scholes / Garman-Kohlhagen pricing for European FX options,
//  Newton-Raphson implied-volatility calibration, and a multi-threaded Monte
//  Carlo engine built on std::jthread.
//
//  Convention note: FX options are priced under Garman-Kohlhagen, i.e. Black-
//  Scholes with a continuous foreign-currency yield r_f acting as a dividend
//  yield. Setting r_f = 0 collapses the model to textbook Black-Scholes, so the
//  same class serves equity-style and FX-style payoffs.
// =============================================================================
#ifndef VALUATION_ENGINE_OPTION_PRICER_HPP
#define VALUATION_ENGINE_OPTION_PRICER_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace valuation {

// -----------------------------------------------------------------------------
// Errors
// -----------------------------------------------------------------------------

/// Thrown when an option specification is economically meaningless
/// (non-positive spot/strike, negative variance inputs, and so on).
class InvalidOptionSpec : public std::invalid_argument {
public:
    explicit InvalidOptionSpec(const std::string& what_arg)
        : std::invalid_argument(what_arg) {}
};

// -----------------------------------------------------------------------------
// Concepts
// -----------------------------------------------------------------------------

/// Numeric type usable by the closed-form maths helpers.
template <typename T>
concept Real = std::floating_point<T>;

/// A terminal payoff: callable with a terminal spot level, yielding a value.
template <typename F>
concept TerminalPayoff = requires(F f, double terminal_spot) {
    { f(terminal_spot) } -> std::convertible_to<double>;
};

// -----------------------------------------------------------------------------
// Maths helpers
// -----------------------------------------------------------------------------

namespace math {

/// Standard normal probability density.
template <Real T>
[[nodiscard]] constexpr T norm_pdf(T x) noexcept {
    constexpr T inv_sqrt_2pi = static_cast<T>(0.39894228040143267793994605993438);
    return inv_sqrt_2pi * std::exp(T{-0.5} * x * x);
}

/// Standard normal cumulative distribution, via erfc for tail accuracy.
template <Real T>
[[nodiscard]] T norm_cdf(T x) noexcept {
    return T{0.5} * std::erfc(-x * std::numbers::sqrt2_v<T> / T{2});
}

}  // namespace math

// -----------------------------------------------------------------------------
// Value types
// -----------------------------------------------------------------------------

enum class OptionType { Call, Put };

[[nodiscard]] std::string to_string(OptionType type);

/// A European FX option. Rates and volatility are continuously-compounded
/// annualised decimals (0.05 == 5%); time_to_expiry is in years.
struct OptionSpec {
    double spot{0.0};             ///< S: spot FX rate (domestic per foreign).
    double strike{0.0};           ///< K: strike.
    double time_to_expiry{0.0};   ///< T: year fraction to expiry (>= 0).
    double rate_domestic{0.0};    ///< r_d: domestic (discounting) rate.
    double rate_foreign{0.0};     ///< r_f: foreign rate (dividend-yield analogue).
    double volatility{0.0};       ///< sigma: annualised volatility (>= 0).
    OptionType type{OptionType::Call};

    /// Throws InvalidOptionSpec if the specification cannot be priced.
    void validate() const;
};

/// Price plus the first- and second-order risk the IPV/reserve process needs.
struct PricingResult {
    double price{0.0};
    double delta{0.0};   ///< dV/dS.
    double gamma{0.0};   ///< d2V/dS2.
    double vega{0.0};    ///< dV/dsigma, per 1.00 of vol (divide by 100 for per-vol-point).
    double theta{0.0};   ///< dV/dt, per year (negative for long premium, typically).
    double d1{0.0};
    double d2{0.0};
};

/// Tunables for the Newton-Raphson implied-vol solve.
struct CalibrationSettings {
    int max_iterations{100};
    double price_tolerance{1.0e-10};  ///< Absolute price convergence criterion.
    double vol_tolerance{1.0e-12};    ///< Absolute step-size convergence criterion.
    double min_vega{1.0e-12};         ///< Below this vega the solve is ill-conditioned.
    double min_volatility{1.0e-8};    ///< Lower clamp on the iterate.
    double max_volatility{5.0};       ///< Upper clamp on the iterate (500% vol).
};

/// Monte Carlo run configuration.
struct MonteCarloConfig {
    std::size_t num_paths{100'000};  ///< Total terminal draws requested.
    unsigned num_threads{0};         ///< 0 => std::thread::hardware_concurrency().
    std::uint64_t seed{42};          ///< Base seed; runs are reproducible per (seed, threads).
    bool antithetic{true};           ///< Pair each draw with its mirror to cut variance.
};

/// Monte Carlo output, including the sampling error so a desk can judge it.
struct MonteCarloResult {
    double price{0.0};
    double standard_error{0.0};
    std::size_t paths{0};
    unsigned threads_used{0};

    /// Half-width of the ~95% confidence interval around `price`.
    [[nodiscard]] double confidence_interval_95() const noexcept {
        return 1.959963984540054 * standard_error;
    }
};

// -----------------------------------------------------------------------------
// OptionPricer
// -----------------------------------------------------------------------------

class OptionPricer {
public:
    OptionPricer() = default;
    explicit OptionPricer(CalibrationSettings settings) noexcept
        : settings_(settings) {}

    [[nodiscard]] const CalibrationSettings& settings() const noexcept { return settings_; }

    /// Closed-form price. Throws InvalidOptionSpec on a malformed spec.
    [[nodiscard]] double price(const OptionSpec& spec) const;

    /// Closed-form price and Greeks in a single pass.
    [[nodiscard]] PricingResult evaluate(const OptionSpec& spec) const;

    /// Newton-Raphson implied volatility, safeguarded by bisection on the
    /// bracket [min_volatility, max_volatility] wherever vega is too small for
    /// the Newton step to be trusted.
    /// `spec.volatility`, when strictly positive, seeds the iteration; otherwise a
    /// Brenner-Subrahmanyam approximation is used. Returns std::nullopt when the
    /// target price is outside the attainable range (below the discounted
    /// intrinsic or above the price at max_volatility) or when the iteration
    /// fails to converge within `settings().max_iterations`.
    [[nodiscard]] std::optional<double> calibrate_volatility(const OptionSpec& spec,
                                                             double target_price) const;

    /// Multi-threaded Monte Carlo price of the option's own payoff.
    [[nodiscard]] MonteCarloResult monte_carlo_price(const OptionSpec& spec,
                                                     const MonteCarloConfig& config) const;

    /// Multi-threaded Monte Carlo under an arbitrary terminal payoff, discounted
    /// at the domestic rate. Paths are split across std::jthread workers, each
    /// with an independently seeded generator, so the result is reproducible for
    /// a given (seed, num_threads, num_paths, antithetic) tuple.
    template <TerminalPayoff F>
    [[nodiscard]] MonteCarloResult monte_carlo(const OptionSpec& spec,
                                               const MonteCarloConfig& config,
                                               F payoff) const;

    /// Intrinsic value discounted to today; the lower no-arbitrage bound.
    [[nodiscard]] static double discounted_intrinsic(const OptionSpec& spec);

private:
    CalibrationSettings settings_{};
};

// -----------------------------------------------------------------------------
// Template implementation
// -----------------------------------------------------------------------------

namespace detail {

/// Per-worker accumulation of discounted payoffs.
///
/// `observations` counts statistically independent samples, which is not the
/// same as `paths`: an antithetic pair is one observation drawn from two paths.
/// Accumulating the pair average rather than the two legs separately is what
/// makes the reported standard error reflect the negative correlation the
/// technique introduces — summing the legs independently would report the
/// crude-sampling error and hide the variance reduction entirely.
struct PathAccumulator {
    double sum{0.0};
    double sum_squares{0.0};
    std::size_t observations{0};
    std::size_t paths{0};
};

/// Splits `total` items into `buckets` near-equal chunks (remainder to the front).
[[nodiscard]] inline std::size_t chunk_size(std::size_t total, unsigned buckets, unsigned index) noexcept {
    const std::size_t base = total / buckets;
    const std::size_t remainder = total % buckets;
    return base + (static_cast<std::size_t>(index) < remainder ? 1U : 0U);
}

}  // namespace detail

template <TerminalPayoff F>
MonteCarloResult OptionPricer::monte_carlo(const OptionSpec& spec,
                                           const MonteCarloConfig& config,
                                           F payoff) const {
    spec.validate();
    if (config.num_paths == 0) {
        throw std::invalid_argument("MonteCarloConfig::num_paths must be greater than zero");
    }

    unsigned threads = config.num_threads;
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
        if (threads == 0) {
            threads = 1U;
        }
    }
    if (static_cast<std::size_t>(threads) > config.num_paths) {
        threads = static_cast<unsigned>(config.num_paths);
    }

    const double discount = std::exp(-spec.rate_domestic * spec.time_to_expiry);

    // Degenerate diffusion: the terminal spot is deterministic.
    if (spec.time_to_expiry <= 0.0 || spec.volatility <= 0.0) {
        const double forward =
            spec.spot * std::exp((spec.rate_domestic - spec.rate_foreign) * spec.time_to_expiry);
        const double value = discount * static_cast<double>(payoff(forward));
        return MonteCarloResult{value, 0.0, config.num_paths, threads};
    }

    const double drift =
        (spec.rate_domestic - spec.rate_foreign - 0.5 * spec.volatility * spec.volatility) *
        spec.time_to_expiry;
    const double diffusion = spec.volatility * std::sqrt(spec.time_to_expiry);

    std::vector<detail::PathAccumulator> partials(threads);

    {
        std::vector<std::jthread> workers;
        workers.reserve(threads);

        for (unsigned t = 0; t < threads; ++t) {
            const std::size_t paths_for_worker = detail::chunk_size(config.num_paths, threads, t);
            // No std::stop_token parameter here on purpose: ~jthread requests a
            // stop before joining, so a worker that polled the token would
            // abandon its remaining paths as soon as the first thread was
            // destroyed, silently truncating the sample.
            workers.emplace_back([&, t, paths_for_worker]() {
                detail::PathAccumulator local{};
                std::seed_seq seq{static_cast<std::uint32_t>(config.seed & 0xFFFFFFFFULL),
                                  static_cast<std::uint32_t>(config.seed >> 32),
                                  static_cast<std::uint32_t>(t)};
                std::mt19937_64 engine(seq);
                std::normal_distribution<double> gaussian(0.0, 1.0);

                std::size_t drawn = 0;
                while (drawn < paths_for_worker) {
                    const double z = gaussian(engine);
                    const double terminal = spec.spot * std::exp(drift + diffusion * z);
                    double observation = discount * static_cast<double>(payoff(terminal));
                    ++drawn;
                    ++local.paths;

                    if (config.antithetic && drawn < paths_for_worker) {
                        const double mirrored = spec.spot * std::exp(drift - diffusion * z);
                        const double mirrored_value =
                            discount * static_cast<double>(payoff(mirrored));
                        observation = 0.5 * (observation + mirrored_value);
                        ++drawn;
                        ++local.paths;
                    }

                    local.sum += observation;
                    local.sum_squares += observation * observation;
                    ++local.observations;
                }
                partials[t] = local;
            });
        }
    }  // jthreads join here.

    double sum = 0.0;
    double sum_squares = 0.0;
    std::size_t observations = 0;
    std::size_t paths = 0;
    for (const auto& partial : partials) {
        sum += partial.sum;
        sum_squares += partial.sum_squares;
        observations += partial.observations;
        paths += partial.paths;
    }

    if (observations == 0) {
        return MonteCarloResult{0.0, 0.0, 0, threads};
    }

    const double n = static_cast<double>(observations);
    const double mean = sum / n;
    double standard_error = 0.0;
    if (observations > 1) {
        const double variance = std::max(0.0, (sum_squares - n * mean * mean) / (n - 1.0));
        standard_error = std::sqrt(variance / n);
    }
    return MonteCarloResult{mean, standard_error, paths, threads};
}

}  // namespace valuation

#endif  // VALUATION_ENGINE_OPTION_PRICER_HPP
