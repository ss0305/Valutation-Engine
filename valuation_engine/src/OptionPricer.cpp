// =============================================================================
//  OptionPricer.cpp — Module 1 implementation.
// =============================================================================
#include "OptionPricer.hpp"

#include <algorithm>
#include <sstream>

namespace valuation {

std::string to_string(OptionType type) {
    return type == OptionType::Call ? "Call" : "Put";
}

void OptionSpec::validate() const {
    std::ostringstream problem;
    if (!(spot > 0.0) || !std::isfinite(spot)) {
        problem << "spot must be finite and strictly positive (got " << spot << ")";
    } else if (!(strike > 0.0) || !std::isfinite(strike)) {
        problem << "strike must be finite and strictly positive (got " << strike << ")";
    } else if (time_to_expiry < 0.0 || !std::isfinite(time_to_expiry)) {
        problem << "time_to_expiry must be finite and non-negative (got " << time_to_expiry << ")";
    } else if (volatility < 0.0 || !std::isfinite(volatility)) {
        problem << "volatility must be finite and non-negative (got " << volatility << ")";
    } else if (!std::isfinite(rate_domestic) || !std::isfinite(rate_foreign)) {
        problem << "rates must be finite";
    } else {
        return;
    }
    throw InvalidOptionSpec("Invalid OptionSpec: " + problem.str());
}

double OptionPricer::discounted_intrinsic(const OptionSpec& spec) {
    spec.validate();
    const double df_domestic = std::exp(-spec.rate_domestic * spec.time_to_expiry);
    const double df_foreign = std::exp(-spec.rate_foreign * spec.time_to_expiry);
    const double forward_pv_spot = spec.spot * df_foreign;
    const double pv_strike = spec.strike * df_domestic;
    return spec.type == OptionType::Call ? std::max(0.0, forward_pv_spot - pv_strike)
                                         : std::max(0.0, pv_strike - forward_pv_spot);
}

PricingResult OptionPricer::evaluate(const OptionSpec& spec) const {
    spec.validate();

    PricingResult result{};
    const double df_domestic = std::exp(-spec.rate_domestic * spec.time_to_expiry);
    const double df_foreign = std::exp(-spec.rate_foreign * spec.time_to_expiry);

    const double total_vol = spec.volatility * std::sqrt(spec.time_to_expiry);

    // Expired or zero-variance options: value is the discounted intrinsic, the
    // density is a point mass, so gamma/vega vanish and delta is a step.
    if (total_vol <= 0.0) {
        const double forward_pv_spot = spec.spot * df_foreign;
        const double pv_strike = spec.strike * df_domestic;
        const bool call = spec.type == OptionType::Call;
        const bool in_the_money = call ? forward_pv_spot > pv_strike : pv_strike > forward_pv_spot;

        result.price = call ? std::max(0.0, forward_pv_spot - pv_strike)
                            : std::max(0.0, pv_strike - forward_pv_spot);
        result.delta = in_the_money ? (call ? df_foreign : -df_foreign) : 0.0;
        result.gamma = 0.0;
        result.vega = 0.0;
        // Limit of the general theta as sigma -> 0 with the option in the money.
        result.theta = in_the_money ? (call ? (spec.rate_foreign * forward_pv_spot -
                                               spec.rate_domestic * pv_strike)
                                            : (spec.rate_domestic * pv_strike -
                                               spec.rate_foreign * forward_pv_spot))
                                    : 0.0;
        result.d1 = in_the_money ? std::numeric_limits<double>::infinity()
                                 : -std::numeric_limits<double>::infinity();
        result.d2 = result.d1;
        return result;
    }

    const double d1 = (std::log(spec.spot / spec.strike) +
                       (spec.rate_domestic - spec.rate_foreign +
                        0.5 * spec.volatility * spec.volatility) *
                           spec.time_to_expiry) /
                      total_vol;
    const double d2 = d1 - total_vol;

    const double nd1 = math::norm_cdf(d1);
    const double nd2 = math::norm_cdf(d2);
    const double pdf_d1 = math::norm_pdf(d1);

    result.d1 = d1;
    result.d2 = d2;

    if (spec.type == OptionType::Call) {
        result.price = spec.spot * df_foreign * nd1 - spec.strike * df_domestic * nd2;
        result.delta = df_foreign * nd1;
        result.theta = -(spec.spot * df_foreign * pdf_d1 * spec.volatility) /
                           (2.0 * std::sqrt(spec.time_to_expiry)) +
                       spec.rate_foreign * spec.spot * df_foreign * nd1 -
                       spec.rate_domestic * spec.strike * df_domestic * nd2;
    } else {
        result.price = spec.strike * df_domestic * math::norm_cdf(-d2) -
                       spec.spot * df_foreign * math::norm_cdf(-d1);
        result.delta = -df_foreign * math::norm_cdf(-d1);
        result.theta = -(spec.spot * df_foreign * pdf_d1 * spec.volatility) /
                           (2.0 * std::sqrt(spec.time_to_expiry)) -
                       spec.rate_foreign * spec.spot * df_foreign * math::norm_cdf(-d1) +
                       spec.rate_domestic * spec.strike * df_domestic * math::norm_cdf(-d2);
    }

    // Gamma and vega are payoff-independent (identical for calls and puts).
    result.gamma = df_foreign * pdf_d1 / (spec.spot * total_vol);
    result.vega = spec.spot * df_foreign * pdf_d1 * std::sqrt(spec.time_to_expiry);
    return result;
}

double OptionPricer::price(const OptionSpec& spec) const {
    return evaluate(spec).price;
}

std::optional<double> OptionPricer::calibrate_volatility(const OptionSpec& spec,
                                                         double target_price) const {
    spec.validate();
    if (!std::isfinite(target_price) || target_price < 0.0) {
        return std::nullopt;
    }
    if (spec.time_to_expiry <= 0.0) {
        // With no time left there is no volatility to imply.
        return std::nullopt;
    }

    if (settings_.min_volatility <= 0.0 || settings_.max_volatility <= settings_.min_volatility) {
        return std::nullopt;
    }

    OptionSpec working = spec;

    // Option value is strictly increasing in volatility, so the admissible
    // range of quotes is bounded by the prices at the two vol clamps. A target
    // outside that range — including anything below the discounted intrinsic or
    // above the no-arbitrage ceiling — has no implied volatility.
    working.volatility = settings_.min_volatility;
    const double price_at_floor = evaluate(working).price;
    working.volatility = settings_.max_volatility;
    const double price_at_ceiling = evaluate(working).price;

    const double slack = std::max(settings_.price_tolerance, 1.0e-12);
    if (target_price < price_at_floor - slack || target_price > price_at_ceiling + slack) {
        return std::nullopt;
    }
    if (std::abs(target_price - price_at_floor) <= settings_.price_tolerance) {
        return settings_.min_volatility;
    }
    if (std::abs(target_price - price_at_ceiling) <= settings_.price_tolerance) {
        return settings_.max_volatility;
    }

    // Safeguarded Newton-Raphson: Newton drives the iteration, and a bisection
    // step on the maintained bracket rescues it wherever vega is too small for
    // the tangent to be trusted or the step would leave the bracket. Plain
    // Newton stalls on wings where the Brenner-Subrahmanyam seed is poor.
    double lower = settings_.min_volatility;
    double upper = settings_.max_volatility;

    double sigma = spec.volatility;  // Caller-supplied seed, when present.
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        const double df_domestic = std::exp(-spec.rate_domestic * spec.time_to_expiry);
        const double forward =
            spec.spot * std::exp((spec.rate_domestic - spec.rate_foreign) * spec.time_to_expiry);
        const double scale = 0.5 * (forward + spec.strike) * df_domestic;
        constexpr double sqrt_two_pi = 2.5066282746310002;
        sigma = scale > 0.0
                    ? (sqrt_two_pi / std::sqrt(spec.time_to_expiry)) * (target_price / scale)
                    : 0.2;
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            sigma = 0.2;
        }
        // The approximation is an at-the-money result; away from the money it can
        // land arbitrarily close to zero, where vega vanishes. Keep the seed in a
        // range where the tangent carries information.
        sigma = std::clamp(sigma, 0.01, 2.0);
    }
    sigma = std::clamp(sigma, settings_.min_volatility, settings_.max_volatility);

    for (int iteration = 0; iteration < settings_.max_iterations; ++iteration) {
        working.volatility = sigma;
        const PricingResult evaluated = evaluate(working);
        const double diff = evaluated.price - target_price;

        if (std::abs(diff) <= settings_.price_tolerance) {
            return sigma;
        }

        // Tighten the bracket around the root before stepping.
        if (diff > 0.0) {
            upper = sigma;
        } else {
            lower = sigma;
        }

        double next = std::numeric_limits<double>::quiet_NaN();
        if (evaluated.vega >= settings_.min_vega) {
            next = sigma - diff / evaluated.vega;
        }
        if (!std::isfinite(next) || next <= lower || next >= upper) {
            next = 0.5 * (lower + upper);  // Bisection safeguard.
        }

        const double step = std::abs(next - sigma);
        sigma = next;
        if (step <= settings_.vol_tolerance) {
            working.volatility = sigma;
            const double residual = std::abs(evaluate(working).price - target_price);
            return residual <= std::max(settings_.price_tolerance, 1.0e-8)
                       ? std::optional<double>{sigma}
                       : std::nullopt;
        }
    }
    return std::nullopt;
}

MonteCarloResult OptionPricer::monte_carlo_price(const OptionSpec& spec,
                                                 const MonteCarloConfig& config) const {
    const double strike = spec.strike;
    if (spec.type == OptionType::Call) {
        return monte_carlo(spec, config,
                           [strike](double terminal) { return std::max(0.0, terminal - strike); });
    }
    return monte_carlo(spec, config,
                       [strike](double terminal) { return std::max(0.0, strike - terminal); });
}

}  // namespace valuation
