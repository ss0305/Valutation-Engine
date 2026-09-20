// =============================================================================
//  IPVEngine.hpp
//  Module 2: Independent Price Verification
//
//  Compares Front Office model valuations against independent consensus marks,
//  flags tolerance breaches, and summarises the variance for a portfolio.
//
//  Denominator convention: the specification writes the variance percentage as
//  |V_FO - V_Indep| / V_Indep. Derivative valuations are routinely negative, and
//  an unsigned denominator would then produce a negative percentage that can
//  never exceed a positive tolerance — i.e. genuine breaches on short positions
//  would go unflagged. This engine therefore divides by |V_Indep|, which is the
//  standard market-risk convention and agrees with the specification for every
//  positive consensus mark.
// =============================================================================
#ifndef VALUATION_ENGINE_IPV_ENGINE_HPP
#define VALUATION_ENGINE_IPV_ENGINE_HPP

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace valuation {

/// Thrown when IPV configuration is unusable (e.g. a negative tolerance).
class IPVConfigurationError : public std::invalid_argument {
public:
    explicit IPVConfigurationError(const std::string& what_arg)
        : std::invalid_argument(what_arg) {}
};

/// One trade presented for verification.
struct IPVInput {
    std::string trade_id;
    double fo_value{0.0};           ///< V_FO: Front Office fair value.
    double independent_value{0.0};  ///< V_Indep: independent consensus price.
    /// Per-trade tolerance as a decimal fraction (0.01 == 1%). When empty the
    /// engine's default tolerance applies.
    std::optional<double> tolerance{};
    std::string book;               ///< Optional desk/book tag for aggregation.
};

/// Structured variance metrics for a single trade.
struct IPVResult {
    std::string trade_id;
    std::string book;
    double fo_value{0.0};
    double independent_value{0.0};
    double variance{0.0};                   ///< V_FO - V_Indep (signed).
    double absolute_variance{0.0};          ///< |V_FO - V_Indep|.
    /// |V_FO - V_Indep| / |V_Indep|, empty when the consensus mark is zero and
    /// the ratio is undefined.
    std::optional<double> variance_pct{};
    double tolerance_applied{0.0};
    bool breached{false};

    [[nodiscard]] std::string to_string() const;
};

/// Portfolio-level roll-up of a verification run.
struct IPVSummary {
    std::size_t trade_count{0};
    std::size_t breach_count{0};
    double total_fo_value{0.0};
    double total_independent_value{0.0};
    double net_variance{0.0};           ///< Sum of signed variances.
    double gross_absolute_variance{0.0};  ///< Sum of |variance|, i.e. no netting.
    double largest_absolute_variance{0.0};
    std::string largest_variance_trade_id;

    [[nodiscard]] double breach_rate() const noexcept {
        return trade_count == 0 ? 0.0
                                : static_cast<double>(breach_count) /
                                      static_cast<double>(trade_count);
    }
};

// -----------------------------------------------------------------------------
// IPVEngine
// -----------------------------------------------------------------------------

/// Not internally synchronised: one engine instance per verification run.
class IPVEngine {
public:
    /// @param default_tolerance decimal fraction applied when an input carries
    ///        no tolerance of its own. Must be finite and non-negative.
    explicit IPVEngine(double default_tolerance = 0.01);

    [[nodiscard]] double default_tolerance() const noexcept { return default_tolerance_; }

    /// Verifies one trade without recording it.
    [[nodiscard]] IPVResult evaluate(const IPVInput& input) const;

    /// Verifies a portfolio, replacing any previously stored run, and returns
    /// the per-trade metrics in input order.
    std::vector<IPVResult> process_portfolio(std::span<const IPVInput> portfolio);

    /// Results of the last process_portfolio call, in input order.
    [[nodiscard]] const std::vector<IPVResult>& results() const noexcept { return results_; }

    /// Breaching trades from the last run, worst absolute variance first.
    [[nodiscard]] std::vector<IPVResult> get_breaches() const;

    /// Roll-up of the last run.
    [[nodiscard]] IPVSummary summary() const;

    /// Per-book roll-up of the last run.
    [[nodiscard]] std::unordered_map<std::string, IPVSummary> summary_by_book() const;

    /// Metrics for one trade of the last run, empty if the id is unknown.
    [[nodiscard]] std::optional<IPVResult> find(std::string_view trade_id) const;

    /// Discards the stored run.
    void clear() noexcept;

private:
    double default_tolerance_;
    std::vector<IPVResult> results_;
};

}  // namespace valuation

#endif  // VALUATION_ENGINE_IPV_ENGINE_HPP
