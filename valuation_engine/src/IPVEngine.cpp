// =============================================================================
//  IPVEngine.cpp — Module 2 implementation.
// =============================================================================
#include "IPVEngine.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iterator>
#include <sstream>

namespace valuation {
namespace {

/// Consensus marks below this magnitude are treated as zero for the ratio.
constexpr double kZeroMarkEpsilon = 1.0e-12;

void accumulate(IPVSummary& summary, const IPVResult& result) {
    ++summary.trade_count;
    if (result.breached) {
        ++summary.breach_count;
    }
    summary.total_fo_value += result.fo_value;
    summary.total_independent_value += result.independent_value;
    summary.net_variance += result.variance;
    summary.gross_absolute_variance += result.absolute_variance;
    if (result.absolute_variance > summary.largest_absolute_variance) {
        summary.largest_absolute_variance = result.absolute_variance;
        summary.largest_variance_trade_id = result.trade_id;
    }
}

}  // namespace

std::string IPVResult::to_string() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "IPVResult(trade_id='" << trade_id << "', fo=" << fo_value
        << ", independent=" << independent_value << ", variance=" << variance << ", variance_pct=";
    if (variance_pct.has_value()) {
        out << (*variance_pct * 100.0) << '%';
    } else {
        out << "n/a";
    }
    out << ", tolerance=" << (tolerance_applied * 100.0) << '%'
        << ", breached=" << (breached ? "true" : "false") << ')';
    return out.str();
}

IPVEngine::IPVEngine(double default_tolerance) : default_tolerance_(default_tolerance) {
    if (!std::isfinite(default_tolerance) || default_tolerance < 0.0) {
        throw IPVConfigurationError(
            "IPVEngine: default tolerance must be finite and non-negative");
    }
}

IPVResult IPVEngine::evaluate(const IPVInput& input) const {
    const double tolerance = input.tolerance.value_or(default_tolerance_);
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw IPVConfigurationError("IPVEngine: tolerance for trade '" + input.trade_id +
                                    "' must be finite and non-negative");
    }
    if (!std::isfinite(input.fo_value) || !std::isfinite(input.independent_value)) {
        throw IPVConfigurationError("IPVEngine: valuations for trade '" + input.trade_id +
                                    "' must be finite");
    }

    IPVResult result{};
    result.trade_id = input.trade_id;
    result.book = input.book;
    result.fo_value = input.fo_value;
    result.independent_value = input.independent_value;
    result.variance = input.fo_value - input.independent_value;
    result.absolute_variance = std::abs(result.variance);
    result.tolerance_applied = tolerance;

    if (std::abs(input.independent_value) > kZeroMarkEpsilon) {
        const double pct = result.absolute_variance / std::abs(input.independent_value);
        result.variance_pct = pct;
        result.breached = pct > tolerance;
    } else {
        // No usable consensus denominator: the percentage is undefined, so any
        // non-zero difference against a zero mark is escalated for review.
        result.variance_pct.reset();
        result.breached = result.absolute_variance > kZeroMarkEpsilon;
    }
    return result;
}

std::vector<IPVResult> IPVEngine::process_portfolio(std::span<const IPVInput> portfolio) {
    results_.clear();
    results_.reserve(portfolio.size());
    for (const IPVInput& input : portfolio) {
        results_.push_back(evaluate(input));
    }
    return results_;
}

std::vector<IPVResult> IPVEngine::get_breaches() const {
    std::vector<IPVResult> breaches;
    breaches.reserve(results_.size());
    std::copy_if(results_.begin(), results_.end(), std::back_inserter(breaches),
                 [](const IPVResult& result) { return result.breached; });
    std::sort(breaches.begin(), breaches.end(),
              [](const IPVResult& lhs, const IPVResult& rhs) {
                  if (lhs.absolute_variance != rhs.absolute_variance) {
                      return lhs.absolute_variance > rhs.absolute_variance;
                  }
                  return lhs.trade_id < rhs.trade_id;
              });
    return breaches;
}

IPVSummary IPVEngine::summary() const {
    IPVSummary summary{};
    for (const IPVResult& result : results_) {
        accumulate(summary, result);
    }
    return summary;
}

std::unordered_map<std::string, IPVSummary> IPVEngine::summary_by_book() const {
    std::unordered_map<std::string, IPVSummary> by_book;
    by_book.reserve(results_.size());
    for (const IPVResult& result : results_) {
        accumulate(by_book[result.book], result);
    }
    return by_book;
}

std::optional<IPVResult> IPVEngine::find(std::string_view trade_id) const {
    const auto it = std::find_if(results_.begin(), results_.end(),
                                 [trade_id](const IPVResult& result) {
                                     return result.trade_id == trade_id;
                                 });
    if (it == results_.end()) {
        return std::nullopt;
    }
    return *it;
}

void IPVEngine::clear() noexcept {
    results_.clear();
}

}  // namespace valuation
