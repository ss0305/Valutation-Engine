// =============================================================================
//  ReserveCalculator.cpp — Module 3 implementation.
// =============================================================================
#include "ReserveCalculator.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace valuation {
namespace {

void require_finite(double value, std::string_view field, const std::string& trade_id) {
    if (!std::isfinite(value)) {
        throw ReserveInputError("Position '" + trade_id + "': " + std::string(field) +
                                " must be finite");
    }
}

}  // namespace

void Position::validate() const {
    require_finite(position_size, "position_size", trade_id);
    require_finite(bid, "bid", trade_id);
    require_finite(ask, "ask", trade_id);
    require_finite(unearned_spread_fraction, "unearned_spread_fraction", trade_id);
    require_finite(close_out_cost_rate, "close_out_cost_rate", trade_id);
    require_finite(liquidity_horizon_factor, "liquidity_horizon_factor", trade_id);
    require_finite(reference_price, "reference_price", trade_id);

    if (ask < bid) {
        std::ostringstream out;
        out << "Position '" << trade_id << "': crossed market, ask (" << ask << ") is below bid ("
            << bid << ")";
        throw ReserveInputError(out.str());
    }
    if (unearned_spread_fraction < 0.0 || unearned_spread_fraction > 1.0) {
        throw ReserveInputError("Position '" + trade_id +
                                "': unearned_spread_fraction must lie in [0, 1]");
    }
    if (close_out_cost_rate < 0.0) {
        throw ReserveInputError("Position '" + trade_id +
                                "': close_out_cost_rate must be non-negative");
    }
    if (liquidity_horizon_factor < 0.0) {
        throw ReserveInputError("Position '" + trade_id +
                                "': liquidity_horizon_factor must be non-negative");
    }
    if (reference_price < 0.0) {
        throw ReserveInputError("Position '" + trade_id +
                                "': reference_price must be non-negative");
    }
}

std::string ReserveBreakdown::to_string() const {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "ReserveBreakdown(bid_offer=" << bid_offer
        << ", market_unearned_spread_pva=" << market_unearned_spread_pva
        << ", close_out_cost_pva=" << close_out_cost_pva << ", total=" << total()
        << ", positions=" << position_count << ')';
    return out.str();
}

double ReserveCalculator::compute_bid_offer_reserve(double position_size, double bid, double ask) {
    if (!std::isfinite(position_size) || !std::isfinite(bid) || !std::isfinite(ask)) {
        throw ReserveInputError("compute_bid_offer_reserve: inputs must be finite");
    }
    if (ask < bid) {
        std::ostringstream out;
        out << "compute_bid_offer_reserve: crossed market, ask (" << ask << ") is below bid ("
            << bid << ')';
        throw ReserveInputError(out.str());
    }
    return 0.5 * std::abs(position_size) * (ask - bid);
}

double ReserveCalculator::compute_bid_offer_reserve(const Position& position) {
    position.validate();
    return 0.5 * std::abs(position.position_size) * position.spread();
}

double ReserveCalculator::compute_market_unearned_spread_pva(const Position& position) {
    position.validate();
    return std::abs(position.position_size) * (0.5 * position.spread()) *
           position.unearned_spread_fraction;
}

double ReserveCalculator::compute_close_out_cost_pva(const Position& position) {
    position.validate();
    const double reference =
        position.reference_price > 0.0 ? position.reference_price : position.mid();
    return std::abs(position.position_size) * reference * position.close_out_cost_rate *
           position.liquidity_horizon_factor;
}

ReserveBreakdown ReserveCalculator::compute_position_reserves(const Position& position) {
    position.validate();  // Validate once; the helpers below revalidate cheaply.
    ReserveBreakdown breakdown{};
    breakdown.bid_offer = compute_bid_offer_reserve(position);
    breakdown.market_unearned_spread_pva = compute_market_unearned_spread_pva(position);
    breakdown.close_out_cost_pva = compute_close_out_cost_pva(position);
    breakdown.position_count = 1;
    return breakdown;
}

ReserveBreakdown ReserveCalculator::compute_portfolio_reserves(
    std::span<const Position> positions) {
    ReserveBreakdown total{};
    for (const Position& position : positions) {
        total += compute_position_reserves(position);
    }
    return total;
}

std::unordered_map<std::string, ReserveBreakdown> ReserveCalculator::aggregate_by_book(
    std::span<const Position> positions) {
    std::unordered_map<std::string, ReserveBreakdown> by_book;
    by_book.reserve(positions.size());
    for (const Position& position : positions) {
        by_book[position.book] += compute_position_reserves(position);
    }
    return by_book;
}

std::unordered_map<std::string, ReserveBreakdown> ReserveCalculator::aggregate_by_category(
    std::span<const Position> positions) {
    std::unordered_map<std::string, ReserveBreakdown> by_category;
    by_category.reserve(positions.size());
    for (const Position& position : positions) {
        by_category[position.category] += compute_position_reserves(position);
    }
    return by_category;
}

std::vector<KeyedReserve> ReserveCalculator::ranked_by_book(std::span<const Position> positions) {
    const auto by_book = aggregate_by_book(positions);
    std::vector<KeyedReserve> ranked;
    ranked.reserve(by_book.size());
    for (const auto& [book, breakdown] : by_book) {
        ranked.push_back(KeyedReserve{book, breakdown});
    }
    std::sort(ranked.begin(), ranked.end(), [](const KeyedReserve& lhs, const KeyedReserve& rhs) {
        const double lhs_total = lhs.breakdown.total();
        const double rhs_total = rhs.breakdown.total();
        if (lhs_total != rhs_total) {
            return lhs_total > rhs_total;
        }
        return lhs.key < rhs.key;
    });
    return ranked;
}

}  // namespace valuation
