// =============================================================================
//  ReserveCalculator.hpp
//  Module 3: Valuation Reserves & Prudent Valuation Adjustments
//
//  Bid-offer reserves, category-level PVA (market unearned spread and close-out
//  costs), and memory-efficient aggregation per book / category / portfolio.
//
//  Sign convention: every reserve is returned as a non-negative cost, i.e. an
//  amount to be deducted from fair value. Position direction (long or short)
//  does not change the cost of crossing the spread, so position size enters the
//  formulae through its magnitude.
// =============================================================================
#ifndef VALUATION_ENGINE_RESERVE_CALCULATOR_HPP
#define VALUATION_ENGINE_RESERVE_CALCULATOR_HPP

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace valuation {

/// Thrown when position data cannot support a reserve calculation
/// (crossed market, non-finite inputs, out-of-range fractions).
class ReserveInputError : public std::invalid_argument {
public:
    explicit ReserveInputError(const std::string& what_arg)
        : std::invalid_argument(what_arg) {}
};

/// A single holding presented for reserving.
struct Position {
    std::string trade_id;
    std::string book;      ///< Desk or portfolio the position rolls up to.
    std::string category;  ///< Risk category, e.g. "FX Options", "Rates".

    /// Signed position size in the units the bid/ask are quoted against
    /// (contracts, notional, or units of the foreign currency).
    double position_size{0.0};
    double bid{0.0};  ///< Independent bid price. Must not exceed `ask`.
    double ask{0.0};  ///< Independent ask price.

    /// Fraction of the bid-offer spread booked as day-one P&L but not yet
    /// earned through the passage of time or hedging, in [0, 1]. Drives the
    /// Market Unearned Spread PVA.
    double unearned_spread_fraction{0.0};

    /// Additional cost of exiting, as a fraction of notional, over and above
    /// half the quoted spread — the concession a large or illiquid position
    /// pays on close-out. Must be non-negative.
    double close_out_cost_rate{0.0};

    /// Scales the close-out cost for the exit horizon (e.g. sqrt(days to
    /// liquidate)). Must be non-negative; 1.0 means a one-period exit.
    double liquidity_horizon_factor{1.0};

    /// Reference price used to turn `close_out_cost_rate` into currency. When
    /// left at zero the mid of bid/ask is used.
    double reference_price{0.0};

    /// Throws ReserveInputError if the position cannot be reserved.
    void validate() const;

    [[nodiscard]] double mid() const noexcept { return 0.5 * (bid + ask); }
    [[nodiscard]] double spread() const noexcept { return ask - bid; }
};

/// Reserve components for a position, book, category, or whole portfolio.
struct ReserveBreakdown {
    double bid_offer{0.0};                   ///< Bid-offer valuation reserve.
    double market_unearned_spread_pva{0.0};  ///< Category-level MUS PVA.
    double close_out_cost_pva{0.0};          ///< Category-level close-out PVA.
    std::size_t position_count{0};

    [[nodiscard]] double total_pva() const noexcept {
        return market_unearned_spread_pva + close_out_cost_pva;
    }
    [[nodiscard]] double total() const noexcept { return bid_offer + total_pva(); }

    ReserveBreakdown& operator+=(const ReserveBreakdown& other) noexcept {
        bid_offer += other.bid_offer;
        market_unearned_spread_pva += other.market_unearned_spread_pva;
        close_out_cost_pva += other.close_out_cost_pva;
        position_count += other.position_count;
        return *this;
    }

    [[nodiscard]] std::string to_string() const;
};

/// A breakdown tagged with the key it was aggregated under.
struct KeyedReserve {
    std::string key;
    ReserveBreakdown breakdown;
};

// -----------------------------------------------------------------------------
// ReserveCalculator
// -----------------------------------------------------------------------------

class ReserveCalculator {
public:
    ReserveCalculator() = default;

    /// Reserve_BidOffer = 1/2 * |position size| * (ask - bid).
    [[nodiscard]] static double compute_bid_offer_reserve(double position_size, double bid,
                                                          double ask);

    /// Convenience overload taking a validated position.
    [[nodiscard]] static double compute_bid_offer_reserve(const Position& position);

    /// Market Unearned Spread PVA: the portion of the spread captured at
    /// inception that is not yet earned.
    ///   MUS = |size| * (spread / 2) * unearned_spread_fraction
    [[nodiscard]] static double compute_market_unearned_spread_pva(const Position& position);

    /// Close-out Costs PVA: the exit concession beyond half the quoted spread.
    ///   CloseOut = |size| * reference_price * close_out_cost_rate * horizon_factor
    [[nodiscard]] static double compute_close_out_cost_pva(const Position& position);

    /// All three components for one position.
    [[nodiscard]] static ReserveBreakdown compute_position_reserves(const Position& position);

    /// Portfolio total across every position.
    [[nodiscard]] static ReserveBreakdown compute_portfolio_reserves(
        std::span<const Position> positions);

    /// Per-book aggregation.
    [[nodiscard]] static std::unordered_map<std::string, ReserveBreakdown> aggregate_by_book(
        std::span<const Position> positions);

    /// Per-category aggregation.
    [[nodiscard]] static std::unordered_map<std::string, ReserveBreakdown> aggregate_by_category(
        std::span<const Position> positions);

    /// Per-book aggregation flattened and sorted by descending total reserve —
    /// the shape a reserve report or a Python caller wants.
    [[nodiscard]] static std::vector<KeyedReserve> ranked_by_book(
        std::span<const Position> positions);
};

}  // namespace valuation

#endif  // VALUATION_ENGINE_RESERVE_CALCULATOR_HPP
