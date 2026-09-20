// =============================================================================
//  bindings.cpp
//  Module 4: pybind11 interface exposing the valuation core to Python.
//
//  Import as:  import valuation_engine as ve
// =============================================================================
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "IPVEngine.hpp"
#include "OptionPricer.hpp"
#include "ReserveCalculator.hpp"

namespace py = pybind11;
using namespace valuation;

namespace {

/// Vectorised analytical pricing: one NumPy array in, one out.
py::array_t<double> price_curve(const OptionPricer& pricer, const OptionSpec& base,
                                const py::array_t<double, py::array::c_style | py::array::forcecast>&
                                    spots) {
    const auto view = spots.unchecked<1>();
    py::array_t<double> out(view.shape(0));
    auto out_view = out.mutable_unchecked<1>();
    OptionSpec working = base;
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
        working.spot = view(i);
        out_view(i) = pricer.price(working);
    }
    return out;
}

/// Implied-vol smile calibration across strikes and market prices.
py::list calibrate_curve(const OptionPricer& pricer, const OptionSpec& base,
                         const std::vector<double>& strikes,
                         const std::vector<double>& market_prices) {
    if (strikes.size() != market_prices.size()) {
        throw std::invalid_argument("strikes and market_prices must be the same length");
    }
    py::list out;
    OptionSpec working = base;
    for (std::size_t i = 0; i < strikes.size(); ++i) {
        working.strike = strikes[i];
        const std::optional<double> implied = pricer.calibrate_volatility(working, market_prices[i]);
        if (implied.has_value()) {
            out.append(py::cast(*implied));
        } else {
            out.append(py::none());
        }
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(valuation_engine, m) {
    m.doc() =
        "Real-time fair value, IPV and valuation reserves engine.\n"
        "C++20 core exposed through pybind11: Black-Scholes/Garman-Kohlhagen pricing,\n"
        "Newton-Raphson implied volatility, multi-threaded Monte Carlo, Independent\n"
        "Price Verification and prudent valuation reserves.";

    // -- Exceptions ----------------------------------------------------------
    py::register_exception<InvalidOptionSpec>(m, "InvalidOptionSpec", PyExc_ValueError);
    py::register_exception<IPVConfigurationError>(m, "IPVConfigurationError", PyExc_ValueError);
    py::register_exception<ReserveInputError>(m, "ReserveInputError", PyExc_ValueError);

    // =========================== Module 1 ===================================
    py::enum_<OptionType>(m, "OptionType", "Payoff direction of a European option.")
        .value("Call", OptionType::Call)
        .value("Put", OptionType::Put);

    py::class_<OptionSpec>(m, "OptionSpec", "European FX option under Garman-Kohlhagen.")
        .def(py::init<>())
        .def(py::init([](double spot, double strike, double time_to_expiry, double rate_domestic,
                         double rate_foreign, double volatility, OptionType type) {
                 return OptionSpec{spot,      strike,     time_to_expiry, rate_domestic,
                                   rate_foreign, volatility, type};
             }),
             py::arg("spot"), py::arg("strike"), py::arg("time_to_expiry"),
             py::arg("rate_domestic") = 0.0, py::arg("rate_foreign") = 0.0,
             py::arg("volatility") = 0.0, py::arg("type") = OptionType::Call)
        .def_readwrite("spot", &OptionSpec::spot)
        .def_readwrite("strike", &OptionSpec::strike)
        .def_readwrite("time_to_expiry", &OptionSpec::time_to_expiry)
        .def_readwrite("rate_domestic", &OptionSpec::rate_domestic)
        .def_readwrite("rate_foreign", &OptionSpec::rate_foreign)
        .def_readwrite("volatility", &OptionSpec::volatility)
        .def_readwrite("type", &OptionSpec::type)
        .def("validate", &OptionSpec::validate,
             "Raise ValueError if the specification cannot be priced.")
        .def("__repr__", [](const OptionSpec& spec) {
            std::ostringstream out;
            out << "OptionSpec(spot=" << spec.spot << ", strike=" << spec.strike
                << ", time_to_expiry=" << spec.time_to_expiry
                << ", rate_domestic=" << spec.rate_domestic
                << ", rate_foreign=" << spec.rate_foreign << ", volatility=" << spec.volatility
                << ", type=" << to_string(spec.type) << ')';
            return out.str();
        });

    py::class_<PricingResult>(m, "PricingResult", "Price and Greeks from the closed form.")
        .def_readonly("price", &PricingResult::price)
        .def_readonly("delta", &PricingResult::delta)
        .def_readonly("gamma", &PricingResult::gamma)
        .def_readonly("vega", &PricingResult::vega)
        .def_readonly("theta", &PricingResult::theta)
        .def_readonly("d1", &PricingResult::d1)
        .def_readonly("d2", &PricingResult::d2)
        .def("as_dict",
             [](const PricingResult& r) {
                 py::dict d;
                 d["price"] = r.price;
                 d["delta"] = r.delta;
                 d["gamma"] = r.gamma;
                 d["vega"] = r.vega;
                 d["theta"] = r.theta;
                 return d;
             })
        .def("__repr__", [](const PricingResult& r) {
            std::ostringstream out;
            out << "PricingResult(price=" << r.price << ", delta=" << r.delta
                << ", gamma=" << r.gamma << ", vega=" << r.vega << ", theta=" << r.theta << ')';
            return out.str();
        });

    py::class_<CalibrationSettings>(m, "CalibrationSettings",
                                    "Newton-Raphson solver tolerances and clamps.")
        .def(py::init<>())
        .def(py::init([](int max_iterations, double price_tolerance, double vol_tolerance,
                         double min_vega, double min_volatility, double max_volatility) {
                 return CalibrationSettings{max_iterations, price_tolerance, vol_tolerance,
                                            min_vega,       min_volatility,  max_volatility};
             }),
             py::arg("max_iterations") = 100, py::arg("price_tolerance") = 1.0e-10,
             py::arg("vol_tolerance") = 1.0e-12, py::arg("min_vega") = 1.0e-12,
             py::arg("min_volatility") = 1.0e-8, py::arg("max_volatility") = 5.0)
        .def_readwrite("max_iterations", &CalibrationSettings::max_iterations)
        .def_readwrite("price_tolerance", &CalibrationSettings::price_tolerance)
        .def_readwrite("vol_tolerance", &CalibrationSettings::vol_tolerance)
        .def_readwrite("min_vega", &CalibrationSettings::min_vega)
        .def_readwrite("min_volatility", &CalibrationSettings::min_volatility)
        .def_readwrite("max_volatility", &CalibrationSettings::max_volatility);

    py::class_<MonteCarloConfig>(m, "MonteCarloConfig", "Monte Carlo run configuration.")
        .def(py::init<>())
        .def(py::init([](std::size_t num_paths, unsigned num_threads, std::uint64_t seed,
                         bool antithetic) {
                 return MonteCarloConfig{num_paths, num_threads, seed, antithetic};
             }),
             py::arg("num_paths") = 100000, py::arg("num_threads") = 0, py::arg("seed") = 42,
             py::arg("antithetic") = true)
        .def_readwrite("num_paths", &MonteCarloConfig::num_paths)
        .def_readwrite("num_threads", &MonteCarloConfig::num_threads)
        .def_readwrite("seed", &MonteCarloConfig::seed)
        .def_readwrite("antithetic", &MonteCarloConfig::antithetic);

    py::class_<MonteCarloResult>(m, "MonteCarloResult", "Monte Carlo estimate and sampling error.")
        .def_readonly("price", &MonteCarloResult::price)
        .def_readonly("standard_error", &MonteCarloResult::standard_error)
        .def_readonly("paths", &MonteCarloResult::paths)
        .def_readonly("threads_used", &MonteCarloResult::threads_used)
        .def_property_readonly("confidence_interval_95",
                               &MonteCarloResult::confidence_interval_95)
        .def("__repr__", [](const MonteCarloResult& r) {
            std::ostringstream out;
            out << "MonteCarloResult(price=" << r.price << ", standard_error=" << r.standard_error
                << ", paths=" << r.paths << ", threads_used=" << r.threads_used << ')';
            return out.str();
        });

    py::class_<OptionPricer>(m, "OptionPricer",
                             "Black-Scholes/Garman-Kohlhagen pricing, calibration and Monte Carlo.")
        .def(py::init<>())
        .def(py::init<CalibrationSettings>(), py::arg("settings"))
        .def_property_readonly("settings", &OptionPricer::settings)
        .def("price", &OptionPricer::price, py::arg("spec"),
             "Closed-form price of a European FX option.")
        .def("evaluate", &OptionPricer::evaluate, py::arg("spec"),
             "Closed-form price together with delta, gamma, vega and theta.")
        .def("calibrate_volatility", &OptionPricer::calibrate_volatility, py::arg("spec"),
             py::arg("target_price"),
             "Newton-Raphson implied volatility; returns None if it does not converge.")
        .def("monte_carlo_price", &OptionPricer::monte_carlo_price, py::arg("spec"),
             py::arg("config") = MonteCarloConfig{},
             py::call_guard<py::gil_scoped_release>(),
             "Multi-threaded Monte Carlo price (releases the GIL while simulating).")
        .def_static("discounted_intrinsic", &OptionPricer::discounted_intrinsic, py::arg("spec"))
        .def("price_curve", &price_curve, py::arg("spec"), py::arg("spots"),
             "Price across a NumPy array of spot levels, returning a NumPy array.")
        .def("calibrate_curve", &calibrate_curve, py::arg("spec"), py::arg("strikes"),
             py::arg("market_prices"),
             "Calibrate a volatility smile; non-convergent strikes come back as None.");

    // =========================== Module 2 ===================================
    py::class_<IPVInput>(m, "IPVInput", "One trade presented for price verification.")
        .def(py::init<>())
        .def(py::init([](std::string trade_id, double fo_value, double independent_value,
                         std::optional<double> tolerance, std::string book) {
                 return IPVInput{std::move(trade_id), fo_value, independent_value, tolerance,
                                 std::move(book)};
             }),
             py::arg("trade_id"), py::arg("fo_value"), py::arg("independent_value"),
             py::arg("tolerance") = py::none(), py::arg("book") = std::string{})
        .def_readwrite("trade_id", &IPVInput::trade_id)
        .def_readwrite("fo_value", &IPVInput::fo_value)
        .def_readwrite("independent_value", &IPVInput::independent_value)
        .def_readwrite("tolerance", &IPVInput::tolerance)
        .def_readwrite("book", &IPVInput::book)
        .def("__repr__", [](const IPVInput& input) {
            std::ostringstream out;
            out << "IPVInput(trade_id='" << input.trade_id << "', fo_value=" << input.fo_value
                << ", independent_value=" << input.independent_value << ')';
            return out.str();
        });

    py::class_<IPVResult>(m, "IPVResult", "Structured variance metrics for one trade.")
        .def_readonly("trade_id", &IPVResult::trade_id)
        .def_readonly("book", &IPVResult::book)
        .def_readonly("fo_value", &IPVResult::fo_value)
        .def_readonly("independent_value", &IPVResult::independent_value)
        .def_readonly("variance", &IPVResult::variance)
        .def_readonly("absolute_variance", &IPVResult::absolute_variance)
        .def_readonly("variance_pct", &IPVResult::variance_pct)
        .def_readonly("tolerance_applied", &IPVResult::tolerance_applied)
        .def_readonly("breached", &IPVResult::breached)
        .def("as_dict",
             [](const IPVResult& r) {
                 py::dict d;
                 d["trade_id"] = r.trade_id;
                 d["book"] = r.book;
                 d["fo_value"] = r.fo_value;
                 d["independent_value"] = r.independent_value;
                 d["variance"] = r.variance;
                 d["absolute_variance"] = r.absolute_variance;
                 d["variance_pct"] = r.variance_pct ? py::cast(*r.variance_pct) : py::none();
                 d["tolerance_applied"] = r.tolerance_applied;
                 d["breached"] = r.breached;
                 return d;
             })
        .def("__repr__", &IPVResult::to_string);

    py::class_<IPVSummary>(m, "IPVSummary", "Portfolio roll-up of a verification run.")
        .def_readonly("trade_count", &IPVSummary::trade_count)
        .def_readonly("breach_count", &IPVSummary::breach_count)
        .def_readonly("total_fo_value", &IPVSummary::total_fo_value)
        .def_readonly("total_independent_value", &IPVSummary::total_independent_value)
        .def_readonly("net_variance", &IPVSummary::net_variance)
        .def_readonly("gross_absolute_variance", &IPVSummary::gross_absolute_variance)
        .def_readonly("largest_absolute_variance", &IPVSummary::largest_absolute_variance)
        .def_readonly("largest_variance_trade_id", &IPVSummary::largest_variance_trade_id)
        .def_property_readonly("breach_rate", &IPVSummary::breach_rate)
        .def("__repr__", [](const IPVSummary& s) {
            std::ostringstream out;
            out << "IPVSummary(trades=" << s.trade_count << ", breaches=" << s.breach_count
                << ", net_variance=" << s.net_variance
                << ", gross_absolute_variance=" << s.gross_absolute_variance << ')';
            return out.str();
        });

    py::class_<IPVEngine>(m, "IPVEngine", "Front Office versus independent consensus verification.")
        .def(py::init<double>(), py::arg("default_tolerance") = 0.01)
        .def_property_readonly("default_tolerance", &IPVEngine::default_tolerance)
        .def("evaluate", &IPVEngine::evaluate, py::arg("input"),
             "Verify a single trade without recording it.")
        .def(
            "process_portfolio",
            [](IPVEngine& engine, const std::vector<IPVInput>& portfolio) {
                return engine.process_portfolio(std::span<const IPVInput>(portfolio));
            },
            py::arg("portfolio"),
            "Verify a list of trades and return per-trade metrics in input order.")
        .def("get_breaches", &IPVEngine::get_breaches,
             "Breaching trades from the last run, worst absolute variance first.")
        .def("results", &IPVEngine::results, "All results from the last run, in input order.")
        .def("summary", &IPVEngine::summary, "Roll-up of the last run.")
        .def("summary_by_book", &IPVEngine::summary_by_book,
             "Roll-up of the last run keyed by book.")
        .def(
            "find", [](const IPVEngine& engine, std::string_view id) { return engine.find(id); },
            py::arg("trade_id"), "Metrics for one trade of the last run, or None.")
        .def("clear", &IPVEngine::clear, "Discard the stored run.");

    // =========================== Module 3 ===================================
    py::class_<Position>(m, "Position", "A holding presented for reserving.")
        .def(py::init<>())
        .def(py::init([](std::string trade_id, std::string book, std::string category,
                         double position_size, double bid, double ask,
                         double unearned_spread_fraction, double close_out_cost_rate,
                         double liquidity_horizon_factor, double reference_price) {
                 return Position{std::move(trade_id),
                                 std::move(book),
                                 std::move(category),
                                 position_size,
                                 bid,
                                 ask,
                                 unearned_spread_fraction,
                                 close_out_cost_rate,
                                 liquidity_horizon_factor,
                                 reference_price};
             }),
             py::arg("trade_id"), py::arg("book") = std::string{},
             py::arg("category") = std::string{}, py::arg("position_size") = 0.0,
             py::arg("bid") = 0.0, py::arg("ask") = 0.0,
             py::arg("unearned_spread_fraction") = 0.0, py::arg("close_out_cost_rate") = 0.0,
             py::arg("liquidity_horizon_factor") = 1.0, py::arg("reference_price") = 0.0)
        .def_readwrite("trade_id", &Position::trade_id)
        .def_readwrite("book", &Position::book)
        .def_readwrite("category", &Position::category)
        .def_readwrite("position_size", &Position::position_size)
        .def_readwrite("bid", &Position::bid)
        .def_readwrite("ask", &Position::ask)
        .def_readwrite("unearned_spread_fraction", &Position::unearned_spread_fraction)
        .def_readwrite("close_out_cost_rate", &Position::close_out_cost_rate)
        .def_readwrite("liquidity_horizon_factor", &Position::liquidity_horizon_factor)
        .def_readwrite("reference_price", &Position::reference_price)
        .def_property_readonly("mid", &Position::mid)
        .def_property_readonly("spread", &Position::spread)
        .def("validate", &Position::validate)
        .def("__repr__", [](const Position& p) {
            std::ostringstream out;
            out << "Position(trade_id='" << p.trade_id << "', book='" << p.book << "', size="
                << p.position_size << ", bid=" << p.bid << ", ask=" << p.ask << ')';
            return out.str();
        });

    py::class_<ReserveBreakdown>(m, "ReserveBreakdown", "Reserve components at any aggregation.")
        .def(py::init<>())
        .def_readonly("bid_offer", &ReserveBreakdown::bid_offer)
        .def_readonly("market_unearned_spread_pva", &ReserveBreakdown::market_unearned_spread_pva)
        .def_readonly("close_out_cost_pva", &ReserveBreakdown::close_out_cost_pva)
        .def_readonly("position_count", &ReserveBreakdown::position_count)
        .def_property_readonly("total_pva", &ReserveBreakdown::total_pva)
        .def_property_readonly("total", &ReserveBreakdown::total)
        .def("as_dict",
             [](const ReserveBreakdown& b) {
                 py::dict d;
                 d["bid_offer"] = b.bid_offer;
                 d["market_unearned_spread_pva"] = b.market_unearned_spread_pva;
                 d["close_out_cost_pva"] = b.close_out_cost_pva;
                 d["total_pva"] = b.total_pva();
                 d["total"] = b.total();
                 d["position_count"] = b.position_count;
                 return d;
             })
        .def("__repr__", &ReserveBreakdown::to_string);

    py::class_<KeyedReserve>(m, "KeyedReserve", "A reserve breakdown tagged with its key.")
        .def_readonly("key", &KeyedReserve::key)
        .def_readonly("breakdown", &KeyedReserve::breakdown)
        .def("__repr__", [](const KeyedReserve& k) {
            return "KeyedReserve(key='" + k.key + "', " + k.breakdown.to_string() + ')';
        });

    py::class_<ReserveCalculator>(m, "ReserveCalculator",
                                  "Bid-offer reserves, PVA and portfolio aggregation.")
        .def(py::init<>())
        .def_static("compute_bid_offer_reserve",
                    py::overload_cast<double, double, double>(
                        &ReserveCalculator::compute_bid_offer_reserve),
                    py::arg("position_size"), py::arg("bid"), py::arg("ask"),
                    "0.5 * |position_size| * (ask - bid).")
        .def_static(
            "compute_bid_offer_reserve_for",
            py::overload_cast<const Position&>(&ReserveCalculator::compute_bid_offer_reserve),
            py::arg("position"), "Bid-offer reserve for a Position.")
        .def_static("compute_market_unearned_spread_pva",
                    &ReserveCalculator::compute_market_unearned_spread_pva, py::arg("position"))
        .def_static("compute_close_out_cost_pva", &ReserveCalculator::compute_close_out_cost_pva,
                    py::arg("position"))
        .def_static("compute_position_reserves", &ReserveCalculator::compute_position_reserves,
                    py::arg("position"))
        .def_static(
            "compute_portfolio_reserves",
            [](const std::vector<Position>& positions) {
                return ReserveCalculator::compute_portfolio_reserves(
                    std::span<const Position>(positions));
            },
            py::arg("positions"))
        .def_static(
            "aggregate_by_book",
            [](const std::vector<Position>& positions) {
                return ReserveCalculator::aggregate_by_book(std::span<const Position>(positions));
            },
            py::arg("positions"), "Per-book reserves as a dict of book -> ReserveBreakdown.")
        .def_static(
            "aggregate_by_category",
            [](const std::vector<Position>& positions) {
                return ReserveCalculator::aggregate_by_category(
                    std::span<const Position>(positions));
            },
            py::arg("positions"), "Per-category reserves as a dict.")
        .def_static(
            "ranked_by_book",
            [](const std::vector<Position>& positions) {
                return ReserveCalculator::ranked_by_book(std::span<const Position>(positions));
            },
            py::arg("positions"), "Per-book reserves as a list ordered by descending total.");

    m.attr("__version__") = "1.0.0";
}
