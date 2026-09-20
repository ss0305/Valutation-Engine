import sqlite3
from datetime import datetime, timezone
import valuation_engine as ve


def create_schema(conn: sqlite3.Connection) -> None:
    """Initializes schema for Front Office marks, consensus quotes, and IPV/PVA audit logs."""
    with conn:
        # Table 1: Front Office Trades & Market Quotes
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                book TEXT NOT NULL,
                currency_pair TEXT NOT NULL,
                position_size REAL NOT NULL,
                fo_val REAL NOT NULL,
                spot_bid REAL NOT NULL,
                spot_ask REAL NOT NULL,
                unearned_fraction REAL DEFAULT 0.5
            )
        """)

        # Table 2: Independent Consensus Marks (Consensus/Broker Data)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consensus_marks (
                trade_id TEXT PRIMARY KEY,
                consensus_val REAL NOT NULL,
                as_of_date TEXT NOT NULL,
                FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
            )
        """)

        # Table 3: Product Control Audit Log (Calculated Reserves & IPV Variance)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS valuation_audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                fo_val REAL NOT NULL,
                consensus_val REAL NOT NULL,
                variance REAL NOT NULL,
                variance_pct REAL,
                breach_flag INTEGER NOT NULL,
                bid_offer_reserve REAL NOT NULL,
                calculated_at TEXT NOT NULL
            )
        """)


def seed_mock_data(conn: sqlite3.Connection) -> None:
    """Seeds sample FX portfolio data to test the workflow."""
    with conn:
        conn.executemany("""
            INSERT OR REPLACE INTO trades 
            (trade_id, book, currency_pair, position_size, fo_val, spot_bid, spot_ask, unearned_fraction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            ("FX-OPT-001", "FX_G10", "EURUSD", 1_000_000.0, 1050000.0, 1.0995, 1.1005, 0.5),
            ("FX-OPT-002", "FX_EM",  "USDINR", 5_000_000.0, 2100000.0, 83.1000, 83.2500, 0.4),
            ("FX-OPT-003", "FX_G10", "GBPUSD", -500_000.0, -12000.0,  1.2650, 1.2660, 0.5), # Short position
            ("FX-OPT-004", "FX_EM",  "USDZAR", 2_000_000.0,  45000.0, 18.5000, 18.6000, 0.3)  # Zero consensus test
        ])

        conn.executemany("""
            INSERT OR REPLACE INTO consensus_marks (trade_id, consensus_val, as_of_date)
            VALUES (?, ?, ?)
        """, [
            ("FX-OPT-001", 1000000.0, "2026-09-20"), # 5% variance -> Breach (>2%)
            ("FX-OPT-002", 2095000.0, "2026-09-20"), # Within tolerance
            ("FX-OPT-003", -12100.0,  "2026-09-20"), # Short position check
            ("FX-OPT-004", 0.0,       "2026-09-20")  # Zero-mark breach trigger
        ])


def run_pipeline(db_path: str = "valuation_data.db") -> None:
    """Executes SQL extraction, C++ engine calculations, and SQL persistence."""
    conn = sqlite3.connect(db_path)
    
    # 1. Setup DB Schema and Seed Mock Data
    create_schema(conn)
    seed_mock_data(conn)

    # 2. Extract Data from SQL JOIN Query
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            t.trade_id, t.book, t.position_size, t.fo_val, 
            c.consensus_val, t.spot_bid, t.spot_ask
        FROM trades t
        JOIN consensus_marks c ON t.trade_id = c.trade_id
    """)
    rows = cursor.fetchall()

    # 3. Instantiate C++ Engines via Python Bindings
    ipv_engine = ve.IPVEngine(default_tolerance=0.02) # 2% breach tolerance
    ipv_inputs = []
    audit_records = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for trade_id, book, pos_size, fo_val, consensus_val, spot_bid, spot_ask in rows:
        # Construct C++ IPVInput struct
        ipv_inputs.append(ve.IPVInput(trade_id, fo_val, consensus_val, book=book))

        # Compute Bid-Offer Reserve using C++ ReserveCalculator static method
        bid_offer_res = ve.ReserveCalculator.compute_bid_offer_reserve(
            abs(pos_size), spot_bid, spot_ask
        )

        # Store intermediate mapping for result persistence
        audit_records.append({
            "trade_id": trade_id,
            "fo_val": fo_val,
            "consensus_val": consensus_val,
            "bid_offer_reserve": bid_offer_res
        })

    # 4. Process Entire Portfolio in C++ Engine
    summary = ipv_engine.process_portfolio(ipv_inputs)
    results = ipv_engine.get_results()

    # 5. Persist Output Back to SQL Audit Table
    audit_db_rows = []
    for rec, res in zip(audit_records, results):
        audit_db_rows.append((
            rec["trade_id"],
            rec["fo_val"],
            rec["consensus_val"],
            res.variance,
            res.variance_pct,  # None/nullopt if consensus == 0
            1 if res.is_breach else 0,
            rec["bid_offer_reserve"],
            timestamp
        ))

    with conn:
        conn.executemany("""
            INSERT INTO valuation_audit_log 
            (trade_id, fo_val, consensus_val, variance, variance_pct, breach_flag, bid_offer_reserve, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, audit_db_rows)

    print(f"[SUCCESS] Pipeline executed. Portfolio Total Abs Variance: {summary.total_abs_variance:,.2f}")
    print(f"[SUMMARY] Total Breaches Detected: {summary.breach_count}")

    conn.close()


if __name__ == "__main__":
    run_pipeline()