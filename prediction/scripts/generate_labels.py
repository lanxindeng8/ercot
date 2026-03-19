#!/usr/bin/env python
"""Generate spike labels for all settlement points and store in SQLite."""

import argparse
import sqlite3
import sys
from pathlib import Path

from loguru import logger

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from prediction.src.config import SETTLEMENT_POINTS
from prediction.src.labels.spike_labels import label_all_zones

DEFAULT_DB = Path(__file__).resolve().parent.parent.parent / "scraper" / "data" / "ercot_archive.db"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS spike_labels (
    time TEXT NOT NULL,
    settlement_point TEXT NOT NULL,
    lmp REAL,
    hub_lmp REAL,
    spread REAL,
    spike_event INTEGER,
    lead_spike_60 INTEGER,
    regime TEXT,
    PRIMARY KEY (time, settlement_point)
);
"""


def main():
    parser = argparse.ArgumentParser(description="Generate spike labels for ERCOT RTM")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to ercot_archive.db")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-03-19", help="End date (YYYY-MM-DD)")
    parser.add_argument("--horizon", type=int, default=60, help="Lead-spike horizon in minutes")
    args = parser.parse_args()

    if not args.db.exists():
        logger.error("Database not found: {}", args.db)
        sys.exit(1)

    # Exclude HB_HUBAVG from labeling (it's the hub reference)
    sps = [sp for sp in SETTLEMENT_POINTS if sp != "HB_HUBAVG"]

    logger.info("Generating labels for {} settlement points", len(sps))
    labels = label_all_zones(
        db_path=args.db,
        settlement_points=sps,
        start_date=args.start,
        end_date=args.end,
        horizon_minutes=args.horizon,
    )

    # Write to DB
    logger.info("Writing {:,} rows to spike_labels table", len(labels))
    with sqlite3.connect(str(args.db)) as conn:
        conn.execute("DROP TABLE IF EXISTS spike_labels")
        conn.execute(CREATE_TABLE)

        # Convert for storage
        out = labels.copy()
        out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out["spike_event"] = out["spike_event"].astype(int)
        out["lead_spike_60"] = out["lead_spike_60"].astype(int)
        out["regime"] = out["regime"].astype(str)

        out.to_sql("spike_labels", conn, if_exists="append", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spike_labels_sp ON spike_labels(settlement_point)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_spike_labels_time ON spike_labels(time)")

    logger.info("Done — spike_labels table written to {}", args.db)


if __name__ == "__main__":
    main()
