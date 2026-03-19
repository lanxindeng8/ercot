#!/usr/bin/env python
"""CLI to build spike prediction features for all settlement points.

Usage:
    python -m prediction.scripts.build_features [--sp HB_WEST] [--start 2015-01-01] [--end 2026-03-19]

Outputs parquet files to prediction/data/spike_features/<SP>.parquet
"""

import argparse
import time
from pathlib import Path

from loguru import logger

from prediction.src.features.spike_features import (
    SETTLEMENT_POINTS,
    build_spike_features,
)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "scraper" / "data" / "ercot_archive.db"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "spike_features"


def main():
    parser = argparse.ArgumentParser(description="Build spike prediction features")
    parser.add_argument("--sp", type=str, nargs="*", default=None,
                        help="Settlement points (default: all 14)")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2026-03-19")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    points = [s.upper() for s in args.sp] if args.sp else SETTLEMENT_POINTS

    logger.info("Building spike features for {} settlement points", len(points))
    logger.info("DB: {}", db_path)
    logger.info("Output: {}", OUT_DIR)

    t0 = time.time()
    for sp in points:
        t1 = time.time()
        df = build_spike_features(db_path, sp, args.start, args.end)
        if df.empty:
            logger.warning("  {} — empty, skipping", sp)
            continue

        out_path = OUT_DIR / f"{sp}.parquet"
        df.to_parquet(out_path, engine="pyarrow")
        logger.info("  {} — {:,} rows → {} ({:.1f}s)", sp, len(df), out_path, time.time() - t1)

    logger.info("Done in {:.1f}s", time.time() - t0)


if __name__ == "__main__":
    main()
