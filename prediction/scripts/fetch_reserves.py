#!/usr/bin/env python3
"""
CLI script to fetch ERCOT RT reserves / ORDC data (NP6-792-ER archive).

Usage:
    python prediction/scripts/fetch_reserves.py \
        --db-path scraper/data/ercot_archive.db \
        --output-dir /tmp/ercot_reserves
"""

import argparse
from pathlib import Path

from loguru import logger

from scraper.src.ercot_client import create_client_from_env
from prediction.src.data.ercot.reserves import (
    list_archive_files,
    download_archive,
    parse_xlsx,
    save_to_sqlite,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch ERCOT RT reserves / ORDC archive data"
    )
    parser.add_argument(
        "--db-path",
        default="scraper/data/ercot_archive.db",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/ercot_reserves",
        help="Directory to save downloaded XLSX files",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path)
    output_dir = Path(args.output_dir)

    client = create_client_from_env()

    archives = list_archive_files(client)
    logger.info(f"Found {len(archives)} archive files")

    # Filter: only RTM_ORDC files (skip HIST_RT_SCED which has different format)
    archives = [a for a in archives if a["friendlyName"].startswith("RTM_ORDC")]
    logger.info(f"Filtered to {len(archives)} RTM_ORDC files")

    # Keep only the latest archive per year (most recent postDatetime)
    # Each weekly update overwrites the yearly file, so latest = most complete
    from collections import defaultdict
    by_year = defaultdict(list)
    for a in archives:
        year = a["friendlyName"].rsplit("_", 1)[-1]  # e.g. "2025"
        by_year[year].append(a)
    archives = []
    for year in sorted(by_year.keys()):
        # Sort by postDatetime descending, take first (latest)
        candidates = sorted(by_year[year], key=lambda x: x.get("postDatetime", ""), reverse=True)
        archives.append(candidates[0])
        logger.info(f"  Year {year}: using latest archive (docId={candidates[0]['docId']}, posted={candidates[0].get('postDatetime', '?')})")
    logger.info(f"Deduplicated to {len(archives)} archives (one per year)")

    total_rows = 0
    for i, archive in enumerate(archives, 1):
        doc_id = archive["docId"]
        name = archive["friendlyName"]
        logger.info(f"[{i}/{len(archives)}] {name} (docId={doc_id})")

        # Check if XLSX already exists in output dir
        existing = list(output_dir.glob(f"*{doc_id}*")) + list(
            output_dir.glob(f"*{name}*")
        )
        xlsx_path = None

        # Check for any xlsx already extracted
        if output_dir.exists():
            for f in output_dir.iterdir():
                if f.suffix == ".xlsx" and name and name.split(".")[0] in f.name:
                    xlsx_path = f
                    logger.info(f"  Already downloaded: {f.name}")
                    break

        if xlsx_path is None:
            try:
                xlsx_path = download_archive(client, doc_id, output_dir)
                import time; time.sleep(5)  # rate limit courtesy
            except Exception:
                logger.exception(f"  Failed to download {name}")
                continue

        try:
            df = parse_xlsx(xlsx_path)
            if df.empty:
                logger.warning(f"  No data parsed from {xlsx_path.name}")
                continue
            save_to_sqlite(df, db_path)
            total_rows += len(df)
        except Exception:
            logger.exception(f"  Failed to parse/save {xlsx_path.name}")

    logger.info(f"Done. Total rows saved: {total_rows}")


if __name__ == "__main__":
    main()
