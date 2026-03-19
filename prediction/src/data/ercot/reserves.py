"""
RT Reserves / ORDC data downloader for ERCOT NP6-792-ER archive report.

Downloads yearly XLSX files from ERCOT's archive API, parses monthly sheets
with SCED-interval reserve and ORDC price adder data, and stores to SQLite.
"""

import io
import sqlite3
import time
import zipfile
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from loguru import logger

from scraper.src.ercot_client import ErcotClient

ARCHIVE_URL = "https://api.ercot.com/api/public-reports/archive/np6-792-er"

# Columns we keep (mapped from raw → clean name)
# Raw column names have varying whitespace/casing across years
KEEP_COLUMNS = {
    "batch_id": "batch_id",
    "sced_timestamp": "sced_timestamp",
    "repeated_hour_flag": "repeated_hour",
    "system_lamda": "system_lambda",
    "system_lambda": "system_lambda",
    "prc": "prc",
    "rtolcap": "rtolcap",
    "rtoffcap": "rtoffcap",
    "rtorpa": "rtorpa",
    "rtoffpa": "rtoffpa",
    "rtolhsl": "rtolhsl",
    "rtbp": "rtbp",
    "rtordpa": "rtordpa",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rt_reserves (
    sced_timestamp TEXT NOT NULL,
    repeated_hour TEXT,
    batch_id INTEGER,
    system_lambda REAL,
    prc REAL,
    rtolcap REAL,
    rtoffcap REAL,
    rtorpa REAL,
    rtoffpa REAL,
    rtolhsl REAL,
    rtbp REAL,
    rtordpa REAL,
    PRIMARY KEY (sced_timestamp, batch_id)
);
"""

UPSERT_SQL = """
INSERT OR REPLACE INTO rt_reserves
    (sced_timestamp, batch_id, repeated_hour, system_lambda, prc,
     rtolcap, rtoffcap, rtorpa, rtoffpa, rtolhsl, rtbp, rtordpa)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def list_archive_files(client: ErcotClient) -> List[Dict[str, Any]]:
    """List available archive files for NP6-792-ER.

    Returns:
        List of dicts with docId, friendlyName, postDatetime.
    """
    headers = client.get_headers("public")
    response = client.session.get(ARCHIVE_URL, headers=headers, timeout=60)
    response.raise_for_status()
    data = response.json()

    archives = data.get("archives", [])
    logger.info(f"Found {len(archives)} archive files for NP6-792-ER")
    return [
        {
            "docId": a["docId"],
            "friendlyName": a.get("friendlyName", ""),
            "postDatetime": a.get("postDatetime", ""),
        }
        for a in archives
    ]


def download_archive(
    client: ErcotClient, doc_id: int, output_dir: Path
) -> Path:
    """Download an archive ZIP and extract the XLSX inside.

    Args:
        client: Authenticated ErcotClient.
        doc_id: Document ID from list_archive_files.
        output_dir: Directory to save extracted XLSX.

    Returns:
        Path to extracted XLSX file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    headers = client.get_headers("public")
    url = f"{ARCHIVE_URL}?download={doc_id}"
    logger.info(f"Downloading archive doc_id={doc_id}")

    for attempt in range(4):
        try:
            response = client.session.get(url, headers=headers, timeout=300, stream=True)
            if response.status_code == 429:
                raise Exception("429")
            response.raise_for_status()
            break
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                wait = 60 * (attempt + 1)
                logger.warning(f"Rate limited (429), waiting {wait}s (attempt {attempt + 1}/4)")
                time.sleep(wait)
                # Refresh token in case it expired
                headers = client.get_headers("public")
            else:
                raise

    content = response.content
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xlsx_names = [n for n in zf.namelist() if n.endswith(".xlsx")]
        if not xlsx_names:
            raise ValueError(f"No XLSX file found in archive doc_id={doc_id}")

        xlsx_name = xlsx_names[0]
        xlsx_path = output_dir / xlsx_name
        zf.extract(xlsx_name, output_dir)
        logger.info(f"Extracted {xlsx_name} to {output_dir}")

    return xlsx_path


def parse_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """Parse a yearly NP6-792-ER XLSX file into a DataFrame.

    The XLSX has 12 monthly sheets (Jan-Dec) plus sometimes a "Report Info"
    sheet. Data header is at row 8 (0-indexed). SCED timestamps are in
    America/Chicago and get converted to UTC.

    Args:
        xlsx_path: Path to XLSX file.

    Returns:
        DataFrame with cleaned column names and UTC timestamps.
    """
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    frames = []

    for sheet_name in xls.sheet_names:
        if sheet_name.lower().strip() in ("report info", "reportinfo"):
            continue

        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=8)
        except Exception:
            logger.warning(f"Could not read sheet '{sheet_name}' in {xlsx_path.name}")
            continue

        if df.empty or len(df) == 0:
            continue

        # Drop rows that are all NaN
        df = df.dropna(how="all")
        if df.empty:
            continue

        frames.append(df)

    if not frames:
        logger.warning(f"No data found in {xlsx_path.name}")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Clean column names: lowercase, strip, replace spaces with underscores
    combined.columns = [
        str(c).strip().lower().replace(" ", "_") for c in combined.columns
    ]

    # Normalize the SCED timestamp column name (varies across years)
    ts_col = None
    for col in combined.columns:
        if "sced" in col and "timestamp" in col:
            ts_col = col
            break
        if "sced" in col and "time" in col:
            ts_col = col
            break

    if ts_col is None:
        logger.error(f"No SCED timestamp column found in {xlsx_path.name}. Columns: {list(combined.columns)}")
        return pd.DataFrame()

    combined = combined.rename(columns={ts_col: "sced_timestamp"})

    # Parse timestamps: localize to Chicago, convert to UTC
    combined["sced_timestamp"] = pd.to_datetime(
        combined["sced_timestamp"], errors="coerce"
    )
    combined = combined.dropna(subset=["sced_timestamp"])

    combined["sced_timestamp"] = (
        combined["sced_timestamp"]
        .dt.tz_localize("America/Chicago", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    # Map to our canonical columns
    result_cols = {}
    for raw_col in combined.columns:
        clean = raw_col.strip().lower().replace(" ", "_")
        if clean in KEEP_COLUMNS:
            result_cols[raw_col] = KEEP_COLUMNS[clean]

    combined = combined.rename(columns=result_cols)

    # Drop duplicate columns (e.g. system_lamda + system_lambda → both mapped to system_lambda)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]

    # Keep only canonical columns that exist
    canonical = list(dict.fromkeys(KEEP_COLUMNS.values()))  # dedupe preserving order
    existing = [c for c in canonical if c in combined.columns]
    combined = combined[existing].copy()

    # Deduplicate (keep last occurrence)
    if "batch_id" in combined.columns:
        combined = combined.drop_duplicates(
            subset=["sced_timestamp", "batch_id"], keep="last"
        )

    logger.info(f"Parsed {len(combined)} rows from {xlsx_path.name}")
    return combined


def save_to_sqlite(df: pd.DataFrame, db_path: Path) -> None:
    """Write reserves data to SQLite, upserting on primary key.

    Args:
        df: DataFrame from parse_xlsx.
        db_path: Path to SQLite database file.
    """
    if df.empty:
        logger.warning("No data to save")
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(CREATE_TABLE_SQL)

        ordered_cols = [
            "sced_timestamp", "batch_id", "repeated_hour", "system_lambda",
            "prc", "rtolcap", "rtoffcap", "rtorpa", "rtoffpa", "rtolhsl",
            "rtbp", "rtordpa",
        ]

        # Fill missing columns with None
        for col in ordered_cols:
            if col not in df.columns:
                df[col] = None

        rows = df[ordered_cols].values.tolist()
        conn.executemany(UPSERT_SQL, rows)
        conn.commit()
        logger.info(f"Saved {len(rows)} rows to {db_path}")
    finally:
        conn.close()
