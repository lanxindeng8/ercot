"""Tests for wind forecast data fetcher."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from prediction.src.data.ercot.wind_forecast import (
    deduplicate_latest,
    fetch_wind_forecast,
    pivot_to_regions,
    save_to_sqlite,
)


def _make_raw_df(posted_times, delivery_date="2025-06-15", hour_ending="14"):
    """Create a synthetic raw DataFrame with multiple posted versions."""
    rows = []
    for i, pt in enumerate(posted_times):
        rows.append({
            "postedDatetime": pt,
            "deliveryDate": delivery_date,
            "hourEnding": hour_ending,
            "genSystemWide": 1000.0 + i,
            "COPHSLSystemWide": 2000.0,
            "STWPFSystemWide": 1500.0 + i,
            "WGRPPSystemWide": 1400.0,
            "genLoadZoneSouthHouston": 100.0,
            "COPHSLLoadZoneSouthHouston": 200.0,
            "STWPFLoadZoneSouthHouston": 150.0,
            "WGRPPLoadZoneSouthHouston": 140.0,
            "genLoadZoneWest": 500.0,
            "COPHSLLoadZoneWest": 600.0,
            "STWPFLoadZoneWest": 550.0,
            "WGRPPLoadZoneWest": 540.0,
            "genLoadZoneNorth": 300.0,
            "COPHSLLoadZoneNorth": 400.0,
            "STWPFLoadZoneNorth": 350.0,
            "WGRPPLoadZoneNorth": 340.0,
            "HSLSystemWide": 5000.0,
            "DSTFlag": "N",
        })
    return pd.DataFrame(rows)


class TestDeduplicateLatest:
    def test_keeps_latest_posted(self):
        df = _make_raw_df([
            "2025-06-15T08:00:00",
            "2025-06-15T10:00:00",
            "2025-06-15T06:00:00",
        ])
        result = deduplicate_latest(df)
        assert len(result) == 1
        # Latest posted (10:00) has genSystemWide = 1001.0 (index 1)
        assert result.iloc[0]["genSystemWide"] == 1001.0

    def test_preserves_different_hours(self):
        df1 = _make_raw_df(["2025-06-15T08:00:00"], hour_ending="14")
        df2 = _make_raw_df(["2025-06-15T09:00:00"], hour_ending="15")
        df = pd.concat([df1, df2], ignore_index=True)
        result = deduplicate_latest(df)
        assert len(result) == 2

    def test_empty_dataframe(self):
        result = deduplicate_latest(pd.DataFrame())
        assert result.empty


class TestPivotToRegions:
    def test_produces_four_regions(self):
        df = _make_raw_df(["2025-06-15T10:00:00"])
        df = deduplicate_latest(df)
        result = pivot_to_regions(df)
        assert len(result) == 4
        assert set(result["region"]) == {"system", "south_houston", "west", "north"}

    def test_column_names(self):
        df = _make_raw_df(["2025-06-15T10:00:00"])
        df = deduplicate_latest(df)
        result = pivot_to_regions(df)
        expected_cols = {"delivery_date", "hour_ending", "region",
                         "gen_mw", "stwpf_mw", "wgrpp_mw", "cop_hsl_mw"}
        assert set(result.columns) == expected_cols

    def test_system_values(self):
        df = _make_raw_df(["2025-06-15T10:00:00"])
        df = deduplicate_latest(df)
        result = pivot_to_regions(df)
        system = result[result["region"] == "system"].iloc[0]
        assert system["gen_mw"] == 1000.0
        assert system["stwpf_mw"] == 1500.0

    def test_hour_ending_is_int(self):
        df = _make_raw_df(["2025-06-15T10:00:00"])
        df = deduplicate_latest(df)
        result = pivot_to_regions(df)
        assert all(isinstance(h, int) for h in result["hour_ending"])

    def test_empty_dataframe(self):
        result = pivot_to_regions(pd.DataFrame())
        assert result.empty
        assert "region" in result.columns


class TestSaveToSqlite:
    def test_roundtrip(self):
        df = _make_raw_df(["2025-06-15T10:00:00"])
        df = deduplicate_latest(df)
        long_df = pivot_to_regions(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            save_to_sqlite(long_df, db_path)

            conn = sqlite3.connect(str(db_path))
            result = pd.read_sql("SELECT * FROM wind_forecast", conn)
            conn.close()

            assert len(result) == 4
            assert set(result["region"]) == {"system", "south_houston", "west", "north"}

    def test_upsert(self):
        df = _make_raw_df(["2025-06-15T10:00:00"])
        df = deduplicate_latest(df)
        long_df = pivot_to_regions(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            save_to_sqlite(long_df, db_path)
            # Save again — should upsert, not duplicate
            save_to_sqlite(long_df, db_path)

            conn = sqlite3.connect(str(db_path))
            result = pd.read_sql("SELECT * FROM wind_forecast", conn)
            conn.close()

            assert len(result) == 4

    def test_empty_df_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            save_to_sqlite(pd.DataFrame(), db_path)
            assert not db_path.exists()


class TestFetchWindForecast:
    def test_assembles_pages(self):
        mock_client = MagicMock()
        mock_client.fetch_paginated_data.return_value = iter([
            [{"deliveryDate": "2025-06-15", "hourEnding": "14", "genSystemWide": 1000}],
            [{"deliveryDate": "2025-06-15", "hourEnding": "15", "genSystemWide": 1100}],
        ])

        result = fetch_wind_forecast(mock_client, "2025-06-15", "2025-06-15")
        assert len(result) == 2

        mock_client.fetch_paginated_data.assert_called_once_with(
            "/np4-732-cd/wpp_hrly_avrg_actl_fcast",
            {"deliveryDateFrom": "2025-06-15", "deliveryDateTo": "2025-06-15"},
        )

    def test_empty_response(self):
        mock_client = MagicMock()
        mock_client.fetch_paginated_data.return_value = iter([])

        result = fetch_wind_forecast(mock_client, "2025-06-15", "2025-06-15")
        assert result.empty
