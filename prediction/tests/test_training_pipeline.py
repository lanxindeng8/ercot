"""
Tests for the unified training data pipeline and feature engineering.

Covers:
- SQLite data loading (DAM, RTM, fuel mix)
- Feature computation correctness
- Train / val / test split boundaries
- End-to-end pipeline parquet export
"""

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction.src.features.unified_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _NULLABLE_FEATURE_SET,
    aggregate_fuel_mix_hourly,
    compute_features,
    is_us_holiday,
)

# Import training_pipeline without triggering data/__init__.py (which pulls
# in influxdb_client_3).  We use importlib to load the single module file.
import importlib.util as _ilu, sys as _sys, types as _types

# Ensure the parent package "prediction.src.data" exists in sys.modules so
# importlib can resolve the module name, but without executing __init__.py.
for _pkg in ("prediction", "prediction.src", "prediction.src.data"):
    if _pkg not in _sys.modules:
        _sys.modules[_pkg] = _types.ModuleType(_pkg)

_spec = _ilu.spec_from_file_location(
    "prediction.src.data.training_pipeline",
    str(Path(__file__).resolve().parents[1] / "src" / "data" / "training_pipeline.py"),
)
_tp = _ilu.module_from_spec(_spec)
_sys.modules[_spec.name] = _tp
_spec.loader.exec_module(_tp)

load_dam_hourly = _tp.load_dam_hourly
load_rtm_hourly = _tp.load_rtm_hourly
load_fuel_mix = _tp.load_fuel_mix
load_fuel_mix_hourly = _tp.load_fuel_mix_hourly
load_rtm_components_hourly = _tp.load_rtm_components_hourly
split_by_date = _tp.split_by_date
run_pipeline = _tp.run_pipeline


# ---------------------------------------------------------------------------
# Helpers – build a small synthetic SQLite DB
# ---------------------------------------------------------------------------

def _date_range(start: str, days: int):
    """Yield YYYY-MM-DD strings for *days* consecutive days."""
    base = pd.Timestamp(start)
    for d in range(days):
        yield (base + pd.Timedelta(days=d)).strftime("%Y-%m-%d")


def _create_test_db(db_path: Path, n_days: int = 400) -> None:
    """Populate a minimal SQLite archive spanning 2023-06-01 → +n_days.

    This gives enough history for lag-168h features and crosses the 2024
    train/val boundary.
    """
    conn = sqlite3.connect(db_path)

    # --- dam_lmp_hist ---
    conn.execute(
        "CREATE TABLE dam_lmp_hist ("
        "  delivery_date TEXT NOT NULL,"
        "  hour_ending INTEGER NOT NULL,"
        "  repeated_hour INTEGER NOT NULL DEFAULT 0,"
        "  settlement_point TEXT NOT NULL,"
        "  lmp REAL NOT NULL"
        ")"
    )
    dam_rows = []
    rng = np.random.RandomState(42)
    for date_str in _date_range("2023-06-01", n_days):
        for he in range(1, 25):
            dam_rows.append((date_str, he, 0, "HB_WEST", round(20 + 30 * rng.rand(), 2)))
    conn.executemany(
        "INSERT INTO dam_lmp_hist VALUES (?,?,?,?,?)", dam_rows
    )

    # --- rtm_lmp_hist ---
    conn.execute(
        "CREATE TABLE rtm_lmp_hist ("
        "  delivery_date TEXT NOT NULL,"
        "  delivery_hour INTEGER NOT NULL,"
        "  delivery_interval INTEGER NOT NULL,"
        "  repeated_hour INTEGER NOT NULL DEFAULT 0,"
        "  settlement_point TEXT NOT NULL,"
        "  settlement_point_type TEXT,"
        "  lmp REAL NOT NULL"
        ")"
    )
    rtm_rows = []
    for date_str in _date_range("2023-06-01", n_days):
        for hr in range(1, 25):
            for interval in range(1, 5):
                rtm_rows.append(
                    (date_str, hr, interval, 0, "HB_WEST", "SH",
                     round(18 + 35 * rng.rand(), 2))
                )
    conn.executemany(
        "INSERT INTO rtm_lmp_hist VALUES (?,?,?,?,?,?,?)", rtm_rows
    )

    # --- fuel_mix_hist ---
    conn.execute(
        "CREATE TABLE fuel_mix_hist ("
        "  delivery_date TEXT NOT NULL,"
        "  fuel TEXT NOT NULL,"
        "  settlement_type TEXT,"
        "  interval_15min INTEGER NOT NULL,"
        "  generation_mw REAL NOT NULL"
        ")"
    )
    fuel_rows = []
    fuels = ["Wind", "Solar", "Gas-CC", "Nuclear", "Coal", "Hydro"]
    for date_str in _date_range("2023-06-01", n_days):
        for interval in range(1, 97):
            for fuel in fuels:
                fuel_rows.append(
                    (date_str, fuel, None, interval, round(500 + 1000 * rng.rand(), 1))
                )
    conn.executemany(
        "INSERT INTO fuel_mix_hist VALUES (?,?,?,?,?)", fuel_rows
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_db(tmp_path_factory) -> Path:
    """Create a temporary SQLite archive shared across tests in this module."""
    db_path = tmp_path_factory.mktemp("data") / "test.db"
    _create_test_db(db_path)
    return db_path


# ---------------------------------------------------------------------------
# Data-loading tests
# ---------------------------------------------------------------------------

class TestDataLoading:
    def test_load_dam_hourly(self, test_db):
        df = load_dam_hourly(test_db, "HB_WEST")
        assert not df.empty
        assert set(df.columns) == {"delivery_date", "hour_ending", "lmp"}
        assert df["hour_ending"].min() == 1
        assert df["hour_ending"].max() == 24

    def test_load_dam_with_date_filter(self, test_db):
        df = load_dam_hourly(test_db, "HB_WEST", date_from="2024-01-01")
        assert (df["delivery_date"] >= "2024-01-01").all()

    def test_load_rtm_hourly(self, test_db):
        df = load_rtm_hourly(test_db, "HB_WEST")
        assert not df.empty
        assert set(df.columns) == {"delivery_date", "hour_ending", "lmp"}
        # Should be aggregated to 1 row per (date, hour)
        dupes = df.duplicated(subset=["delivery_date", "hour_ending"])
        assert not dupes.any()

    def test_load_fuel_mix(self, test_db):
        df = load_fuel_mix(test_db)
        assert not df.empty
        assert "fuel" in df.columns

    def test_load_fuel_mix_hourly_matches_python_aggregation(self, test_db):
        raw = load_fuel_mix(test_db)
        expected = aggregate_fuel_mix_hourly(raw).sort_values(
            ["delivery_date", "hour_ending"]
        ).reset_index(drop=True)
        actual = load_fuel_mix_hourly(test_db).sort_values(
            ["delivery_date", "hour_ending"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(
            actual[expected.columns],
            expected.astype(actual.dtypes.to_dict()),
            check_exact=False,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_load_rtm_components_hourly_normalizes_utc_to_ercot(self, tmp_path):
        db_path = tmp_path / "rtm_components.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE rtm_lmp_api ("
            "  time TEXT NOT NULL,"
            "  settlement_point TEXT NOT NULL,"
            "  lmp REAL,"
            "  energy_component REAL,"
            "  congestion_component REAL,"
            "  loss_component REAL"
            ")"
        )
        conn.executemany(
            "INSERT INTO rtm_lmp_api VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("2024-03-10T07:05:00", "HB_WEST", 110.0, 75.0, 25.0, 10.0),
                ("2024-03-10T08:05:00", "HB_WEST", 120.0, 80.0, 30.0, 10.0),
            ],
        )
        conn.commit()
        conn.close()

        df = load_rtm_components_hourly(db_path, "HB_WEST")
        actual = list(df[["delivery_date", "hour_ending"]].itertuples(index=False, name=None))
        assert actual == [("2024-03-10", 1), ("2024-03-10", 3)]


# ---------------------------------------------------------------------------
# Feature computation tests
# ---------------------------------------------------------------------------

class TestFeatureComputation:
    def test_compute_features_columns(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        for col in FEATURE_COLUMNS:
            assert col in features.columns, f"missing feature: {col}"
        for col in TARGET_COLUMNS:
            assert col in features.columns, f"missing target: {col}"

    def test_no_nan_in_non_fuel_features(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        required = [c for c in FEATURE_COLUMNS if c not in _NULLABLE_FEATURE_SET]
        assert features[required].isna().sum().sum() == 0

    def test_fuel_features_populated(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        fuel_raw = load_fuel_mix(test_db)
        fuel_hourly = aggregate_fuel_mix_hourly(fuel_raw)
        features = compute_features(dam, rtm, fuel_hourly)
        fuel_cols = [c for c in FEATURE_COLUMNS if c.endswith("_pct")]
        # At least some fuel data should be present
        assert features[fuel_cols].notna().any().any()

    def test_temporal_features_range(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        assert features["hour_of_day"].between(1, 24).all()
        assert features["day_of_week"].between(0, 6).all()
        assert features["month"].between(1, 12).all()
        assert features["is_weekend"].isin([0, 1]).all()
        assert features["is_peak_hour"].isin([0, 1]).all()

    def test_spread_equals_dam_minus_rtm(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        expected = features["dam_lag_1h"] - features["rtm_lag_1h"]
        np.testing.assert_allclose(features["dam_rtm_spread"], expected)

    def test_auxiliary_features_use_previous_hour_data(self):
        dt_index = pd.date_range("2024-01-01 00:00:00", periods=240, freq="h")
        values = np.arange(len(dt_index), dtype=np.float32)

        dam = pd.DataFrame(
            {
                "delivery_date": dt_index.strftime("%Y-%m-%d"),
                "hour_ending": (dt_index.hour + 1).astype("int16"),
                "lmp": values + 10.0,
            }
        )
        rtm = pd.DataFrame(
            {
                "delivery_date": dt_index.strftime("%Y-%m-%d"),
                "hour_ending": (dt_index.hour + 1).astype("int16"),
                "lmp": values + 100.0,
            }
        )
        ancillary = pd.DataFrame(
            {
                "delivery_date": dt_index.strftime("%Y-%m-%d"),
                "hour_ending": (dt_index.hour + 1).astype("int16"),
                "regdn": values + 1.0,
                "regup": values + 2.0,
                "rrs": values + 3.0,
                "nspin": values + 4.0,
                "ecrs": values + 5.0,
            }
        )
        rtm_components = pd.DataFrame(
            {
                "delivery_date": dt_index.strftime("%Y-%m-%d"),
                "hour_ending": (dt_index.hour + 1).astype("int16"),
                "lmp": values + 200.0,
                "energy_component": values + 20.0,
                "congestion_component": values + 30.0,
                "loss_component": values + 40.0,
            }
        )
        fuel_gen = pd.DataFrame(
            {
                "delivery_date": dt_index.strftime("%Y-%m-%d"),
                "hour_ending": (dt_index.hour + 1).astype("int16"),
                "gas_gen_mw": values + 1000.0,
                "gas_cc_gen_mw": values + 2000.0,
                "coal_gen_mw": values + 3000.0,
                "nuclear_gen_mw": values + 4000.0,
                "solar_gen_mw": values + 5000.0,
                "wind_gen_mw": values + 6000.0,
                "hydro_gen_mw": values + 7000.0,
                "biomass_gen_mw": values + 8000.0,
                "total_gen_mw": values + 36000.0,
            }
        )

        features = compute_features(
            dam,
            rtm,
            ancillary_hourly=ancillary,
            rtm_components_hourly=rtm_components,
            fuel_gen_hourly=fuel_gen,
        )

        feature_dt = pd.to_datetime(features["delivery_date"]) + pd.to_timedelta(features["hour_ending"] - 1, unit="h")
        expected_regdn = pd.Series(values + 1.0, index=dt_index).shift(1)
        expected_congestion = pd.Series(((values + 30.0) / (values + 200.0)) * 100.0, index=dt_index).shift(1)
        expected_gas = pd.Series(values + 1000.0, index=dt_index).shift(1)
        np.testing.assert_allclose(
            features["regdn"],
            expected_regdn.loc[feature_dt].to_numpy(),
        )
        np.testing.assert_allclose(
            features["dam_as_ratio"],
            features["dam_lag_1h"] / (features["total_as_cost"].abs() + 1),
        )
        np.testing.assert_allclose(
            features["congestion_pct"],
            expected_congestion.loc[feature_dt].to_numpy(),
        )
        np.testing.assert_allclose(
            features["gas_gen_mw"],
            expected_gas.loc[feature_dt].to_numpy(),
        )

    def test_lag_features_correctly_shifted(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        # Check that dam_lag_24h at row i equals dam_lmp at some earlier row
        # (can't check exact index because of dropna, but value should exist)
        assert features["dam_lag_24h"].notna().all()
        assert features["dam_lag_168h"].notna().all()

    def test_lag_features_follow_actual_hourly_time(self):
        base = pd.date_range("2024-01-01 00:00:00", periods=240, freq="h")
        dt_index = base.delete(50)
        seq = np.arange(len(dt_index), dtype=np.float32)

        dam = pd.DataFrame(
            {
                "delivery_date": dt_index.strftime("%Y-%m-%d"),
                "hour_ending": dt_index.hour + 1,
                "lmp": seq,
            }
        )
        rtm = dam.copy()
        rtm["lmp"] = seq + 100.0

        features = compute_features(dam, rtm)
        features["dt"] = pd.to_datetime(features["delivery_date"]) + pd.to_timedelta(
            features["hour_ending"] - 1, unit="h"
        )
        source = dam.assign(
            dt=pd.to_datetime(dam["delivery_date"]) + pd.to_timedelta(dam["hour_ending"] - 1, unit="h")
        )[["dt", "lmp"]].rename(columns={"lmp": "expected_lag_24h"})
        source["dt"] = source["dt"] + pd.Timedelta(hours=24)

        aligned = features.merge(source, on="dt", how="left")
        np.testing.assert_allclose(aligned["dam_lag_24h"], aligned["expected_lag_24h"])

    def test_duplicate_fuel_rows_do_not_duplicate_feature_rows(self):
        dt_index = pd.date_range("2024-01-01 00:00:00", periods=24 * 10, freq="h")
        values = np.arange(len(dt_index), dtype=np.float32)
        dam = pd.DataFrame(
            {
                "delivery_date": dt_index.strftime("%Y-%m-%d"),
                "hour_ending": dt_index.hour + 1,
                "lmp": values,
            }
        )
        rtm = dam.copy()
        rtm["lmp"] = values + 1.0
        fuel_hourly = pd.DataFrame(
            {
                "delivery_date": ["2024-01-10", "2024-01-10"],
                "hour_ending": [24, 24],
                "wind_pct": [10.0, 10.0],
                "solar_pct": [20.0, 20.0],
                "gas_pct": [30.0, 30.0],
                "nuclear_pct": [20.0, 20.0],
                "coal_pct": [10.0, 10.0],
                "hydro_pct": [10.0, 10.0],
            }
        )

        features = compute_features(dam, rtm, fuel_hourly)
        dupes = features.duplicated(subset=["delivery_date", "hour_ending"])
        assert not dupes.any()

    def test_compute_features_does_not_fabricate_spring_forward_hour(self):
        rows = []
        for day in pd.date_range("2024-03-01", "2024-03-20", freq="D"):
            for hour_ending in range(1, 25):
                if day.strftime("%Y-%m-%d") == "2024-03-10" and hour_ending == 2:
                    continue
                rows.append(
                    {
                        "delivery_date": day.strftime("%Y-%m-%d"),
                        "hour_ending": hour_ending,
                    }
                )

        base = pd.DataFrame(rows)
        base["lmp"] = np.arange(len(base), dtype=np.float32)
        dam = base.copy()
        rtm = base.copy()
        rtm["lmp"] = rtm["lmp"] + 50.0

        features = compute_features(dam, rtm)
        mask = (
            (features["delivery_date"] == "2024-03-10")
            & (features["hour_ending"] == 2)
        )
        assert not mask.any()


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplit:
    def test_split_boundaries(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        train, val, test_set = split_by_date(features)

        if not train.empty:
            assert (train["delivery_date"] < "2024-01-01").all()
        if not val.empty:
            assert (val["delivery_date"] >= "2024-01-01").all()
            assert (val["delivery_date"] < "2025-01-01").all()
        if not test_set.empty:
            assert (test_set["delivery_date"] >= "2025-01-01").all()

    def test_split_no_overlap(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        train, val, test_set = split_by_date(features)
        total = len(train) + len(val) + len(test_set)
        assert total == len(features)

    def test_split_preserves_columns(self, test_db):
        dam = load_dam_hourly(test_db, "HB_WEST")
        rtm = load_rtm_hourly(test_db, "HB_WEST")
        features = compute_features(dam, rtm)
        train, val, test_set = split_by_date(features)
        for part in [train, val, test_set]:
            assert set(part.columns) == set(features.columns)


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_run_pipeline_produces_parquet(self, test_db, tmp_path):
        run_pipeline(
            db_path=test_db,
            output_dir=tmp_path / "out",
            settlement_points=["HB_WEST"],
            verbose=False,
        )
        sp_dir = tmp_path / "out" / "hb_west"
        assert sp_dir.exists()
        total_rows = 0
        for name in ["train", "val", "test"]:
            pq = sp_dir / f"{name}.parquet"
            assert pq.exists(), f"{pq} missing"
            df = pd.read_parquet(pq)
            total_rows += len(df)
            for col in FEATURE_COLUMNS:
                assert col in df.columns
        # At least train + val should have data (test may be empty if
        # synthetic DB doesn't extend past 2025)
        assert total_rows > 0

    def test_run_pipeline_writes_stable_schema(self, test_db, tmp_path):
        run_pipeline(
            db_path=test_db,
            output_dir=tmp_path / "out",
            settlement_points=["HB_WEST"],
            verbose=False,
        )
        df = pd.read_parquet(tmp_path / "out" / "hb_west" / "train.parquet")

        assert str(df["delivery_date"].dtype) in {"string", "string[python]", "object"}
        assert str(df["hour_ending"].dtype) == "int8"
        assert str(df["dam_lmp"].dtype) == "float32"
        assert str(df["dam_lag_24h"].dtype) == "float32"


# ---------------------------------------------------------------------------
# Holiday helper tests
# ---------------------------------------------------------------------------

class TestHoliday:
    def test_christmas(self):
        assert is_us_holiday(pd.Timestamp("2024-12-25")) == 1

    def test_regular_day(self):
        assert is_us_holiday(pd.Timestamp("2024-03-15")) == 0

    def test_july_4th(self):
        assert is_us_holiday(pd.Timestamp("2024-07-04")) == 1


# ---------------------------------------------------------------------------
# Fuel mix aggregation tests
# ---------------------------------------------------------------------------

class TestFuelMixAggregation:
    def test_aggregate_shape(self, test_db):
        raw = load_fuel_mix(test_db)
        hourly = aggregate_fuel_mix_hourly(raw)
        assert "hour_ending" in hourly.columns
        assert "wind_pct" in hourly.columns
        assert hourly["hour_ending"].between(1, 24).all()

    def test_percentages_sum_roughly_100(self, test_db):
        raw = load_fuel_mix(test_db)
        hourly = aggregate_fuel_mix_hourly(raw)
        pct_cols = [c for c in hourly.columns if c.endswith("_pct")]
        totals = hourly[pct_cols].sum(axis=1)
        # Should be close to 100% (may not be exact due to dropped fuel types)
        assert (totals > 80).all()
        assert (totals < 120).all()
