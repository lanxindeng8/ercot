"""Tests for the three-layer spike labeling system."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction.src.labels.spike_labels import (
    compute_spike_events,
    compute_lead_spike,
    compute_regime,
    _ercot_rows_to_utc,
)

DB_PATH = Path(__file__).resolve().parent.parent.parent / "scraper" / "data" / "ercot_archive.db"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(prices_z, prices_hub, freq="15min", start="2025-01-01"):
    """Build a simple DataFrame with zone + hub LMP columns."""
    idx = pd.date_range(start, periods=len(prices_z), freq=freq, tz="UTC")
    return pd.DataFrame({"LZ_TEST": prices_z, "HB_HUBAVG": prices_hub}, index=idx)


# ---------------------------------------------------------------------------
# Layer 1: SpikeEvent
# ---------------------------------------------------------------------------

class TestSpikeEvents:
    def test_synthetic_spike_detected(self):
        """Prices above p_hi with large spread → spike detected."""
        n = 100
        prices_z = np.full(n, 30.0)
        prices_hub = np.full(n, 25.0)
        # Create a spike at intervals 50-55
        prices_z[50:56] = 500.0
        prices_hub[50:56] = 50.0  # spread = 450

        df = _make_df(prices_z, prices_hub)
        result = compute_spike_events(df, "LZ_TEST", p_hi=400.0, s_hi=50.0, min_consecutive=2)

        assert result.iloc[50:56].all(), "Spike intervals should be flagged"
        assert not result.iloc[:50].any(), "Pre-spike should be clean"

    def test_consecutive_filter_rejects_single_interval(self):
        """A single-interval spike should NOT be flagged with min_consecutive=2."""
        n = 50
        prices_z = np.full(n, 30.0)
        prices_hub = np.full(n, 25.0)
        # Single spike at interval 20
        prices_z[20] = 500.0
        prices_hub[20] = 50.0

        df = _make_df(prices_z, prices_hub)
        result = compute_spike_events(df, "LZ_TEST", p_hi=400.0, s_hi=50.0, min_consecutive=2)

        assert not result.any(), "Single-interval spike should be filtered out"

    def test_consecutive_filter_accepts_exactly_min(self):
        """Exactly min_consecutive intervals should be flagged."""
        n = 50
        prices_z = np.full(n, 30.0)
        prices_hub = np.full(n, 25.0)
        prices_z[20:22] = 500.0
        prices_hub[20:22] = 50.0

        df = _make_df(prices_z, prices_hub)
        result = compute_spike_events(df, "LZ_TEST", p_hi=400.0, s_hi=50.0, min_consecutive=2)

        assert result.iloc[20:22].all()
        assert result.sum() == 2

    def test_spread_condition_required(self):
        """High price but small spread → no spike."""
        n = 50
        prices_z = np.full(n, 500.0)
        prices_hub = np.full(n, 490.0)  # spread = 10 < s_hi=50

        df = _make_df(prices_z, prices_hub)
        result = compute_spike_events(df, "LZ_TEST", p_hi=400.0, s_hi=50.0, min_consecutive=2)

        assert not result.any(), "Small spread should prevent spike flag"

    def test_rolling_q99_triggers_spike(self):
        """Price below p_hi but above rolling Q99 should trigger (with spread)."""
        n = 3000  # need 2880+ for a full 30-day window
        prices_z = np.full(n, 30.0)
        prices_hub = np.full(n, 25.0)
        # At the end, price rises to 35 — above Q99 of a flat 30 series,
        # but below p_hi=400. Add spread.
        prices_z[-5:] = 200.0
        prices_hub[-5:] = 25.0  # spread = 175 > 50

        df = _make_df(prices_z, prices_hub)
        result = compute_spike_events(df, "LZ_TEST", p_hi=400.0, s_hi=50.0, min_consecutive=2)

        # 200 should be well above Q99 of a flat-30 series
        assert result.iloc[-5:].all(), "Q99 exceedance with spread should flag spike"

    def test_no_lookahead(self):
        """SpikeEvent uses only current-time data — no future leakage.

        Verify by checking that adding future spikes doesn't change past labels.
        """
        n = 100
        prices_z = np.full(n, 30.0)
        prices_hub = np.full(n, 25.0)

        df1 = _make_df(prices_z.copy(), prices_hub.copy())
        result1 = compute_spike_events(df1, "LZ_TEST", min_consecutive=1)

        # Add a future spike
        prices_z2 = prices_z.copy()
        prices_z2[80:85] = 500.0
        prices_hub2 = prices_hub.copy()
        prices_hub2[80:85] = 50.0
        df2 = _make_df(prices_z2, prices_hub2)
        result2 = compute_spike_events(df2, "LZ_TEST", min_consecutive=1)

        # First 80 intervals should be identical
        pd.testing.assert_series_equal(result1.iloc[:80], result2.iloc[:80])


# ---------------------------------------------------------------------------
# Layer 2: LeadSpike
# ---------------------------------------------------------------------------

class TestLeadSpike:
    def test_lead_spike_before_event(self):
        """LeadSpike should be True in the horizon window BEFORE the spike."""
        n = 100
        spike_events = pd.Series(False, index=pd.date_range(
            "2025-01-01", periods=n, freq="15min", tz="UTC"
        ))
        # Spike at intervals 50-55
        spike_events.iloc[50:56] = True

        lead = compute_lead_spike(spike_events, horizon_minutes=60, interval_minutes=15)

        # 60 min / 15 min = 4 steps ahead
        # lead_spike should be True at t=46..49 (4 steps before spike start at 50)
        assert lead.iloc[46:50].all(), "Should flag 4 intervals before spike"

    def test_lead_spike_not_after_event(self):
        """LeadSpike should NOT be True after the last spike interval."""
        n = 100
        spike_events = pd.Series(False, index=pd.date_range(
            "2025-01-01", periods=n, freq="15min", tz="UTC"
        ))
        spike_events.iloc[50:52] = True

        lead = compute_lead_spike(spike_events, horizon_minutes=60, interval_minutes=15)

        # After interval 55 (spike at 50-51, horizon = 4 steps),
        # nothing should be flagged
        assert not lead.iloc[56:].any(), "Should not flag intervals well after spike"

    def test_lead_spike_excludes_current(self):
        """LeadSpike at t should not include spike at t itself."""
        n = 20
        spike_events = pd.Series(False, index=pd.date_range(
            "2025-01-01", periods=n, freq="15min", tz="UTC"
        ))
        spike_events.iloc[10] = True

        lead = compute_lead_spike(spike_events, horizon_minutes=15, interval_minutes=15)

        # With horizon=1 step, lead_spike at t=10 should be False (spike is at t=10, not future)
        # lead_spike at t=9 should be True
        assert lead.iloc[9], "One interval before should be True"
        assert not lead.iloc[10], "Current spike interval should not self-flag"

    def test_lead_spike_is_forward_looking(self):
        """LeadSpike uses future information — this is intentional (training target only).

        Verify it responds to future data that SpikeEvent doesn't see.
        """
        n = 50
        spike_events = pd.Series(False, index=pd.date_range(
            "2025-01-01", periods=n, freq="15min", tz="UTC"
        ))
        spike_events.iloc[40] = True

        lead = compute_lead_spike(spike_events, horizon_minutes=60, interval_minutes=15)

        # Intervals 36-39 should be True (looking 4 steps ahead to see spike at 40)
        assert lead.iloc[36:40].all()
        # Interval 35 should be False (5 steps before, beyond horizon)
        assert not lead.iloc[35]


# ---------------------------------------------------------------------------
# Layer 3: Regime
# ---------------------------------------------------------------------------

class TestRegime:
    def test_normal_regime(self):
        """Low price, low spread → Normal."""
        df = _make_df([30.0] * 10, [28.0] * 10)
        regime = compute_regime(df, "LZ_TEST")
        assert (regime == "Normal").all()

    def test_tight_regime_price(self):
        """Price >= p_mid → Tight."""
        df = _make_df([160.0] * 10, [155.0] * 10)
        regime = compute_regime(df, "LZ_TEST", p_mid=150.0, s_mid=20.0)
        assert (regime == "Tight").all()

    def test_tight_regime_spread(self):
        """Large spread → Tight even with low price."""
        df = _make_df([50.0] * 10, [25.0] * 10)  # spread=25 >= s_mid=20
        regime = compute_regime(df, "LZ_TEST", p_mid=150.0, s_mid=20.0)
        assert (regime == "Tight").all()

    def test_scarcity_overrides_tight(self):
        """Spike events → Scarcity, overriding Tight."""
        df = _make_df([200.0] * 10, [100.0] * 10)
        spike_ev = pd.Series([False] * 5 + [True] * 5, index=df.index)
        regime = pd.Series(compute_regime(df, "LZ_TEST", spike_events=spike_ev), index=df.index)

        assert (regime.iloc[:5] == "Tight").all()
        assert (regime.iloc[5:] == "Scarcity").all()

    def test_regime_categories(self):
        """Regime should be a Categorical with correct categories."""
        df = _make_df([30.0] * 5, [28.0] * 5)
        regime = compute_regime(df, "LZ_TEST")
        assert list(regime.categories) == ["Normal", "Tight", "Scarcity"]


# ---------------------------------------------------------------------------
# Integration: real data — 2025-12-14 LZ_CPS case day
# ---------------------------------------------------------------------------

class TestRealData:
    @pytest.fixture
    def case_day_data(self):
        """Load LZ_CPS and HB_HUBAVG for 2025-12-14 from the archive DB."""
        if not DB_PATH.exists():
            pytest.skip(f"Archive DB not found: {DB_PATH}")

        conn = sqlite3.connect(str(DB_PATH))
        raw = pd.read_sql_query(
            """
            SELECT delivery_date, delivery_hour, delivery_interval,
                   settlement_point, lmp
            FROM rtm_lmp_hist
            WHERE delivery_date BETWEEN '2025-12-01' AND '2025-12-31'
              AND settlement_point IN ('LZ_CPS', 'HB_HUBAVG')
              AND repeated_hour = 0
            ORDER BY delivery_date, delivery_hour, delivery_interval
            """,
            conn,
        )
        conn.close()

        raw = _ercot_rows_to_utc(raw)
        raw = raw.dropna(subset=["time"])

        wide = raw.pivot_table(
            index="time", columns="settlement_point", values="lmp", aggfunc="first"
        ).sort_index()

        return wide

    def test_lz_cps_spikes_detected_2025_12_14(self, case_day_data):
        """LZ_CPS hit $686 at hour 21 CT on 2025-12-14 — should have spike events.

        Hours 20-22 CT = UTC hours 02-04 on 2025-12-15 (CST = UTC-6).
        """
        df = case_day_data
        spike_ev = compute_spike_events(df, "LZ_CPS", p_hi=400.0, s_hi=50.0, min_consecutive=2)

        # Filter to 2025-12-15 UTC hours 02-04 (= 2025-12-14 CT hours 20-22)
        mask = (
            (df.index >= "2025-12-15 02:00:00+00:00")
            & (df.index < "2025-12-15 04:00:00+00:00")
        )
        spike_window = spike_ev[mask]

        assert spike_window.any(), (
            f"Expected spike events during hours 20-22 CT on 2025-12-14. "
            f"LZ_CPS prices in window: {df.loc[mask, 'LZ_CPS'].tolist()}"
        )
        # At least 4 intervals should be flagged (prices sustained > $400)
        assert spike_window.sum() >= 4, f"Expected >= 4 spike intervals, got {spike_window.sum()}"

    def test_regime_scarcity_during_spike(self, case_day_data):
        """Regime should be Scarcity during the 2025-12-14 spike window."""
        df = case_day_data
        spike_ev = compute_spike_events(df, "LZ_CPS")
        regime = compute_regime(df, "LZ_CPS", spike_events=spike_ev)

        mask = (
            (df.index >= "2025-12-15 02:00:00+00:00")
            & (df.index < "2025-12-15 04:00:00+00:00")
        )
        scarcity_count = (regime[mask] == "Scarcity").sum()
        assert scarcity_count >= 4, f"Expected >= 4 Scarcity intervals, got {scarcity_count}"
