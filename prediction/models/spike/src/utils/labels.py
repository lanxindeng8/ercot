"""
Label Generation Module

Generates SpikeEvent, LeadSpike, and Regime labels according to the design document
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional


class SpikeLabels:
    """Spike event label generator"""

    @staticmethod
    def generate_spike_event(
        df: pd.DataFrame,
        zone: str = 'CPS',
        P_hi: float = 400,
        S_hi: float = 50,
        S_cross_hi: float = 80,
        m: int = 3,
        use_percentile: bool = False,
        percentile_window: int = 30
    ) -> pd.Series:
        """Generate SpikeEvent labels

        Args:
            df: DataFrame containing price and spread data
            zone: Zone name
            P_hi: Price threshold ($/MWh)
            S_hi: Zone-hub spread threshold ($/MWh)
            S_cross_hi: Zone-Houston spread threshold ($/MWh)
            m: Duration threshold (number of time steps)
            use_percentile: Whether to use percentile thresholds
            percentile_window: Rolling window in days (if using percentile)

        Returns:
            SpikeEvent_{zone}: 0/1 label
        """
        # Condition A: High price
        if use_percentile:
            window_size = percentile_window * 24 * 12  # Assumes 5-minute data
            P_threshold = df[f'P_{zone}'].rolling(window=window_size, min_periods=1).quantile(0.99)
            cond_price = df[f'P_{zone}'] >= P_threshold
        else:
            cond_price = df[f'P_{zone}'] >= P_hi

        # Condition B: Large spread (constraint-driven)
        spread_zh = df[f'P_{zone}'] - df['P_Hub']

        if use_percentile:
            window_size = percentile_window * 24 * 12
            S_threshold = spread_zh.rolling(window=window_size, min_periods=1).quantile(0.95)
            cond_spread_zh = spread_zh >= S_threshold
        else:
            cond_spread_zh = spread_zh >= S_hi

        # If Houston data is available, add cross-zone spread condition
        if f'P_Houston' in df.columns and zone != 'Houston':
            spread_cross = df[f'P_{zone}'] - df['P_Houston']
            if use_percentile:
                S_cross_threshold = spread_cross.rolling(
                    window=window_size, min_periods=1
                ).quantile(0.95)
                cond_spread_cross = spread_cross >= S_cross_threshold
            else:
                cond_spread_cross = spread_cross >= S_cross_hi
            cond_spread = cond_spread_zh | cond_spread_cross
        else:
            cond_spread = cond_spread_zh

        # Condition C: Duration
        raw_spike = cond_price & cond_spread
        sustained_spike = raw_spike.rolling(window=m, min_periods=1).sum() >= m

        return sustained_spike.fillna(False).astype(int)

    @staticmethod
    def generate_lead_spike(
        spike_event: pd.Series,
        H: int = 60,
        dt: int = 5
    ) -> pd.Series:
        """Generate LeadSpike label (early warning)

        Args:
            spike_event: SpikeEvent label series
            H: Warning time window (minutes)
            dt: Data time resolution (minutes)

        Returns:
            LeadSpike: 0/1 label
        """
        k = int(H / dt)  # Window size

        # Reverse -> rolling max -> reverse back
        # This allows looking ahead to check for spikes within the future window
        lead_spike = spike_event[::-1].rolling(window=k, min_periods=1).max()[::-1]

        return lead_spike.fillna(0).astype(int)

    @staticmethod
    def generate_regime(
        df: pd.DataFrame,
        zone: str = 'CPS',
        P_mid: float = 150,
        S_mid: float = 20,
        spike_event: Optional[pd.Series] = None
    ) -> pd.Series:
        """Generate Regime state label

        Args:
            df: DataFrame
            zone: Zone name
            P_mid: Tight state price threshold
            S_mid: Tight state spread threshold
            spike_event: Pre-computed SpikeEvent (if provided)

        Returns:
            Regime: 'Normal' / 'Tight' / 'Scarcity' label
        """
        spread_zh = df[f'P_{zone}'] - df['P_Hub']

        # Initialize as Normal
        regime = pd.Series('Normal', index=df.index)

        # Tight state
        tight_cond = (df[f'P_{zone}'] >= P_mid) | (spread_zh >= S_mid)
        regime[tight_cond] = 'Tight'

        # Scarcity state (highest priority)
        if spike_event is not None:
            regime[spike_event == 1] = 'Scarcity'

        return regime


class LabelGenerator:
    """Label generator main class"""

    def __init__(
        self,
        zones: list = ['CPS', 'West', 'Houston'],
        P_hi: float = 400,
        S_hi: float = 50,
        S_cross_hi: float = 80,
        P_mid: float = 150,
        S_mid: float = 20,
        m: int = 3,
        H: int = 60,
        dt: int = 5
    ):
        """Initialize

        Args:
            zones: List of zones
            P_hi: Spike price threshold
            S_hi: Spike spread threshold
            S_cross_hi: Spike cross-zone spread threshold
            P_mid: Tight price threshold
            S_mid: Tight spread threshold
            m: Spike duration threshold
            H: Lead Spike warning window
            dt: Data time interval
        """
        self.zones = zones
        self.P_hi = P_hi
        self.S_hi = S_hi
        self.S_cross_hi = S_cross_hi
        self.P_mid = P_mid
        self.S_mid = S_mid
        self.m = m
        self.H = H
        self.dt = dt

    def generate_all_labels(
        self,
        df: pd.DataFrame,
        use_percentile: bool = False
    ) -> pd.DataFrame:
        """Generate all labels

        Args:
            df: Raw data
            use_percentile: Whether to use rolling percentile thresholds

        Returns:
            DataFrame containing all labels
        """
        labels = pd.DataFrame(index=df.index)

        for zone in self.zones:
            if f'P_{zone}' not in df.columns:
                print(f"Warning: Missing P_{zone} column, skipping {zone} zone")
                continue

            print(f"Generating labels for {zone} zone...")

            # 1. SpikeEvent
            spike_event = SpikeLabels.generate_spike_event(
                df,
                zone=zone,
                P_hi=self.P_hi,
                S_hi=self.S_hi,
                S_cross_hi=self.S_cross_hi,
                m=self.m,
                use_percentile=use_percentile
            )
            labels[f'SpikeEvent_{zone}'] = spike_event

            # 2. LeadSpike
            lead_spike = SpikeLabels.generate_lead_spike(
                spike_event,
                H=self.H,
                dt=self.dt
            )
            labels[f'LeadSpike_{zone}_{self.H}m'] = lead_spike

            # 3. Regime
            regime = SpikeLabels.generate_regime(
                df,
                zone=zone,
                P_mid=self.P_mid,
                S_mid=self.S_mid,
                spike_event=spike_event
            )
            labels[f'Regime_{zone}'] = regime

            # Statistics
            n_spikes = spike_event.sum()
            n_lead_spikes = lead_spike.sum()
            regime_counts = regime.value_counts()

            print(f"  - SpikeEvent count: {n_spikes}")
            print(f"  - LeadSpike count: {n_lead_spikes}")
            print(f"  - Regime distribution:")
            for state, count in regime_counts.items():
                print(f"    {state}: {count} ({count/len(regime)*100:.2f}%)")

        return labels

    def identify_spike_events(
        self,
        spike_event: pd.Series,
        min_gap: int = 12  # Minimum gap (time steps) for separating independent events
    ) -> list:
        """Identify independent spike events

        Args:
            spike_event: SpikeEvent label series
            min_gap: Minimum gap; spikes closer than this are treated as the same event

        Returns:
            List of events, each event is {'start': timestamp, 'end': timestamp, 'duration': int, 'max_idx': timestamp}
        """
        events = []
        in_event = False
        event_start = None
        event_indices = []

        for idx, value in spike_event.items():
            if value == 1:
                if not in_event:
                    in_event = True
                    event_start = idx
                    event_indices = [idx]
                else:
                    event_indices.append(idx)
            else:
                if in_event:
                    # Check whether the event should end
                    if len(event_indices) > 0:
                        gap = (idx - event_indices[-1]).total_seconds() / 60 / self.dt
                        if gap > min_gap:
                            # Record the event
                            events.append({
                                'start': event_start,
                                'end': event_indices[-1],
                                'duration': len(event_indices),
                                'indices': event_indices.copy()
                            })
                            in_event = False
                            event_indices = []

        # Handle the last event
        if in_event and len(event_indices) > 0:
            events.append({
                'start': event_start,
                'end': event_indices[-1],
                'duration': len(event_indices),
                'indices': event_indices.copy()
            })

        return events


if __name__ == '__main__':
    # Test code
    print("Label generation module loaded successfully!")
    print("\nSupported label types:")
    print("1. SpikeEvent - Spike event indicator")
    print("2. LeadSpike - Early warning label")
    print("3. Regime - System state label (Normal/Tight/Scarcity)")
