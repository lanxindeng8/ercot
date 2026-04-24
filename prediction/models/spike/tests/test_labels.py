"""
Tests for the label generation module
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from src.utils.labels import SpikeLabels, LabelGenerator


class TestSpikeLabels(unittest.TestCase):
    """Test Spike label generation"""

    def setUp(self):
        """Prepare test data"""
        timestamps = pd.date_range(start='2025-12-14', periods=500, freq='5min')

        # Create data containing spike events
        base_price = 100 + 50 * np.sin(np.arange(500) * 2 * np.pi / 288)
        spike_boost = np.zeros(500)
        spike_boost[200:220] = 400  # Create spike at indices 200-220

        self.df = pd.DataFrame({
            'P_CPS': base_price + spike_boost + np.random.normal(0, 10, 500),
            'P_Hub': base_price + np.random.normal(0, 5, 500),
            'P_Houston': base_price + spike_boost * 0.3 + np.random.normal(0, 5, 500),
        }, index=timestamps)

    def test_generate_spike_event(self):
        """Test SpikeEvent label generation"""
        spike_event = SpikeLabels.generate_spike_event(
            self.df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3
        )

        # Check return type
        self.assertIsInstance(spike_event, pd.Series)

        # Check value range
        self.assertTrue(spike_event.isin([0, 1]).all())

        # Should detect spikes
        self.assertTrue(spike_event.sum() > 0)

    def test_generate_lead_spike(self):
        """Test LeadSpike label generation"""
        spike_event = SpikeLabels.generate_spike_event(
            self.df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3
        )

        lead_spike = SpikeLabels.generate_lead_spike(
            spike_event,
            H=60,
            dt=5
        )

        # Check return type
        self.assertIsInstance(lead_spike, pd.Series)

        # Check value range
        self.assertTrue(lead_spike.isin([0, 1]).all())

        # LeadSpike should >= SpikeEvent (due to early warning)
        self.assertGreaterEqual(lead_spike.sum(), spike_event.sum())

    def test_generate_regime(self):
        """Test Regime label generation"""
        spike_event = SpikeLabels.generate_spike_event(
            self.df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3
        )

        regime = SpikeLabels.generate_regime(
            self.df,
            zone='CPS',
            P_mid=150,
            S_mid=20,
            spike_event=spike_event
        )

        # Check return type
        self.assertIsInstance(regime, pd.Series)

        # Check value range
        valid_states = ['Normal', 'Tight', 'Scarcity']
        self.assertTrue(regime.isin(valid_states).all())

        # Should contain all three states (if data is sufficiently diverse)
        # At minimum, should have Normal and Scarcity
        self.assertIn('Normal', regime.values)
        if spike_event.sum() > 0:
            self.assertIn('Scarcity', regime.values)

    def test_spike_event_with_percentile(self):
        """Test SpikeEvent with percentile thresholds"""
        spike_event = SpikeLabels.generate_spike_event(
            self.df,
            zone='CPS',
            use_percentile=True,
            percentile_window=7
        )

        # Check return type and value range
        self.assertIsInstance(spike_event, pd.Series)
        self.assertTrue(spike_event.isin([0, 1]).all())


class TestLabelGenerator(unittest.TestCase):
    """Test label generator main class"""

    def setUp(self):
        """Prepare test data"""
        timestamps = pd.date_range(start='2025-12-14', periods=1000, freq='5min')

        # Create data containing multiple spike events
        base_price = 100 + 50 * np.sin(np.arange(1000) * 2 * np.pi / 288)

        # Create two spike events
        spike_boost = np.zeros(1000)
        spike_boost[200:220] = 400  # Spike 1
        spike_boost[600:615] = 500  # Spike 2

        self.df = pd.DataFrame({
            'P_CPS': base_price + spike_boost + np.random.normal(0, 10, 1000),
            'P_West': base_price + spike_boost * 0.8 + np.random.normal(0, 10, 1000),
            'P_Houston': base_price + spike_boost * 0.3 + np.random.normal(0, 10, 1000),
            'P_Hub': base_price + np.random.normal(0, 5, 1000),
        }, index=timestamps)

    def test_generate_all_labels(self):
        """Test generating all labels"""
        generator = LabelGenerator(
            zones=['CPS', 'West', 'Houston'],
            P_hi=400,
            S_hi=50,
            H=60
        )

        labels = generator.generate_all_labels(self.df)

        # Check return type
        self.assertIsInstance(labels, pd.DataFrame)

        # Check label columns exist
        for zone in ['CPS', 'West', 'Houston']:
            self.assertIn(f'SpikeEvent_{zone}', labels.columns)
            self.assertIn(f'LeadSpike_{zone}_60m', labels.columns)
            self.assertIn(f'Regime_{zone}', labels.columns)

        # Check data dimensions
        self.assertEqual(len(labels), len(self.df))

    def test_identify_spike_events(self):
        """Test identifying independent spike events"""
        generator = LabelGenerator(
            zones=['CPS'],
            P_hi=400,
            S_hi=50,
            H=60
        )

        labels = generator.generate_all_labels(self.df)
        spike_event = labels['SpikeEvent_CPS']

        # Identify events
        events = generator.identify_spike_events(spike_event, min_gap=12)

        # Check return type
        self.assertIsInstance(events, list)

        # If there are spikes, events should be identified
        if spike_event.sum() > 0:
            self.assertGreater(len(events), 0)

            # Check event structure
            for event in events:
                self.assertIn('start', event)
                self.assertIn('end', event)
                self.assertIn('duration', event)
                self.assertIn('indices', event)

                # Check duration
                self.assertGreater(event['duration'], 0)
                self.assertEqual(len(event['indices']), event['duration'])

    def test_label_consistency(self):
        """Test label consistency"""
        generator = LabelGenerator(
            zones=['CPS'],
            P_hi=400,
            S_hi=50,
            H=60,
            dt=5
        )

        labels = generator.generate_all_labels(self.df)

        spike_event = labels['SpikeEvent_CPS']
        lead_spike = labels['LeadSpike_CPS_60m']
        regime = labels['Regime_CPS']

        # Spike time steps should all be in Scarcity state
        spike_indices = spike_event[spike_event == 1].index
        if len(spike_indices) > 0:
            for idx in spike_indices:
                self.assertEqual(regime[idx], 'Scarcity')

        # LeadSpike should cover SpikeEvent
        spike_indices = spike_event[spike_event == 1].index
        for idx in spike_indices:
            # Within 60 minutes before a spike there should be a lead spike
            # Simplified check here: at least at the spike moment there is a lead spike
            self.assertEqual(lead_spike[idx], 1)

    def test_multiple_zones(self):
        """Test multi-zone label generation"""
        generator = LabelGenerator(
            zones=['CPS', 'West', 'Houston']
        )

        labels = generator.generate_all_labels(self.df)

        # Each zone should have independent labels
        for zone in ['CPS', 'West', 'Houston']:
            spike_col = f'SpikeEvent_{zone}'
            self.assertIn(spike_col, labels.columns)

            # Different zones may have different spikes
            # Here we only check that labels are valid
            self.assertTrue(labels[spike_col].isin([0, 1]).all())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases"""

    def test_no_spike_data(self):
        """Test data with no spikes"""
        timestamps = pd.date_range(start='2025-12-14', periods=100, freq='5min')
        df = pd.DataFrame({
            'P_CPS': np.random.uniform(50, 150, 100),
            'P_Hub': np.random.uniform(50, 150, 100),
        }, index=timestamps)

        spike_event = SpikeLabels.generate_spike_event(
            df,
            zone='CPS',
            P_hi=400,  # Very high threshold
            S_hi=50
        )

        # Should have no spikes
        self.assertEqual(spike_event.sum(), 0)

    def test_all_spike_data(self):
        """Test data that is entirely spikes"""
        timestamps = pd.date_range(start='2025-12-14', periods=100, freq='5min')
        df = pd.DataFrame({
            'P_CPS': np.random.uniform(500, 700, 100),  # All high prices
            'P_Hub': np.random.uniform(50, 100, 100),
        }, index=timestamps)

        spike_event = SpikeLabels.generate_spike_event(
            df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3
        )

        # Most should be spikes
        self.assertGreater(spike_event.sum(), 90)

    def test_short_spike(self):
        """Test short-lived spike (duration < m)"""
        timestamps = pd.date_range(start='2025-12-14', periods=100, freq='5min')
        prices = np.full(100, 100.0)
        prices[50:52] = 500  # Only lasts 2 time steps

        df = pd.DataFrame({
            'P_CPS': prices,
            'P_Hub': np.full(100, 100.0),
        }, index=timestamps)

        spike_event = SpikeLabels.generate_spike_event(
            df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3  # Requires 3 consecutive steps
        )

        # Should not detect a spike (insufficient duration)
        self.assertEqual(spike_event.sum(), 0)


if __name__ == '__main__':
    unittest.main()
