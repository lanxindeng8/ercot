"""
测试标签生成模块
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
    """测试 Spike 标签生成"""

    def setUp(self):
        """准备测试数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=500, freq='5min')

        # 创建包含 spike 事件的数据
        base_price = 100 + 50 * np.sin(np.arange(500) * 2 * np.pi / 288)
        spike_boost = np.zeros(500)
        spike_boost[200:220] = 400  # 在 200-220 索引处创建 spike

        self.df = pd.DataFrame({
            'P_CPS': base_price + spike_boost + np.random.normal(0, 10, 500),
            'P_Hub': base_price + np.random.normal(0, 5, 500),
            'P_Houston': base_price + spike_boost * 0.3 + np.random.normal(0, 5, 500),
        }, index=timestamps)

    def test_generate_spike_event(self):
        """测试 SpikeEvent 标签生成"""
        spike_event = SpikeLabels.generate_spike_event(
            self.df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3
        )

        # 检查返回类型
        self.assertIsInstance(spike_event, pd.Series)

        # 检查值范围
        self.assertTrue(spike_event.isin([0, 1]).all())

        # 应该检测到 spike
        self.assertTrue(spike_event.sum() > 0)

    def test_generate_lead_spike(self):
        """测试 LeadSpike 标签生成"""
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

        # 检查返回类型
        self.assertIsInstance(lead_spike, pd.Series)

        # 检查值范围
        self.assertTrue(lead_spike.isin([0, 1]).all())

        # LeadSpike 应该 >= SpikeEvent（因为提前预警）
        self.assertGreaterEqual(lead_spike.sum(), spike_event.sum())

    def test_generate_regime(self):
        """测试 Regime 标签生成"""
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

        # 检查返回类型
        self.assertIsInstance(regime, pd.Series)

        # 检查值范围
        valid_states = ['Normal', 'Tight', 'Scarcity']
        self.assertTrue(regime.isin(valid_states).all())

        # 应该包含所有三种状态（如果数据足够多样化）
        # 至少应该有 Normal 和 Scarcity
        self.assertIn('Normal', regime.values)
        if spike_event.sum() > 0:
            self.assertIn('Scarcity', regime.values)

    def test_spike_event_with_percentile(self):
        """测试使用分位数阈值的 SpikeEvent"""
        spike_event = SpikeLabels.generate_spike_event(
            self.df,
            zone='CPS',
            use_percentile=True,
            percentile_window=7
        )

        # 检查返回类型和值范围
        self.assertIsInstance(spike_event, pd.Series)
        self.assertTrue(spike_event.isin([0, 1]).all())


class TestLabelGenerator(unittest.TestCase):
    """测试标签生成主类"""

    def setUp(self):
        """准备测试数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=1000, freq='5min')

        # 创建包含多个 spike 事件的数据
        base_price = 100 + 50 * np.sin(np.arange(1000) * 2 * np.pi / 288)

        # 创建两个 spike 事件
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
        """测试生成所有标签"""
        generator = LabelGenerator(
            zones=['CPS', 'West', 'Houston'],
            P_hi=400,
            S_hi=50,
            H=60
        )

        labels = generator.generate_all_labels(self.df)

        # 检查返回类型
        self.assertIsInstance(labels, pd.DataFrame)

        # 检查标签列存在
        for zone in ['CPS', 'West', 'Houston']:
            self.assertIn(f'SpikeEvent_{zone}', labels.columns)
            self.assertIn(f'LeadSpike_{zone}_60m', labels.columns)
            self.assertIn(f'Regime_{zone}', labels.columns)

        # 检查数据维度
        self.assertEqual(len(labels), len(self.df))

    def test_identify_spike_events(self):
        """测试识别独立 spike 事件"""
        generator = LabelGenerator(
            zones=['CPS'],
            P_hi=400,
            S_hi=50,
            H=60
        )

        labels = generator.generate_all_labels(self.df)
        spike_event = labels['SpikeEvent_CPS']

        # 识别事件
        events = generator.identify_spike_events(spike_event, min_gap=12)

        # 检查返回类型
        self.assertIsInstance(events, list)

        # 如果有 spike，应该识别出事件
        if spike_event.sum() > 0:
            self.assertGreater(len(events), 0)

            # 检查事件结构
            for event in events:
                self.assertIn('start', event)
                self.assertIn('end', event)
                self.assertIn('duration', event)
                self.assertIn('indices', event)

                # 检查持续时间
                self.assertGreater(event['duration'], 0)
                self.assertEqual(len(event['indices']), event['duration'])

    def test_label_consistency(self):
        """测试标签一致性"""
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

        # Spike 时刻应该都是 Scarcity 状态
        spike_indices = spike_event[spike_event == 1].index
        if len(spike_indices) > 0:
            for idx in spike_indices:
                self.assertEqual(regime[idx], 'Scarcity')

        # LeadSpike 应该覆盖 SpikeEvent
        spike_indices = spike_event[spike_event == 1].index
        for idx in spike_indices:
            # 在 spike 前 60 分钟内应该有 lead spike
            # 这里简化检查：至少在 spike 时刻有 lead spike
            self.assertEqual(lead_spike[idx], 1)

    def test_multiple_zones(self):
        """测试多区域标签生成"""
        generator = LabelGenerator(
            zones=['CPS', 'West', 'Houston']
        )

        labels = generator.generate_all_labels(self.df)

        # 每个区域应该有独立的标签
        for zone in ['CPS', 'West', 'Houston']:
            spike_col = f'SpikeEvent_{zone}'
            self.assertIn(spike_col, labels.columns)

            # 不同区域的 spike 可能不同
            # 这里只检查标签是否有效
            self.assertTrue(labels[spike_col].isin([0, 1]).all())


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def test_no_spike_data(self):
        """测试没有 spike 的数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=100, freq='5min')
        df = pd.DataFrame({
            'P_CPS': np.random.uniform(50, 150, 100),
            'P_Hub': np.random.uniform(50, 150, 100),
        }, index=timestamps)

        spike_event = SpikeLabels.generate_spike_event(
            df,
            zone='CPS',
            P_hi=400,  # 很高的阈值
            S_hi=50
        )

        # 应该没有 spike
        self.assertEqual(spike_event.sum(), 0)

    def test_all_spike_data(self):
        """测试全是 spike 的数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=100, freq='5min')
        df = pd.DataFrame({
            'P_CPS': np.random.uniform(500, 700, 100),  # 全部高价
            'P_Hub': np.random.uniform(50, 100, 100),
        }, index=timestamps)

        spike_event = SpikeLabels.generate_spike_event(
            df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3
        )

        # 大部分应该是 spike
        self.assertGreater(spike_event.sum(), 90)

    def test_short_spike(self):
        """测试短暂 spike（持续时间 < m）"""
        timestamps = pd.date_range(start='2025-12-14', periods=100, freq='5min')
        prices = np.full(100, 100.0)
        prices[50:52] = 500  # 只持续 2 个时间步

        df = pd.DataFrame({
            'P_CPS': prices,
            'P_Hub': np.full(100, 100.0),
        }, index=timestamps)

        spike_event = SpikeLabels.generate_spike_event(
            df,
            zone='CPS',
            P_hi=400,
            S_hi=50,
            m=3  # 需要持续 3 步
        )

        # 不应该检测到 spike（持续时间不足）
        self.assertEqual(spike_event.sum(), 0)


if __name__ == '__main__':
    unittest.main()
