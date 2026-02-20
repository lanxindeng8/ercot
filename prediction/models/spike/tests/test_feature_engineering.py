"""
测试特征工程模块
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.feature_engineering import (
    PriceStructureFeatures,
    SupplyDemandFeatures,
    WeatherFeatures,
    TemporalFeatures,
    FeatureEngineer
)


class TestPriceStructureFeatures(unittest.TestCase):
    """测试价格结构特征"""

    def setUp(self):
        """准备测试数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=100, freq='5min')
        self.df = pd.DataFrame({
            'P_CPS': np.random.uniform(50, 200, 100),
            'P_West': np.random.uniform(50, 200, 100),
            'P_Houston': np.random.uniform(50, 200, 100),
            'P_Hub': np.random.uniform(50, 150, 100),
            'P_CPS_DA': np.random.uniform(50, 150, 100),
            'P_West_DA': np.random.uniform(50, 150, 100),
            'P_Houston_DA': np.random.uniform(50, 150, 100),
        }, index=timestamps)

    def test_calculate_features(self):
        """测试特征计算"""
        features = PriceStructureFeatures.calculate(self.df)

        # 检查返回类型
        self.assertIsInstance(features, pd.DataFrame)

        # 检查特征列存在
        self.assertIn('spread_CPS_hub', features.columns)
        self.assertIn('spread_rt_da_CPS', features.columns)
        self.assertIn('price_ramp_5m_CPS', features.columns)
        self.assertIn('price_accel_CPS', features.columns)
        self.assertIn('spread_CPS_Houston', features.columns)

        # 检查数据维度
        self.assertEqual(len(features), len(self.df))

    def test_spread_calculation(self):
        """测试价差计算"""
        features = PriceStructureFeatures.calculate(self.df)

        # 验证价差计算正确性
        expected_spread = self.df['P_CPS'] - self.df['P_Hub']
        pd.testing.assert_series_equal(
            features['spread_CPS_hub'],
            expected_spread,
            check_names=False
        )


class TestSupplyDemandFeatures(unittest.TestCase):
    """测试供需平衡特征"""

    def setUp(self):
        """准备测试数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=1000, freq='5min')
        self.df = pd.DataFrame({
            'Load': np.random.uniform(35000, 45000, 1000),
            'Wind': np.random.uniform(5000, 12000, 1000),
            'Solar': np.random.uniform(0, 8000, 1000),
            'Gas': np.random.uniform(20000, 30000, 1000),
            'Coal': np.random.uniform(7000, 9000, 1000),
            'ESR': np.random.uniform(-2000, 2000, 1000),
        }, index=timestamps)

    def test_calculate_features(self):
        """测试特征计算"""
        features = SupplyDemandFeatures.calculate(self.df, lookback_days=7)

        # 检查返回类型
        self.assertIsInstance(features, pd.DataFrame)

        # 检查特征列存在
        self.assertIn('net_load', features.columns)
        self.assertIn('wind_anomaly', features.columns)
        self.assertIn('gas_saturation', features.columns)
        self.assertIn('coal_stress', features.columns)
        self.assertIn('esr_net_output', features.columns)

    def test_net_load_calculation(self):
        """测试净负荷计算"""
        features = SupplyDemandFeatures.calculate(self.df)

        # 验证净负荷计算
        expected_net_load = self.df['Load'] - self.df['Wind'] - self.df['Solar']
        pd.testing.assert_series_equal(
            features['net_load'],
            expected_net_load,
            check_names=False
        )

    def test_coal_stress(self):
        """测试煤电压力标志"""
        features = SupplyDemandFeatures.calculate(self.df)

        # 煤电压力应该是 0 或 1
        self.assertTrue(features['coal_stress'].isin([0, 1]).all())


class TestWeatherFeatures(unittest.TestCase):
    """测试天气驱动特征"""

    def setUp(self):
        """准备测试数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=500, freq='15min')
        self.df = pd.DataFrame({
            'T_CPS': np.random.uniform(50, 80, 500),
            'T_West': np.random.uniform(50, 80, 500),
            'T_Houston': np.random.uniform(55, 85, 500),
            'WindSpeed_CPS': np.random.uniform(5, 20, 500),
            'WindSpeed_West': np.random.uniform(5, 20, 500),
            'WindSpeed_Houston': np.random.uniform(3, 15, 500),
            'WindDir_CPS': np.random.uniform(0, 360, 500),
            'WindDir_West': np.random.uniform(0, 360, 500),
            'WindDir_Houston': np.random.uniform(0, 360, 500),
        }, index=timestamps)

    def test_calculate_features(self):
        """测试特征计算"""
        features = WeatherFeatures.calculate(self.df)

        # 检查返回类型
        self.assertIsInstance(features, pd.DataFrame)

        # 检查特征列存在
        self.assertIn('T_anomaly_CPS', features.columns)
        self.assertIn('T_ramp_CPS', features.columns)
        self.assertIn('WindChill_CPS', features.columns)
        self.assertIn('ColdFront_CPS', features.columns)

    def test_wind_chill_calculation(self):
        """测试风寒指数计算"""
        features = WeatherFeatures.calculate(self.df)

        # 风寒指数不应该有 NaN（除非输入数据有 NaN）
        self.assertFalse(features['WindChill_CPS'].isna().any())

    def test_cold_front_flag(self):
        """测试冷锋标志"""
        features = WeatherFeatures.calculate(self.df)

        # 冷锋标志应该是 0 或 1
        self.assertTrue(features['ColdFront_CPS'].isin([0, 1]).all())


class TestTemporalFeatures(unittest.TestCase):
    """测试时间特征"""

    def setUp(self):
        """准备测试数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=288, freq='5min')  # 1天
        self.df = pd.DataFrame({
            'Solar': np.random.uniform(0, 8000, 288),
        }, index=timestamps)

    def test_calculate_features(self):
        """测试特征计算"""
        features = TemporalFeatures.calculate(self.df)

        # 检查返回类型
        self.assertIsInstance(features, pd.DataFrame)

        # 检查特征列存在
        self.assertIn('hour', features.columns)
        self.assertIn('is_evening_peak', features.columns)
        self.assertIn('minutes_to_sunrise', features.columns)

    def test_hour_range(self):
        """测试小时范围"""
        features = TemporalFeatures.calculate(self.df)

        # 小时应该在 0-23 范围内
        self.assertTrue((features['hour'] >= 0).all())
        self.assertTrue((features['hour'] <= 23).all())

    def test_evening_peak_flag(self):
        """测试晚高峰标志"""
        features = TemporalFeatures.calculate(self.df)

        # 晚高峰标志应该是 0 或 1
        self.assertTrue(features['is_evening_peak'].isin([0, 1]).all())

        # 验证晚高峰时段
        peak_hours = features[features['is_evening_peak'] == 1]['hour'].unique()
        self.assertTrue(all(h in range(17, 23) for h in peak_hours))


class TestFeatureEngineer(unittest.TestCase):
    """测试特征工程主类"""

    def setUp(self):
        """准备完整测试数据"""
        timestamps = pd.date_range(start='2025-12-14', periods=500, freq='5min')
        self.df = pd.DataFrame({
            # 价格数据
            'P_CPS': np.random.uniform(50, 200, 500),
            'P_West': np.random.uniform(50, 200, 500),
            'P_Houston': np.random.uniform(50, 200, 500),
            'P_Hub': np.random.uniform(50, 150, 500),
            'P_CPS_DA': np.random.uniform(50, 150, 500),
            'P_West_DA': np.random.uniform(50, 150, 500),
            'P_Houston_DA': np.random.uniform(50, 150, 500),

            # 系统数据
            'Load': np.random.uniform(35000, 45000, 500),
            'Wind': np.random.uniform(5000, 12000, 500),
            'Solar': np.random.uniform(0, 8000, 500),
            'Gas': np.random.uniform(20000, 30000, 500),
            'Coal': np.random.uniform(7000, 9000, 500),
            'ESR': np.random.uniform(-2000, 2000, 500),

            # 天气数据
            'T_CPS': np.random.uniform(50, 80, 500),
            'T_West': np.random.uniform(50, 80, 500),
            'T_Houston': np.random.uniform(55, 85, 500),
            'WindSpeed_CPS': np.random.uniform(5, 20, 500),
            'WindSpeed_West': np.random.uniform(5, 20, 500),
            'WindSpeed_Houston': np.random.uniform(3, 15, 500),
            'WindDir_CPS': np.random.uniform(0, 360, 500),
            'WindDir_West': np.random.uniform(0, 360, 500),
            'WindDir_Houston': np.random.uniform(0, 360, 500),
        }, index=timestamps)

    def test_calculate_all_features(self):
        """测试计算所有特征"""
        engineer = FeatureEngineer(zones=['CPS', 'West', 'Houston'])
        result = engineer.calculate_all_features(self.df)

        # 检查返回类型
        self.assertIsInstance(result, pd.DataFrame)

        # 检查原始列保留
        for col in self.df.columns:
            self.assertIn(col, result.columns)

        # 检查特征列存在
        self.assertIn('spread_CPS_hub', result.columns)
        self.assertIn('net_load', result.columns)
        self.assertIn('T_anomaly_CPS', result.columns)
        self.assertIn('hour', result.columns)

    def test_get_feature_names(self):
        """测试获取特征名称"""
        engineer = FeatureEngineer(zones=['CPS', 'West', 'Houston'])

        # 获取各类特征名称
        price_features = engineer.get_feature_names('price')
        supply_features = engineer.get_feature_names('supply_demand')
        weather_features = engineer.get_feature_names('weather')
        temporal_features = engineer.get_feature_names('temporal')

        # 检查类型
        self.assertIsInstance(price_features, list)
        self.assertIsInstance(supply_features, list)
        self.assertIsInstance(weather_features, list)
        self.assertIsInstance(temporal_features, list)

        # 检查特征数量
        self.assertGreater(len(price_features), 0)
        self.assertGreater(len(supply_features), 0)
        self.assertGreater(len(weather_features), 0)
        self.assertGreater(len(temporal_features), 0)

        # 获取所有特征
        all_features = engineer.get_feature_names()
        total = len(price_features) + len(supply_features) + len(weather_features) + len(temporal_features)
        self.assertEqual(len(all_features), total)


if __name__ == '__main__':
    unittest.main()
