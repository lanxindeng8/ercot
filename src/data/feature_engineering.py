"""
特征工程模块

实现 ERCOT RTM LMP Spike 预测的特征计算
包括：价格结构、供需平衡、天气驱动、时间特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class PriceStructureFeatures:
    """价格结构特征计算

    目标：捕捉区域性稀缺与拥塞信号
    """

    @staticmethod
    def calculate(df: pd.DataFrame, zones: List[str] = ['CPS', 'West', 'Houston']) -> pd.DataFrame:
        """计算价格结构特征

        Args:
            df: 包含价格数据的 DataFrame
                必需列: P_CPS, P_West, P_Houston, P_Hub, P_CPS_DA, P_West_DA, P_Houston_DA
            zones: 区域列表

        Returns:
            包含价格结构特征的 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        for zone in zones:
            # 1. 区域-系统价差
            features[f'spread_{zone}_hub'] = df[f'P_{zone}'] - df['P_Hub']

            # 2. 实时-日前溢价
            if f'P_{zone}_DA' in df.columns:
                features[f'spread_rt_da_{zone}'] = df[f'P_{zone}'] - df[f'P_{zone}_DA']

            # 3. 价格斜率 (5分钟)
            # 假设数据是5分钟间隔
            features[f'price_ramp_5m_{zone}'] = df[f'P_{zone}'].diff(1) / 5  # $/MWh/min

            # 4. 价格斜率 (15分钟)
            features[f'price_ramp_15m_{zone}'] = df[f'P_{zone}'].diff(3) / 15  # 假设3个5分钟步长

            # 5. 价格加速度
            features[f'price_accel_{zone}'] = features[f'price_ramp_5m_{zone}'].diff(1)

        # 6. 跨区域价差 (特别关注 CPS-Houston)
        if 'CPS' in zones and 'Houston' in zones:
            features['spread_CPS_Houston'] = df['P_CPS'] - df['P_Houston']
            features['spread_CPS_Houston_ramp'] = features['spread_CPS_Houston'].diff(1)

        return features


class SupplyDemandFeatures:
    """供需平衡特征计算

    目标：捕捉系统/区域紧张状态
    """

    @staticmethod
    def calculate(df: pd.DataFrame, lookback_days: int = 30) -> pd.DataFrame:
        """计算供需平衡特征

        Args:
            df: 包含系统数据的 DataFrame
                必需列: Load, Wind, Solar, Gas, Coal, ESR
            lookback_days: 滚动窗口天数（用于计算异常值）

        Returns:
            包含供需平衡特征的 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. 净负荷
        features['net_load'] = df['Load'] - df['Wind'] - df['Solar']

        # 2. 净负荷爬坡速度 (MW/5min)
        features['net_load_ramp_5m'] = features['net_load'].diff(1)
        features['net_load_ramp_15m'] = features['net_load'].diff(3)

        # 3. 净负荷加速度
        features['net_load_accel'] = features['net_load_ramp_5m'].diff(1)

        # 4. 风电特征
        # 滚动均值和标准差（用于计算异常值）
        window_size = lookback_days * 24 * 12  # 假设5分钟数据
        wind_rolling_mean = df['Wind'].rolling(window=window_size, min_periods=1).mean()
        wind_rolling_std = df['Wind'].rolling(window=window_size, min_periods=1).std()

        # 风电异常（标准化偏差）
        features['wind_anomaly'] = (df['Wind'] - wind_rolling_mean) / (wind_rolling_std + 1e-6)

        # 风电爬坡
        features['wind_ramp'] = df['Wind'].diff(1)
        features['wind_ramp_15m'] = df['Wind'].diff(3)

        # 5. 气电饱和度
        # 使用7天滚动95分位数作为参考容量
        gas_window = 7 * 24 * 12
        gas_p95 = df['Gas'].rolling(window=gas_window, min_periods=1).quantile(0.95)
        features['gas_saturation'] = df['Gas'] / (gas_p95 + 1e-6)

        # 6. 煤电压力（夜间上行标志）
        # 煤电变化
        coal_diff = df['Coal'].diff(1)
        # 夜间时段 (0-5点)
        is_night = df.index.hour.isin(range(0, 6))
        # 夜间煤电上行
        features['coal_stress'] = ((coal_diff > 0) & is_night).astype(int)
        features['coal_ramp'] = coal_diff

        # 7. 储能系统净出力
        features['esr_net_output'] = df['ESR']
        features['esr_is_charging'] = (df['ESR'] < 0).astype(int)
        features['esr_is_discharging'] = (df['ESR'] > 0).astype(int)

        # 8. 光伏爬坡
        features['solar_ramp'] = df['Solar'].diff(1)
        features['solar_ramp_15m'] = df['Solar'].diff(3)

        return features


class WeatherFeatures:
    """天气驱动特征计算 (Zone-level)

    目标：捕捉需求侧冲击信号
    """

    @staticmethod
    def calculate(df: pd.DataFrame, zones: List[str] = ['CPS', 'West', 'Houston'],
                  lookback_days: int = 30) -> pd.DataFrame:
        """计算天气驱动特征

        Args:
            df: 包含天气数据的 DataFrame
                必需列: T_{zone}, WindSpeed_{zone}, WindDir_{zone} for each zone
            zones: 区域列表
            lookback_days: 滚动窗口天数

        Returns:
            包含天气特征的 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        for zone in zones:
            temp_col = f'T_{zone}'
            wind_speed_col = f'WindSpeed_{zone}'
            wind_dir_col = f'WindDir_{zone}'

            if temp_col not in df.columns:
                continue

            # 1. 温度异常（相对于历史同小时均值）
            # 按小时分组计算滚动均值
            hourly_temp_mean = df.groupby(df.index.hour)[temp_col].transform(
                lambda x: x.rolling(window=lookback_days, min_periods=1).mean()
            )
            features[f'T_anomaly_{zone}'] = df[temp_col] - hourly_temp_mean

            # 2. 降温速度 (°F/hour)
            # 假设数据是15分钟间隔，12个步长 = 1小时
            features[f'T_ramp_{zone}'] = df[temp_col].diff(12) / 1  # °F/hr

            # 3. 风寒指数 (Wind Chill)
            if wind_speed_col in df.columns:
                T = df[temp_col]
                v = df[wind_speed_col]
                # Wind Chill formula (适用于 T ≤ 50°F 和 v ≥ 3 mph)
                features[f'WindChill_{zone}'] = (
                    35.74 + 0.6215 * T - 35.75 * (v ** 0.16) + 0.4275 * T * (v ** 0.16)
                )
                # 对于不适用的情况，使用实际温度
                mask = (T > 50) | (v < 3)
                features.loc[mask, f'WindChill_{zone}'] = T[mask]

            # 4. 冷锋标志
            if wind_dir_col in df.columns:
                # 北风：风向在 315-45 度之间
                wind_to_north = (df[wind_dir_col] > 315) | (df[wind_dir_col] < 45)
                # 快速降温 + 北风 = 冷锋
                features[f'ColdFront_{zone}'] = (
                    (features[f'T_ramp_{zone}'] < -5) & wind_to_north
                ).astype(int)
            else:
                features[f'ColdFront_{zone}'] = 0

        return features


class TemporalFeatures:
    """时间特征计算

    目标：捕捉日内模式与光伏修复窗口
    """

    @staticmethod
    def calculate(df: pd.DataFrame, latitude: float = 29.76) -> pd.DataFrame:
        """计算时间特征

        Args:
            df: DataFrame with datetime index
            latitude: 纬度（用于计算日出时间，San Antonio ~29.76°N）

        Returns:
            包含时间特征的 DataFrame
        """
        features = pd.DataFrame(index=df.index)

        # 1. 基础时间特征
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month

        # 2. 晚高峰标志
        features['is_evening_peak'] = df.index.hour.isin(range(17, 23)).astype(int)

        # 3. 日出时间估算（简化版）
        # 这里使用简化公式，实际应用中可以使用 ephem 或 astral 库
        day_of_year = df.index.dayofyear
        # 简化的日出时间计算（小时，本地时间）
        # 这只是近似值，实际项目中应使用准确的天文算法
        declination = 23.45 * np.sin(np.radians((360/365) * (day_of_year - 81)))
        sunrise_hour = 12 - (1/15) * np.degrees(
            np.arccos(-np.tan(np.radians(latitude)) * np.tan(np.radians(declination)))
        )

        # 距离日出的时间（分钟）
        current_hour = df.index.hour + df.index.minute / 60
        minutes_to_sunrise = (sunrise_hour - current_hour) * 60
        # 处理跨日情况
        minutes_to_sunrise = np.where(minutes_to_sunrise < -720, minutes_to_sunrise + 1440, minutes_to_sunrise)
        features['minutes_to_sunrise'] = minutes_to_sunrise

        # 4. 日出前后标志
        features['is_pre_sunrise'] = (minutes_to_sunrise > 0) & (minutes_to_sunrise < 120)
        features['is_post_sunrise'] = (minutes_to_sunrise < 0) & (minutes_to_sunrise > -120)

        # 5. 光伏爬坡预期（如果有Solar数据）
        if 'Solar' in df.columns:
            features['solar_ramp_expected'] = df['Solar'].diff(3)  # 15分钟变化

        return features


class FeatureEngineer:
    """特征工程主类

    整合所有特征计算模块
    """

    def __init__(self, zones: List[str] = ['CPS', 'West', 'Houston'],
                 lookback_days: int = 30):
        """初始化

        Args:
            zones: 需要计算特征的区域列表
            lookback_days: 滚动窗口天数
        """
        self.zones = zones
        self.lookback_days = lookback_days

    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有特征

        Args:
            df: 原始数据 DataFrame

        Returns:
            包含所有特征的 DataFrame
        """
        print("计算价格结构特征...")
        price_features = PriceStructureFeatures.calculate(df, self.zones)

        print("计算供需平衡特征...")
        supply_demand_features = SupplyDemandFeatures.calculate(df, self.lookback_days)

        print("计算天气驱动特征...")
        weather_features = WeatherFeatures.calculate(df, self.zones, self.lookback_days)

        print("计算时间特征...")
        temporal_features = TemporalFeatures.calculate(df)

        # 合并所有特征
        all_features = pd.concat([
            df,  # 保留原始数据
            price_features,
            supply_demand_features,
            weather_features,
            temporal_features
        ], axis=1)

        print(f"特征计算完成！总特征数: {len(all_features.columns)}")

        return all_features

    def get_feature_names(self, feature_type: Optional[str] = None) -> List[str]:
        """获取特征名称列表

        Args:
            feature_type: 特征类型 ('price', 'supply_demand', 'weather', 'temporal', None=all)

        Returns:
            特征名称列表
        """
        if feature_type == 'price':
            features = []
            for zone in self.zones:
                features.extend([
                    f'spread_{zone}_hub',
                    f'spread_rt_da_{zone}',
                    f'price_ramp_5m_{zone}',
                    f'price_ramp_15m_{zone}',
                    f'price_accel_{zone}',
                ])
            features.append('spread_CPS_Houston')
            features.append('spread_CPS_Houston_ramp')
            return features

        elif feature_type == 'supply_demand':
            return [
                'net_load', 'net_load_ramp_5m', 'net_load_ramp_15m', 'net_load_accel',
                'wind_anomaly', 'wind_ramp', 'wind_ramp_15m',
                'gas_saturation',
                'coal_stress', 'coal_ramp',
                'esr_net_output', 'esr_is_charging', 'esr_is_discharging',
                'solar_ramp', 'solar_ramp_15m',
            ]

        elif feature_type == 'weather':
            features = []
            for zone in self.zones:
                features.extend([
                    f'T_anomaly_{zone}',
                    f'T_ramp_{zone}',
                    f'WindChill_{zone}',
                    f'ColdFront_{zone}',
                ])
            return features

        elif feature_type == 'temporal':
            return [
                'hour', 'day_of_week', 'month',
                'is_evening_peak',
                'minutes_to_sunrise', 'is_pre_sunrise', 'is_post_sunrise',
                'solar_ramp_expected',
            ]

        else:
            # 返回所有特征
            return (
                self.get_feature_names('price') +
                self.get_feature_names('supply_demand') +
                self.get_feature_names('weather') +
                self.get_feature_names('temporal')
            )


if __name__ == '__main__':
    # 测试代码
    print("特征工程模块加载成功！")
    print("\n支持的特征类型:")
    print("1. 价格结构特征 (Price Structure)")
    print("2. 供需平衡特征 (Supply-Demand Balance)")
    print("3. 天气驱动特征 (Weather-Driven)")
    print("4. 时间特征 (Temporal)")
