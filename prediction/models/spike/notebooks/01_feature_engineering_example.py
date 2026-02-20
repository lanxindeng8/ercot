"""
特征工程示例脚本

演示如何使用特征计算和标签生成模块
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.feature_engineering import FeatureEngineer
from src.utils.labels import LabelGenerator


def create_sample_data(n_days=7):
    """创建示例数据用于测试

    Args:
        n_days: 天数

    Returns:
        示例 DataFrame
    """
    # 生成时间索引 (5分钟间隔)
    start_time = datetime(2025, 12, 10, 0, 0)
    periods = n_days * 24 * 12  # 5分钟间隔
    timestamps = pd.date_range(start=start_time, periods=periods, freq='5min')

    # 创建示例数据
    np.random.seed(42)

    # 基础价格（模拟日内模式）
    hour = timestamps.hour + timestamps.minute / 60
    base_price = 50 + 30 * np.sin((hour - 6) * np.pi / 12)  # 日内波动

    # 添加随机波动
    noise = np.random.normal(0, 10, len(timestamps))

    # 模拟 spike 事件（在某些时段价格飙升）
    spike_mask = (timestamps.day == 14) & (timestamps.hour >= 20) & (timestamps.hour <= 22)
    spike_boost = np.where(spike_mask, 500, 0)

    df = pd.DataFrame({
        # 价格数据
        'P_CPS': base_price + noise + spike_boost + np.random.normal(20, 5, len(timestamps)),
        'P_West': base_price + noise + spike_boost * 0.7 + np.random.normal(15, 5, len(timestamps)),
        'P_Houston': base_price + noise + spike_boost * 0.3 + np.random.normal(0, 5, len(timestamps)),
        'P_Hub': base_price + noise,

        # 日前价格
        'P_CPS_DA': base_price + np.random.normal(0, 5, len(timestamps)),
        'P_West_DA': base_price + np.random.normal(0, 5, len(timestamps)),
        'P_Houston_DA': base_price + np.random.normal(0, 5, len(timestamps)),

        # 系统数据
        'Load': 40000 + 10000 * np.sin((hour - 12) * np.pi / 12) + np.random.normal(0, 500, len(timestamps)),
        'Wind': 8000 + 3000 * np.sin(hour * np.pi / 24) + np.random.normal(0, 500, len(timestamps)),
        'Solar': np.maximum(0, 6000 * np.sin((hour - 6) * np.pi / 12)) + np.random.normal(0, 300, len(timestamps)),
        'Gas': 25000 + 5000 * np.sin((hour - 14) * np.pi / 12) + np.random.normal(0, 300, len(timestamps)),
        'Coal': 8000 + np.random.normal(0, 200, len(timestamps)),
        'ESR': np.random.normal(0, 1000, len(timestamps)),  # 储能净出力

        # 天气数据
        'T_CPS': 65 + 15 * np.sin((hour - 15) * np.pi / 12) + np.random.normal(0, 2, len(timestamps)),
        'T_West': 63 + 16 * np.sin((hour - 15) * np.pi / 12) + np.random.normal(0, 2, len(timestamps)),
        'T_Houston': 68 + 12 * np.sin((hour - 15) * np.pi / 12) + np.random.normal(0, 2, len(timestamps)),

        'WindSpeed_CPS': 10 + 5 * np.random.random(len(timestamps)),
        'WindSpeed_West': 12 + 5 * np.random.random(len(timestamps)),
        'WindSpeed_Houston': 8 + 4 * np.random.random(len(timestamps)),

        'WindDir_CPS': np.random.uniform(0, 360, len(timestamps)),
        'WindDir_West': np.random.uniform(0, 360, len(timestamps)),
        'WindDir_Houston': np.random.uniform(0, 360, len(timestamps)),
    }, index=timestamps)

    # 模拟冷锋事件（12/14）
    cold_front_mask = (timestamps.day == 14) & (timestamps.hour >= 18)
    df.loc[cold_front_mask, 'T_CPS'] -= 10
    df.loc[cold_front_mask, 'WindDir_CPS'] = 0  # 北风

    return df


def main():
    """主函数"""
    print("=" * 80)
    print("ERCOT RTM LMP Spike 预测 - 特征工程示例")
    print("=" * 80)

    # 1. 创建示例数据
    print("\n步骤 1: 创建示例数据...")
    df = create_sample_data(n_days=7)
    print(f"数据维度: {df.shape}")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"\n原始数据列:\n{df.columns.tolist()}")

    # 2. 计算特征
    print("\n" + "=" * 80)
    print("步骤 2: 计算特征...")
    print("=" * 80)

    feature_engineer = FeatureEngineer(
        zones=['CPS', 'West', 'Houston'],
        lookback_days=30
    )

    df_with_features = feature_engineer.calculate_all_features(df)
    print(f"\n添加特征后数据维度: {df_with_features.shape}")

    # 显示特征名称
    price_features = feature_engineer.get_feature_names('price')
    supply_demand_features = feature_engineer.get_feature_names('supply_demand')
    weather_features = feature_engineer.get_feature_names('weather')
    temporal_features = feature_engineer.get_feature_names('temporal')

    print(f"\n价格结构特征 ({len(price_features)} 个):")
    print(f"  {price_features[:5]}... (显示前5个)")

    print(f"\n供需平衡特征 ({len(supply_demand_features)} 个):")
    print(f"  {supply_demand_features[:5]}... (显示前5个)")

    print(f"\n天气驱动特征 ({len(weather_features)} 个):")
    print(f"  {weather_features[:5]}... (显示前5个)")

    print(f"\n时间特征 ({len(temporal_features)} 个):")
    print(f"  {temporal_features}")

    # 3. 生成标签
    print("\n" + "=" * 80)
    print("步骤 3: 生成标签...")
    print("=" * 80)

    label_generator = LabelGenerator(
        zones=['CPS', 'West', 'Houston'],
        P_hi=400,      # Spike 价格阈值
        S_hi=50,       # Spike 价差阈值
        S_cross_hi=80, # 跨区域价差阈值
        P_mid=150,     # Tight 价格阈值
        S_mid=20,      # Tight 价差阈值
        m=3,           # 持续时间阈值 (3个5分钟 = 15分钟)
        H=60,          # Lead Spike 预警窗口 (60分钟)
        dt=5           # 数据时间间隔
    )

    labels = label_generator.generate_all_labels(df_with_features)
    print(f"\n标签维度: {labels.shape}")
    print(f"标签列: {labels.columns.tolist()}")

    # 4. 合并数据
    print("\n" + "=" * 80)
    print("步骤 4: 合并特征和标签...")
    print("=" * 80)

    final_df = pd.concat([df_with_features, labels], axis=1)
    print(f"\n最终数据维度: {final_df.shape}")
    print(f"总列数: {len(final_df.columns)}")

    # 5. 识别 Spike 事件
    print("\n" + "=" * 80)
    print("步骤 5: 识别独立 Spike 事件...")
    print("=" * 80)

    for zone in ['CPS', 'West', 'Houston']:
        spike_col = f'SpikeEvent_{zone}'
        if spike_col in labels.columns:
            events = label_generator.identify_spike_events(labels[spike_col])
            print(f"\n{zone} 区域发现 {len(events)} 个独立 Spike 事件:")
            for i, event in enumerate(events, 1):
                print(f"  事件 {i}:")
                print(f"    开始: {event['start']}")
                print(f"    结束: {event['end']}")
                print(f"    持续: {event['duration']} 个时间步 ({event['duration'] * 5} 分钟)")

    # 6. 数据质量检查
    print("\n" + "=" * 80)
    print("步骤 6: 数据质量检查...")
    print("=" * 80)

    # 检查缺失值
    missing = final_df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n发现缺失值:")
        print(missing[missing > 0])
    else:
        print("\n✓ 无缺失值")

    # 检查无穷值
    inf_count = np.isinf(final_df.select_dtypes(include=[np.number])).sum()
    if inf_count.sum() > 0:
        print(f"\n发现无穷值:")
        print(inf_count[inf_count > 0])
    else:
        print("✓ 无无穷值")

    # 7. 保存示例数据
    print("\n" + "=" * 80)
    print("步骤 7: 保存示例数据...")
    print("=" * 80)

    output_path = '../data/processed/sample_features_labels.csv'
    final_df.to_csv(output_path)
    print(f"数据已保存至: {output_path}")

    # 8. 显示关键时刻的数据（Spike 期间）
    print("\n" + "=" * 80)
    print("步骤 8: 查看 Spike 期间的数据...")
    print("=" * 80)

    spike_mask = labels['SpikeEvent_CPS'] == 1
    if spike_mask.any():
        print(f"\nSpike 期间数据样本 (前5行):")
        spike_data = final_df[spike_mask].head()
        display_cols = ['P_CPS', 'P_Hub', 'spread_CPS_hub', 'net_load',
                       'wind_anomaly', 'gas_saturation', 'T_anomaly_CPS',
                       'Regime_CPS']
        print(spike_data[display_cols])

    print("\n" + "=" * 80)
    print("特征工程示例完成！")
    print("=" * 80)
    print("\n下一步:")
    print("1. 使用真实的 ERCOT 数据替换示例数据")
    print("2. 进行特征分析和可视化")
    print("3. 训练预测模型")
    print("=" * 80)


if __name__ == '__main__':
    main()
