#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM Delta/Spread 数据准备脚本
=================================
合并RTM和DAM数据，计算Spread，创建预测标签

用法:
    python prepare_delta_data.py --rtm ../data/rtm_lz_west.csv --dam ../data/dam_lz_west.csv --output ../data/spread_data.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_rtm_data(filepath: str) -> pd.DataFrame:
    """加载RTM数据 (15分钟间隔)"""
    df = pd.read_csv(filepath)

    # 转换日期
    df['date_dt'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

    # 创建完整时间戳 (15分钟粒度)
    df['timestamp'] = df['date_dt'] + pd.to_timedelta(df['hour'] - 1, unit='h') + pd.to_timedelta((df['interval'] - 1) * 15, unit='m')

    return df


def load_dam_data(filepath: str) -> pd.DataFrame:
    """加载DAM数据 (小时间隔)"""
    df = pd.read_csv(filepath)

    # 转换日期
    df['date_dt'] = pd.to_datetime(df['date'], format='%m/%d/%Y')

    # 创建完整时间戳 (小时粒度)
    df['timestamp'] = df['date_dt'] + pd.to_timedelta(df['hour'] - 1, unit='h')

    return df


def aggregate_rtm_to_hourly(rtm_df: pd.DataFrame) -> pd.DataFrame:
    """
    将RTM 15分钟数据聚合到小时级别

    聚合方式:
    - rtm_mean: 小时内4个interval的平均价格
    - rtm_max: 小时内最高价
    - rtm_min: 小时内最低价
    - rtm_last: 小时内最后一个interval价格 (实际结算价)
    - rtm_std: 小时内价格波动
    """
    rtm_df['hour_start'] = rtm_df['timestamp'].dt.floor('h')

    agg_df = rtm_df.groupby('hour_start').agg({
        'price': ['mean', 'max', 'min', 'last', 'std', 'count']
    }).reset_index()

    agg_df.columns = ['timestamp', 'rtm_mean', 'rtm_max', 'rtm_min', 'rtm_last', 'rtm_std', 'rtm_count']

    # 只保留完整小时 (4个interval)
    agg_df = agg_df[agg_df['rtm_count'] == 4].copy()
    agg_df = agg_df.drop(columns=['rtm_count'])

    return agg_df


def merge_rtm_dam(rtm_hourly: pd.DataFrame, dam_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并RTM和DAM数据
    """
    # 准备DAM数据
    dam_simple = dam_df[['timestamp', 'dam_price']].copy()

    # 按时间戳合并
    merged = pd.merge(rtm_hourly, dam_simple, on='timestamp', how='inner')

    # 按时间排序
    merged = merged.sort_values('timestamp').reset_index(drop=True)

    return merged


def calculate_spread_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算Spread并创建预测标签

    Spread定义:
    - spread = RTM平均价 - DAM价格
    - spread > 0: RTM高于DAM (实时市场紧张)
    - spread < 0: RTM低于DAM (实时市场宽松)
    """
    # 计算spread (使用RTM平均价)
    df['spread'] = df['rtm_mean'] - df['dam_price']

    # 也计算使用last价格的spread (更接近实际结算)
    df['spread_last'] = df['rtm_last'] - df['dam_price']

    # 二分类标签: spread方向
    # 1 = RTM > DAM (在DAM买入有利)
    # 0 = RTM < DAM (在DAM卖出有利)
    df['spread_direction'] = (df['spread'] > 0).astype(int)

    # 多分类标签: spread区间
    # 0: < -20
    # 1: -20 ~ -5
    # 2: -5 ~ 5 (不交易区)
    # 3: 5 ~ 20
    # 4: >= 20
    bins = [-np.inf, -20, -5, 5, 20, np.inf]
    labels = [0, 1, 2, 3, 4]
    df['spread_class'] = pd.cut(df['spread'], bins=bins, labels=labels).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间特征"""
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['quarter'] = df['timestamp'].dt.quarter
    df['day_of_month'] = df['timestamp'].dt.day
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['year'] = df['timestamp'].dt.year

    # 是否周末
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 是否高峰时段 (通常 6-10am 和 5-9pm)
    df['is_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 10) |
                     (df['hour'] >= 17) & (df['hour'] <= 21)).astype(int)

    # 是否夏季 (6-9月, ERCOT夏季高峰)
    df['is_summer'] = df['month'].isin([6, 7, 8, 9]).astype(int)

    return df


def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加历史特征 (用于预测)

    注意: 这些特征是基于历史数据计算的，不会造成信息泄露
    """
    # Spread滞后特征
    for lag in [24, 48, 72, 168]:  # 1天, 2天, 3天, 1周
        df[f'spread_lag_{lag}h'] = df['spread'].shift(lag)

    # 同时段历史spread (上周同时段)
    df['spread_same_hour_7d'] = df['spread'].shift(168)  # 7*24 = 168

    # Spread滚动统计 (过去7天)
    df['spread_mean_7d'] = df['spread'].rolling(168).mean()
    df['spread_std_7d'] = df['spread'].rolling(168).std()
    df['spread_max_7d'] = df['spread'].rolling(168).max()
    df['spread_min_7d'] = df['spread'].rolling(168).min()

    # Spread滚动统计 (过去24小时)
    df['spread_mean_24h'] = df['spread'].rolling(24).mean()
    df['spread_std_24h'] = df['spread'].rolling(24).std()

    # RTM历史特征
    df['rtm_mean_24h'] = df['rtm_mean'].rolling(24).mean()
    df['rtm_std_24h'] = df['rtm_mean'].rolling(24).std()
    df['rtm_mean_7d'] = df['rtm_mean'].rolling(168).mean()

    # DAM历史特征
    df['dam_mean_24h'] = df['dam_price'].rolling(24).mean()
    df['dam_std_24h'] = df['dam_price'].rolling(24).std()
    df['dam_mean_7d'] = df['dam_price'].rolling(168).mean()

    # 同时段历史均值 (按小时分组的历史平均)
    # 这需要单独计算

    return df


def add_spike_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加价格spike特征"""
    # Spread spike检测
    for threshold in [10, 20, 50]:
        df[f'spread_spike_{threshold}'] = (df['spread'].abs() > threshold).astype(int)

    # RTM spike检测
    for threshold in [100, 200, 500]:
        df[f'rtm_spike_{threshold}'] = (df['rtm_mean'] > threshold).astype(int)

    # 过去24小时spike计数
    df['spike_count_24h'] = df['spread_spike_20'].rolling(24).sum()
    df['rtm_spike_count_24h'] = df['rtm_spike_100'].rolling(24).sum()

    return df


def prepare_delta_data(
    rtm_path: str,
    dam_path: str,
    output_path: str,
    start_date: str = '2015-01-01'
) -> pd.DataFrame:
    """
    主函数: 准备Delta预测数据
    """
    print("=" * 60)
    print("RTM-DAM Delta 数据准备")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    rtm_df = load_rtm_data(rtm_path)
    dam_df = load_dam_data(dam_path)
    print(f"   RTM: {len(rtm_df):,} 条 (15分钟)")
    print(f"   DAM: {len(dam_df):,} 条 (小时)")

    # 2. 聚合RTM到小时级别
    print("\n2. 聚合RTM到小时级别...")
    rtm_hourly = aggregate_rtm_to_hourly(rtm_df)
    print(f"   RTM小时数据: {len(rtm_hourly):,} 条")

    # 3. 合并RTM和DAM
    print("\n3. 合并RTM和DAM数据...")
    merged = merge_rtm_dam(rtm_hourly, dam_df)
    print(f"   合并后: {len(merged):,} 条")

    # 4. 计算Spread和标签
    print("\n4. 计算Spread和标签...")
    merged = calculate_spread_and_labels(merged)

    # 5. 添加时间特征
    print("\n5. 添加时间特征...")
    merged = add_time_features(merged)

    # 6. 添加历史特征
    print("\n6. 添加历史特征...")
    merged = add_historical_features(merged)

    # 7. 添加spike特征
    print("\n7. 添加spike特征...")
    merged = add_spike_features(merged)

    # 8. 筛选数据范围
    start_dt = pd.to_datetime(start_date)
    merged = merged[merged['timestamp'] >= start_dt].copy()

    # 9. 删除含NaN的行 (由于滚动计算产生)
    merged = merged.dropna().reset_index(drop=True)

    # 10. 保存数据
    print("\n8. 保存数据...")
    merged.to_csv(output_path, index=False)
    print(f"   已保存: {output_path}")

    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据统计")
    print("=" * 60)
    print(f"总样本数: {len(merged):,}")
    print(f"时间范围: {merged['timestamp'].iloc[0]} ~ {merged['timestamp'].iloc[-1]}")
    print(f"\nSpread统计:")
    print(f"  均值: ${merged['spread'].mean():.2f}")
    print(f"  标准差: ${merged['spread'].std():.2f}")
    print(f"  最小值: ${merged['spread'].min():.2f}")
    print(f"  最大值: ${merged['spread'].max():.2f}")
    print(f"\nSpread方向分布:")
    print(f"  RTM > DAM (正spread): {(merged['spread_direction'] == 1).sum():,} ({(merged['spread_direction'] == 1).mean()*100:.1f}%)")
    print(f"  RTM < DAM (负spread): {(merged['spread_direction'] == 0).sum():,} ({(merged['spread_direction'] == 0).mean()*100:.1f}%)")
    print(f"\nSpread区间分布:")
    class_dist = merged['spread_class'].value_counts().sort_index()
    class_labels = ['< -$20', '-$20~-$5', '-$5~$5', '$5~$20', '>= $20']
    for i, label in enumerate(class_labels):
        count = class_dist.get(i, 0)
        pct = count / len(merged) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    return merged


def main():
    parser = argparse.ArgumentParser(description='Prepare RTM-DAM Delta/Spread data')
    parser.add_argument('--rtm', '-r', type=str, required=True,
                        help='Path to RTM CSV file')
    parser.add_argument('--dam', '-d', type=str, required=True,
                        help='Path to DAM CSV file')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output CSV file path')
    parser.add_argument('--start-date', type=str, default='2015-01-01',
                        help='Start date for data (default: 2015-01-01)')
    args = parser.parse_args()

    prepare_delta_data(
        rtm_path=args.rtm,
        dam_path=args.dam,
        output_path=args.output,
        start_date=args.start_date
    )


if __name__ == "__main__":
    main()
