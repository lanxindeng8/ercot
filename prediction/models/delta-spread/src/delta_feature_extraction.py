#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM Delta 特征工程脚本 (40小时预测)
========================================
为日前套利策略创建特征

预测场景:
    - 预测时间: D-1日 上午10:00
    - 目标时间: D日 00:00 ~ 23:00 (24个小时)
    - 预测提前量: 14~38小时

特征设计原则:
    - 只使用预测时刻之前的历史数据
    - DAM价格在预测时刻已知 (DAM在D-1日清算)
    - 目标小时的时间特征是已知的

用法:
    python delta_feature_extraction.py --input ../data/spread_data.csv --output ../data/train_features.csv
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def create_training_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建训练样本

    每个样本代表:
    - 在某一天D-1的10:00 (或最新可用时刻)
    - 预测D日某小时的spread

    特征:
    - 历史spread/RTM/DAM统计 (截至D-1日的最新数据)
    - 目标小时的DAM价格 (已知)
    - 目标小时的时间特征
    """
    print("创建训练样本...")

    # 按日期分组
    df['date'] = df['timestamp'].dt.date

    # 获取唯一日期
    dates = sorted(df['date'].unique())

    samples = []

    for i in tqdm(range(1, len(dates)), desc="处理每日数据"):
        # D-1日 (历史数据日)
        hist_date = dates[i-1]
        # D日 (目标预测日)
        target_date = dates[i]

        # 获取D-1日数据 (用于计算历史特征)
        hist_data = df[df['date'] <= hist_date].copy()
        if len(hist_data) < 168:  # 至少需要7天历史
            continue

        # 获取D日数据 (用于提取目标)
        target_data = df[df['date'] == target_date].copy()
        if len(target_data) == 0:
            continue

        # 历史统计特征 (基于D-1日及之前所有数据)
        recent_168h = hist_data.tail(168)  # 最近7天
        recent_24h = hist_data.tail(24)    # 最近1天

        hist_features = {
            # 近期spread统计
            'spread_mean_7d': recent_168h['spread'].mean(),
            'spread_std_7d': recent_168h['spread'].std(),
            'spread_max_7d': recent_168h['spread'].max(),
            'spread_min_7d': recent_168h['spread'].min(),
            'spread_median_7d': recent_168h['spread'].median(),

            'spread_mean_24h': recent_24h['spread'].mean(),
            'spread_std_24h': recent_24h['spread'].std(),

            # 近期RTM统计
            'rtm_mean_7d': recent_168h['rtm_mean'].mean(),
            'rtm_std_7d': recent_168h['rtm_mean'].std(),
            'rtm_max_7d': recent_168h['rtm_max'].max(),

            'rtm_mean_24h': recent_24h['rtm_mean'].mean(),
            'rtm_volatility_24h': recent_24h['rtm_std'].mean(),

            # 近期DAM统计
            'dam_mean_7d': recent_168h['dam_price'].mean(),
            'dam_std_7d': recent_168h['dam_price'].std(),
            'dam_mean_24h': recent_24h['dam_price'].mean(),

            # Spread方向统计
            'spread_positive_ratio_7d': (recent_168h['spread'] > 0).mean(),
            'spread_positive_ratio_24h': (recent_24h['spread'] > 0).mean(),

            # Spike计数
            'spike_count_7d': (recent_168h['spread'].abs() > 20).sum(),
            'rtm_spike_count_7d': (recent_168h['rtm_mean'] > 100).sum(),

            # 趋势特征
            'spread_trend_7d': recent_168h['spread'].iloc[-24:].mean() - recent_168h['spread'].iloc[:24].mean(),
        }

        # 按小时分组的历史均值 (用于同时段预测)
        hourly_hist = hist_data.groupby('hour').agg({
            'spread': ['mean', 'std'],
            'rtm_mean': 'mean',
            'dam_price': 'mean'
        })
        hourly_hist.columns = ['spread_by_hour_mean', 'spread_by_hour_std', 'rtm_by_hour_mean', 'dam_by_hour_mean']

        # 按小时+星期几分组
        hist_data['dow'] = pd.to_datetime(hist_data['date']).dt.dayofweek
        dow_hour_hist = hist_data.groupby(['dow', 'hour'])['spread'].mean().to_dict()

        # 为D日每个小时创建样本
        target_dow = pd.to_datetime(target_date).dayofweek
        target_month = pd.to_datetime(target_date).month
        target_day_of_month = pd.to_datetime(target_date).day
        target_week = pd.to_datetime(target_date).isocalendar().week

        for _, row in target_data.iterrows():
            target_hour = row['hour']

            sample = {
                # 时间ID
                'timestamp': row['timestamp'],
                'date': target_date,

                # 目标变量
                'target_spread': row['spread'],
                'target_spread_last': row['spread_last'],
                'target_direction': row['spread_direction'],
                'target_class': row['spread_class'],

                # 目标小时DAM价格 (已知)
                'target_dam_price': row['dam_price'],

                # 目标小时时间特征
                'target_hour': target_hour,
                'target_dow': target_dow,
                'target_month': target_month,
                'target_day_of_month': target_day_of_month,
                'target_week': target_week,
                'target_is_weekend': int(target_dow >= 5),
                'target_is_peak': int((6 <= target_hour <= 10) or (17 <= target_hour <= 21)),
                'target_is_summer': int(target_month in [6, 7, 8, 9]),

                # 历史统计特征
                **hist_features,

                # 同时段历史特征
                'spread_same_hour_hist': hourly_hist.loc[target_hour, 'spread_by_hour_mean'] if target_hour in hourly_hist.index else np.nan,
                'spread_same_hour_std': hourly_hist.loc[target_hour, 'spread_by_hour_std'] if target_hour in hourly_hist.index else np.nan,
                'rtm_same_hour_hist': hourly_hist.loc[target_hour, 'rtm_by_hour_mean'] if target_hour in hourly_hist.index else np.nan,

                # 同时段+同星期几历史
                'spread_same_dow_hour': dow_hour_hist.get((target_dow, target_hour), np.nan),

                # DAM相对特征
                'dam_vs_7d_mean': row['dam_price'] - hist_features['dam_mean_7d'],
                'dam_percentile_7d': (recent_168h['dam_price'] < row['dam_price']).mean(),
            }

            # 上周同时段的实际spread (如果有)
            same_hour_last_week = hist_data[(hist_data['hour'] == target_hour) &
                                            (hist_data['dow'] == target_dow)].tail(1)
            if len(same_hour_last_week) > 0:
                sample['spread_same_dow_hour_last'] = same_hour_last_week['spread'].values[0]
            else:
                sample['spread_same_dow_hour_last'] = np.nan

            samples.append(sample)

    result_df = pd.DataFrame(samples)
    return result_df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加衍生特征"""
    # DAM价格区间
    df['dam_price_level'] = pd.cut(df['target_dam_price'],
                                    bins=[0, 20, 40, 60, 100, 200, np.inf],
                                    labels=[0, 1, 2, 3, 4, 5]).astype(float)

    # 时间周期编码
    df['hour_sin'] = np.sin(2 * np.pi * df['target_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['target_hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['target_dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['target_dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['target_month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['target_month'] / 12)

    # 预测误差基准 (naive预测: 使用历史同时段均值)
    df['naive_pred'] = df['spread_same_hour_hist']

    # 相对特征
    df['dam_vs_same_hour'] = df['target_dam_price'] - df['rtm_same_hour_hist'].fillna(df['rtm_mean_7d'])

    return df


def select_features(df: pd.DataFrame) -> tuple:
    """
    选择特征列

    Returns:
    - feature_cols: 特征列名列表
    - target_cols: 目标列名字典
    """
    # 排除非特征列
    exclude_cols = [
        'timestamp', 'date',
        'target_spread', 'target_spread_last', 'target_direction', 'target_class',
        'naive_pred'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    target_cols = {
        'regression': 'target_spread',
        'binary': 'target_direction',
        'multiclass': 'target_class'
    }

    return feature_cols, target_cols


def extract_delta_features(
    input_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    主函数: 提取Delta预测特征
    """
    print("=" * 60)
    print("RTM-DAM Delta 特征工程 (40小时预测)")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载Spread数据...")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   原始样本数: {len(df):,}")

    # 2. 创建训练样本
    print("\n2. 创建训练样本...")
    samples_df = create_training_samples(df)
    print(f"   训练样本数: {len(samples_df):,}")

    # 3. 添加衍生特征
    print("\n3. 添加衍生特征...")
    samples_df = add_derived_features(samples_df)

    # 4. 处理缺失值
    print("\n4. 处理缺失值...")
    samples_df = samples_df.dropna()
    print(f"   有效样本数: {len(samples_df):,}")

    # 5. 保存
    print("\n5. 保存特征数据...")
    samples_df.to_csv(output_path, index=False)
    print(f"   已保存: {output_path}")

    # 打印特征信息
    feature_cols, target_cols = select_features(samples_df)
    print(f"\n特征数量: {len(feature_cols)}")
    print(f"特征列表:")
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")

    # 打印目标变量统计
    print("\n目标变量统计:")
    print(f"  回归目标 (spread): 均值={samples_df['target_spread'].mean():.2f}, 标准差={samples_df['target_spread'].std():.2f}")
    print(f"  二分类目标: RTM>DAM={samples_df['target_direction'].mean()*100:.1f}%")
    print(f"  多分类目标分布:")
    for c in range(5):
        pct = (samples_df['target_class'] == c).mean() * 100
        print(f"    类别{c}: {pct:.1f}%")

    return samples_df


def main():
    parser = argparse.ArgumentParser(description='Extract features for Delta prediction')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input spread data CSV')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output feature CSV')
    args = parser.parse_args()

    extract_delta_features(
        input_path=args.input,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
