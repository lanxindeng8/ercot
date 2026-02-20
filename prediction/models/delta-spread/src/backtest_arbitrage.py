#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM 套利策略回测脚本
=========================
基于Delta预测模型进行套利策略回测

套利逻辑:
- 预测 RTM > DAM: 在DAM买入 → 在RTM卖出 (获利 = RTM - DAM = Spread)
- 预测 RTM < DAM: 在DAM卖出 → 在RTM买回 (获利 = DAM - RTM = -Spread)

策略版本:
1. 基于方向预测 (二分类模型)
2. 基于区间预测 (多分类模型, 只在高置信区间交易)
3. 基于spread预测值 (回归模型, 设定阈值)

用法:
    python backtest_arbitrage.py --predictions ../models/predictions.csv --output ../results/backtest_results.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_predictions(filepath: str) -> pd.DataFrame:
    """加载模型预测结果"""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    return df


def strategy_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    策略1: 基于二分类模型

    - 预测 direction=1 (RTM>DAM): Long spread (买DAM卖RTM)
    - 预测 direction=0 (RTM<DAM): Short spread (卖DAM买RTM)
    """
    df = df.copy()

    # 交易信号: 1=Long spread, -1=Short spread
    df['signal'] = df['pred_direction'].apply(lambda x: 1 if x == 1 else -1)

    # 每笔交易收益 = signal * actual_spread
    # Long spread收益 = actual_spread (如果RTM确实高于DAM则赚钱)
    # Short spread收益 = -actual_spread (如果RTM确实低于DAM则赚钱)
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def strategy_binary_threshold(df: pd.DataFrame, prob_threshold: float = 0.6) -> pd.DataFrame:
    """
    策略1b: 基于二分类模型 (带概率阈值)

    只在预测概率超过阈值时交易
    """
    df = df.copy()

    # 交易信号
    df['signal'] = 0
    df.loc[df['pred_prob'] >= prob_threshold, 'signal'] = 1  # Long spread
    df.loc[df['pred_prob'] <= (1 - prob_threshold), 'signal'] = -1  # Short spread

    # 收益
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def strategy_multiclass(df: pd.DataFrame, trade_classes: list = [0, 4]) -> pd.DataFrame:
    """
    策略2: 基于多分类模型

    只在极端区间交易:
    - 类别0 (< -$20): Short spread
    - 类别4 (>= $20): Long spread
    - 其他类别不交易
    """
    df = df.copy()

    # 交易信号
    df['signal'] = 0
    if 0 in trade_classes:
        df.loc[df['pred_class'] == 0, 'signal'] = -1  # Short spread (预测RTM大幅低于DAM)
    if 4 in trade_classes:
        df.loc[df['pred_class'] == 4, 'signal'] = 1   # Long spread (预测RTM大幅高于DAM)

    # 也可以交易轻度信号
    if 1 in trade_classes:
        df.loc[df['pred_class'] == 1, 'signal'] = -1  # Short spread
    if 3 in trade_classes:
        df.loc[df['pred_class'] == 3, 'signal'] = 1   # Long spread

    # 收益
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def strategy_regression(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    策略3: 基于回归模型

    - 预测spread > threshold: Long spread
    - 预测spread < -threshold: Short spread
    - |预测spread| < threshold: 不交易
    """
    df = df.copy()

    # 交易信号
    df['signal'] = 0
    df.loc[df['pred_spread'] >= threshold, 'signal'] = 1   # Long spread
    df.loc[df['pred_spread'] <= -threshold, 'signal'] = -1  # Short spread

    # 收益
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def calculate_metrics(df: pd.DataFrame, strategy_name: str) -> dict:
    """计算策略绩效指标"""
    # 只考虑实际交易的样本
    trades = df[df['signal'] != 0].copy()

    if len(trades) == 0:
        return {
            'strategy': strategy_name,
            'total_trades': 0,
            'total_pnl': 0,
            'avg_pnl_per_trade': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }

    # 基本统计
    total_trades = len(trades)
    total_pnl = trades['pnl'].sum()
    avg_pnl = trades['pnl'].mean()

    # 胜率
    winning_trades = (trades['pnl'] > 0).sum()
    win_rate = winning_trades / total_trades

    # 盈亏比
    gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_loss = -trades.loc[trades['pnl'] < 0, 'pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # 按日聚合计算Sharpe
    daily_pnl = df.groupby('date')['pnl'].sum()
    if daily_pnl.std() > 0:
        sharpe = np.sqrt(252) * daily_pnl.mean() / daily_pnl.std()
    else:
        sharpe = 0

    # 最大回撤
    cumulative_pnl = trades['pnl'].cumsum()
    rolling_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - rolling_max
    max_drawdown = drawdown.min()

    # 按方向统计
    long_trades = trades[trades['signal'] == 1]
    short_trades = trades[trades['signal'] == -1]

    return {
        'strategy': strategy_name,
        'total_trades': total_trades,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': avg_pnl,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'long_win_rate': (long_trades['pnl'] > 0).mean() if len(long_trades) > 0 else 0,
        'short_win_rate': (short_trades['pnl'] > 0).mean() if len(short_trades) > 0 else 0
    }


def run_backtest(predictions_path: str, output_path: str = None) -> dict:
    """
    运行所有策略的回测
    """
    print("=" * 60)
    print("RTM-DAM 套利策略回测")
    print("=" * 60)

    # 加载预测
    print("\n1. 加载预测数据...")
    df = load_predictions(predictions_path)
    print(f"   样本数: {len(df):,}")
    print(f"   时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # 运行各策略
    print("\n2. 运行策略回测...")
    strategies = []

    # 策略1: 二分类 (全部交易)
    df_s1 = strategy_binary(df)
    metrics_s1 = calculate_metrics(df_s1, 'Binary (All)')
    strategies.append(metrics_s1)

    # 策略1b: 二分类 (带阈值)
    for threshold in [0.55, 0.6, 0.65, 0.7]:
        df_s1b = strategy_binary_threshold(df, threshold)
        metrics_s1b = calculate_metrics(df_s1b, f'Binary (prob>{threshold})')
        strategies.append(metrics_s1b)

    # 策略2: 多分类 (只交易极端区间)
    df_s2a = strategy_multiclass(df, trade_classes=[0, 4])
    metrics_s2a = calculate_metrics(df_s2a, 'Multiclass (classes 0,4)')
    strategies.append(metrics_s2a)

    df_s2b = strategy_multiclass(df, trade_classes=[0, 1, 3, 4])
    metrics_s2b = calculate_metrics(df_s2b, 'Multiclass (classes 0,1,3,4)')
    strategies.append(metrics_s2b)

    # 策略3: 回归 (带阈值)
    for threshold in [3, 5, 10, 15]:
        df_s3 = strategy_regression(df, threshold)
        metrics_s3 = calculate_metrics(df_s3, f'Regression (|pred|>{threshold})')
        strategies.append(metrics_s3)

    # 基准策略: 总是Short spread (因为RTM通常低于DAM)
    df_baseline = df.copy()
    df_baseline['signal'] = -1
    df_baseline['pnl'] = -df_baseline['actual_spread']
    metrics_baseline = calculate_metrics(df_baseline, 'Baseline (Always Short)')
    strategies.append(metrics_baseline)

    # 汇总结果
    results_df = pd.DataFrame(strategies)

    print("\n3. 策略绩效对比")
    print("=" * 100)
    print(f"{'策略':<35} {'交易数':>8} {'总PnL':>12} {'平均PnL':>10} {'胜率':>8} {'盈亏比':>8} {'Sharpe':>8}")
    print("-" * 100)
    for _, row in results_df.iterrows():
        print(f"{row['strategy']:<35} {row['total_trades']:>8,} {row['total_pnl']:>12,.0f} {row['avg_pnl_per_trade']:>10.2f} {row['win_rate']*100:>7.1f}% {row['profit_factor']:>8.2f} {row['sharpe_ratio']:>8.2f}")

    # 保存结果
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\n结果已保存: {output_path}")

    # 打印最佳策略
    best_strategy = results_df.loc[results_df['total_pnl'].idxmax()]
    print(f"\n最佳策略: {best_strategy['strategy']}")
    print(f"  总收益: ${best_strategy['total_pnl']:,.0f}")
    print(f"  胜率: {best_strategy['win_rate']*100:.1f}%")
    print(f"  Sharpe: {best_strategy['sharpe_ratio']:.2f}")

    return results_df


def analyze_by_period(predictions_path: str) -> None:
    """按时间段分析策略表现"""
    df = load_predictions(predictions_path)

    # 使用最佳策略 (回归模型阈值5)
    df = strategy_regression(df, threshold=5)

    # 按月份分析
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_pnl = df.groupby('month')['pnl'].sum()

    print("\n月度收益:")
    print(monthly_pnl.tail(24))

    # 按小时分析
    hourly_pnl = df.groupby(df['timestamp'].dt.hour)['pnl'].mean()
    print("\n按小时平均收益:")
    print(hourly_pnl)


def main():
    parser = argparse.ArgumentParser(description='Backtest arbitrage strategies')
    parser.add_argument('--predictions', '-p', type=str, required=True,
                        help='Path to predictions CSV')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for results')
    args = parser.parse_args()

    results = run_backtest(args.predictions, args.output)


if __name__ == "__main__":
    main()
