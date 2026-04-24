#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM Arbitrage Strategy Backtest Script
===========================================
Backtest arbitrage strategies based on Delta prediction models

Arbitrage Logic:
- Predict RTM > DAM: Buy in DAM -> Sell in RTM (profit = RTM - DAM = Spread)
- Predict RTM < DAM: Sell in DAM -> Buy back in RTM (profit = DAM - RTM = -Spread)

Strategy Versions:
1. Based on direction prediction (binary classification model)
2. Based on interval prediction (multi-class model, trade only in high-confidence intervals)
3. Based on spread prediction value (regression model, with threshold)

Usage:
    python backtest_arbitrage.py --predictions ../models/predictions.csv --output ../results/backtest_results.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def load_predictions(filepath: str) -> pd.DataFrame:
    """Load model prediction results"""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    return df


def strategy_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy 1: Based on binary classification model

    - Predicted direction=1 (RTM>DAM): Long spread (buy DAM, sell RTM)
    - Predicted direction=0 (RTM<DAM): Short spread (sell DAM, buy RTM)
    """
    df = df.copy()

    # Trading signal: 1=Long spread, -1=Short spread
    df['signal'] = df['pred_direction'].apply(lambda x: 1 if x == 1 else -1)

    # PnL per trade = signal * actual_spread
    # Long spread PnL = actual_spread (profit if RTM is indeed higher than DAM)
    # Short spread PnL = -actual_spread (profit if RTM is indeed lower than DAM)
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def strategy_binary_threshold(df: pd.DataFrame, prob_threshold: float = 0.6) -> pd.DataFrame:
    """
    Strategy 1b: Based on binary classification model (with probability threshold)

    Only trade when predicted probability exceeds the threshold
    """
    df = df.copy()

    # Trading signal
    df['signal'] = 0
    df.loc[df['pred_prob'] >= prob_threshold, 'signal'] = 1  # Long spread
    df.loc[df['pred_prob'] <= (1 - prob_threshold), 'signal'] = -1  # Short spread

    # PnL
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def strategy_multiclass(df: pd.DataFrame, trade_classes: list = [0, 4]) -> pd.DataFrame:
    """
    Strategy 2: Based on multi-class model

    Only trade in extreme intervals:
    - Class 0 (< -$20): Short spread
    - Class 4 (>= $20): Long spread
    - Other classes: no trade
    """
    df = df.copy()

    # Trading signal
    df['signal'] = 0
    if 0 in trade_classes:
        df.loc[df['pred_class'] == 0, 'signal'] = -1  # Short spread (predict RTM significantly lower than DAM)
    if 4 in trade_classes:
        df.loc[df['pred_class'] == 4, 'signal'] = 1   # Long spread (predict RTM significantly higher than DAM)

    # Can also trade moderate signals
    if 1 in trade_classes:
        df.loc[df['pred_class'] == 1, 'signal'] = -1  # Short spread
    if 3 in trade_classes:
        df.loc[df['pred_class'] == 3, 'signal'] = 1   # Long spread

    # PnL
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def strategy_regression(df: pd.DataFrame, threshold: float = 5.0) -> pd.DataFrame:
    """
    Strategy 3: Based on regression model

    - Predicted spread > threshold: Long spread
    - Predicted spread < -threshold: Short spread
    - |predicted spread| < threshold: No trade
    """
    df = df.copy()

    # Trading signal
    df['signal'] = 0
    df.loc[df['pred_spread'] >= threshold, 'signal'] = 1   # Long spread
    df.loc[df['pred_spread'] <= -threshold, 'signal'] = -1  # Short spread

    # PnL
    df['pnl'] = df['signal'] * df['actual_spread']

    return df


def calculate_metrics(df: pd.DataFrame, strategy_name: str) -> dict:
    """Calculate strategy performance metrics"""
    # Only consider samples with actual trades
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

    # Basic statistics
    total_trades = len(trades)
    total_pnl = trades['pnl'].sum()
    avg_pnl = trades['pnl'].mean()

    # Win rate
    winning_trades = (trades['pnl'] > 0).sum()
    win_rate = winning_trades / total_trades

    # Profit factor
    gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_loss = -trades.loc[trades['pnl'] < 0, 'pnl'].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Aggregate by day to calculate Sharpe
    daily_pnl = df.groupby('date')['pnl'].sum()
    if daily_pnl.std() > 0:
        sharpe = np.sqrt(252) * daily_pnl.mean() / daily_pnl.std()
    else:
        sharpe = 0

    # Maximum drawdown
    cumulative_pnl = trades['pnl'].cumsum()
    rolling_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - rolling_max
    max_drawdown = drawdown.min()

    # Statistics by direction
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
    Run backtest for all strategies
    """
    print("=" * 60)
    print("RTM-DAM Arbitrage Strategy Backtest")
    print("=" * 60)

    # Load predictions
    print("\n1. Loading prediction data...")
    df = load_predictions(predictions_path)
    print(f"   Sample count: {len(df):,}")
    print(f"   Time range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    # Run each strategy
    print("\n2. Running strategy backtests...")
    strategies = []

    # Strategy 1: Binary classification (all trades)
    df_s1 = strategy_binary(df)
    metrics_s1 = calculate_metrics(df_s1, 'Binary (All)')
    strategies.append(metrics_s1)

    # Strategy 1b: Binary classification (with threshold)
    for threshold in [0.55, 0.6, 0.65, 0.7]:
        df_s1b = strategy_binary_threshold(df, threshold)
        metrics_s1b = calculate_metrics(df_s1b, f'Binary (prob>{threshold})')
        strategies.append(metrics_s1b)

    # Strategy 2: Multi-class (trade only extreme intervals)
    df_s2a = strategy_multiclass(df, trade_classes=[0, 4])
    metrics_s2a = calculate_metrics(df_s2a, 'Multiclass (classes 0,4)')
    strategies.append(metrics_s2a)

    df_s2b = strategy_multiclass(df, trade_classes=[0, 1, 3, 4])
    metrics_s2b = calculate_metrics(df_s2b, 'Multiclass (classes 0,1,3,4)')
    strategies.append(metrics_s2b)

    # Strategy 3: Regression (with threshold)
    for threshold in [3, 5, 10, 15]:
        df_s3 = strategy_regression(df, threshold)
        metrics_s3 = calculate_metrics(df_s3, f'Regression (|pred|>{threshold})')
        strategies.append(metrics_s3)

    # Baseline strategy: Always Short spread (since RTM is typically lower than DAM)
    df_baseline = df.copy()
    df_baseline['signal'] = -1
    df_baseline['pnl'] = -df_baseline['actual_spread']
    metrics_baseline = calculate_metrics(df_baseline, 'Baseline (Always Short)')
    strategies.append(metrics_baseline)

    # Summarize results
    results_df = pd.DataFrame(strategies)

    print("\n3. Strategy Performance Comparison")
    print("=" * 100)
    print(f"{'Strategy':<35} {'Trades':>8} {'Total PnL':>12} {'Avg PnL':>10} {'Win Rate':>8} {'Profit Factor':>8} {'Sharpe':>8}")
    print("-" * 100)
    for _, row in results_df.iterrows():
        print(f"{row['strategy']:<35} {row['total_trades']:>8,} {row['total_pnl']:>12,.0f} {row['avg_pnl_per_trade']:>10.2f} {row['win_rate']*100:>7.1f}% {row['profit_factor']:>8.2f} {row['sharpe_ratio']:>8.2f}")

    # Save results
    if output_path:
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved: {output_path}")

    # Print best strategy
    best_strategy = results_df.loc[results_df['total_pnl'].idxmax()]
    print(f"\nBest strategy: {best_strategy['strategy']}")
    print(f"  Total return: ${best_strategy['total_pnl']:,.0f}")
    print(f"  Win rate: {best_strategy['win_rate']*100:.1f}%")
    print(f"  Sharpe: {best_strategy['sharpe_ratio']:.2f}")

    return results_df


def analyze_by_period(predictions_path: str) -> None:
    """Analyze strategy performance by time period"""
    df = load_predictions(predictions_path)

    # Use best strategy (regression model threshold 5)
    df = strategy_regression(df, threshold=5)

    # Analyze by month
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_pnl = df.groupby('month')['pnl'].sum()

    print("\nMonthly returns:")
    print(monthly_pnl.tail(24))

    # Analyze by hour
    hourly_pnl = df.groupby(df['timestamp'].dt.hour)['pnl'].mean()
    print("\nAverage return by hour:")
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
