#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RTM-DAM Delta 模型训练脚本
==========================
训练三种模型用于套利决策

模型:
1. 回归模型: 预测具体spread值
2. 二分类模型: 预测spread方向 (RTM > DAM?)
3. 多分类模型: 预测spread区间

用法:
    python train_delta_models.py --input ../data/train_features.csv --output-dir ../models
"""

import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from catboost import CatBoostRegressor, CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """加载特征数据"""
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['date'])
    return df


def prepare_features_targets(df: pd.DataFrame) -> dict:
    """
    准备特征和目标变量

    Returns:
        dict with X, y_reg, y_bin, y_multi
    """
    # 特征列 (排除时间和目标列)
    exclude_cols = [
        'timestamp', 'date',
        'target_spread', 'target_spread_last', 'target_direction', 'target_class',
        'naive_pred'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].copy()
    y_reg = df['target_spread'].copy()
    y_bin = df['target_direction'].copy()
    y_multi = df['target_class'].copy()

    # 记录naive预测基准
    naive_pred = df['spread_same_hour_hist'].copy()

    return {
        'X': X,
        'y_reg': y_reg,
        'y_bin': y_bin,
        'y_multi': y_multi,
        'feature_cols': feature_cols,
        'naive_pred': naive_pred,
        'timestamps': df['timestamp']
    }


def split_train_test(data: dict, test_ratio: float = 0.2) -> dict:
    """
    时序切分训练/测试集
    """
    n = len(data['X'])
    train_size = int(n * (1 - test_ratio))

    result = {
        'X_train': data['X'].iloc[:train_size],
        'X_test': data['X'].iloc[train_size:],
        'y_reg_train': data['y_reg'].iloc[:train_size],
        'y_reg_test': data['y_reg'].iloc[train_size:],
        'y_bin_train': data['y_bin'].iloc[:train_size],
        'y_bin_test': data['y_bin'].iloc[train_size:],
        'y_multi_train': data['y_multi'].iloc[:train_size],
        'y_multi_test': data['y_multi'].iloc[train_size:],
        'naive_train': data['naive_pred'].iloc[:train_size],
        'naive_test': data['naive_pred'].iloc[train_size:],
        'timestamps_test': data['timestamps'].iloc[train_size:],
        'feature_cols': data['feature_cols']
    }

    return result


def train_regression_model(X_train, y_train, X_test, y_test, naive_test) -> dict:
    """
    训练回归模型
    """
    print("\n" + "=" * 50)
    print("训练回归模型 (预测Spread值)")
    print("=" * 50)

    # 定义分类特征
    cat_features = ['target_hour', 'target_dow', 'target_month', 'target_is_weekend',
                    'target_is_peak', 'target_is_summer', 'target_day_of_month', 'target_week']
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

    model = CatBoostRegressor(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function='MAE',
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
        cat_features=cat_indices
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # 预测
    y_pred = model.predict(X_test)

    # 评估
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 与naive baseline对比
    naive_mae = mean_absolute_error(y_test, naive_test)
    improvement = (naive_mae - mae) / naive_mae * 100

    print(f"\n模型性能:")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  R²: {r2:.4f}")
    print(f"\n基线对比:")
    print(f"  Naive MAE: ${naive_mae:.2f}")
    print(f"  相对改进: {improvement:.1f}%")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'predictions': y_pred,
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'naive_mae': naive_mae,
            'improvement': improvement
        },
        'feature_importance': importance
    }


def train_binary_model(X_train, y_train, X_test, y_test) -> dict:
    """
    训练二分类模型 (预测Spread方向)
    """
    print("\n" + "=" * 50)
    print("训练二分类模型 (预测Spread方向)")
    print("=" * 50)

    cat_features = ['target_hour', 'target_dow', 'target_month', 'target_is_weekend',
                    'target_is_peak', 'target_is_summer', 'target_day_of_month', 'target_week']
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function='Logloss',
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
        cat_features=cat_indices,
        auto_class_weights='Balanced'  # 处理类别不平衡
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # 基线: 预测多数类
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    print(f"\n模型性能:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Precision: {precision*100:.1f}%")
    print(f"  Recall: {recall*100:.1f}%")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"\n基线对比:")
    print(f"  多数类准确率: {baseline_acc*100:.1f}%")
    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  [[TN={cm[0,0]:5d}, FP={cm[0,1]:5d}]")
    print(f"   [FN={cm[1,0]:5d}, TP={cm[1,1]:5d}]]")

    # 特征重要性
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'baseline_accuracy': baseline_acc
        },
        'confusion_matrix': cm,
        'feature_importance': importance
    }


def train_multiclass_model(X_train, y_train, X_test, y_test) -> dict:
    """
    训练多分类模型 (预测Spread区间)
    """
    print("\n" + "=" * 50)
    print("训练多分类模型 (预测Spread区间)")
    print("=" * 50)

    cat_features = ['target_hour', 'target_dow', 'target_month', 'target_is_weekend',
                    'target_is_peak', 'target_is_summer', 'target_day_of_month', 'target_week']
    cat_indices = [i for i, col in enumerate(X_train.columns) if col in cat_features]

    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function='MultiClass',
        early_stopping_rounds=100,
        verbose=100,
        random_seed=42,
        cat_features=cat_indices,
        auto_class_weights='Balanced'
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )

    # 预测
    y_pred = model.predict(X_test).flatten()
    y_prob = model.predict_proba(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # 基线: 预测最频繁类
    most_common = y_train.value_counts().idxmax()
    baseline_acc = (y_test == most_common).mean()

    print(f"\n模型性能:")
    print(f"  Accuracy: {accuracy*100:.1f}%")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    print(f"\n基线对比:")
    print(f"  最频繁类准确率: {baseline_acc*100:.1f}%")

    print(f"\n分类报告:")
    class_labels = ['< -$20', '-$20~-$5', '-$5~$5', '$5~$20', '>= $20']
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩阵:")
    print(cm)

    # 特征重要性
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_prob,
        'metrics': {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'baseline_accuracy': baseline_acc
        },
        'confusion_matrix': cm,
        'feature_importance': importance
    }


def save_results(
    output_dir: str,
    reg_result: dict,
    bin_result: dict,
    multi_result: dict,
    split_data: dict
) -> None:
    """保存模型和结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型
    reg_result['model'].save_model(os.path.join(output_dir, 'regression_model.cbm'))
    bin_result['model'].save_model(os.path.join(output_dir, 'binary_model.cbm'))
    multi_result['model'].save_model(os.path.join(output_dir, 'multiclass_model.cbm'))

    print(f"\n模型已保存到: {output_dir}")

    # 保存评估结果
    results_summary = {
        'regression': reg_result['metrics'],
        'binary': bin_result['metrics'],
        'multiclass': multi_result['metrics'],
        'training_samples': len(split_data['X_train']),
        'test_samples': len(split_data['X_test']),
        'feature_count': len(split_data['feature_cols']),
        'timestamp': datetime.now().isoformat()
    }

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)

    # 保存特征重要性
    reg_result['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance_regression.csv'), index=False)
    bin_result['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance_binary.csv'), index=False)
    multi_result['feature_importance'].to_csv(
        os.path.join(output_dir, 'feature_importance_multiclass.csv'), index=False)

    # 保存预测结果 (用于回测)
    predictions_df = pd.DataFrame({
        'timestamp': split_data['timestamps_test'].values,
        'actual_spread': split_data['y_reg_test'].values,
        'pred_spread': reg_result['predictions'],
        'actual_direction': split_data['y_bin_test'].values,
        'pred_direction': bin_result['predictions'],
        'pred_prob': bin_result['probabilities'],
        'actual_class': split_data['y_multi_test'].values,
        'pred_class': multi_result['predictions']
    })
    predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

    print(f"结果已保存")


def train_all_models(
    input_path: str,
    output_dir: str,
    test_ratio: float = 0.2
) -> None:
    """
    主函数: 训练所有模型
    """
    print("=" * 60)
    print("RTM-DAM Delta 模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据...")
    df = load_data(input_path)
    print(f"   总样本数: {len(df):,}")

    # 2. 准备特征和目标
    print("\n2. 准备特征和目标变量...")
    data = prepare_features_targets(df)
    print(f"   特征数: {len(data['feature_cols'])}")

    # 3. 切分数据集
    print("\n3. 切分训练/测试集...")
    split_data = split_train_test(data, test_ratio)
    print(f"   训练集: {len(split_data['X_train']):,}")
    print(f"   测试集: {len(split_data['X_test']):,}")

    # 4. 训练回归模型
    reg_result = train_regression_model(
        split_data['X_train'], split_data['y_reg_train'],
        split_data['X_test'], split_data['y_reg_test'],
        split_data['naive_test']
    )

    # 5. 训练二分类模型
    bin_result = train_binary_model(
        split_data['X_train'], split_data['y_bin_train'],
        split_data['X_test'], split_data['y_bin_test']
    )

    # 6. 训练多分类模型
    multi_result = train_multiclass_model(
        split_data['X_train'], split_data['y_multi_train'],
        split_data['X_test'], split_data['y_multi_test']
    )

    # 7. 保存结果
    print("\n" + "=" * 50)
    print("保存模型和结果")
    print("=" * 50)
    save_results(output_dir, reg_result, bin_result, multi_result, split_data)

    # 打印总结
    print("\n" + "=" * 60)
    print("训练完成 - 结果总结")
    print("=" * 60)
    print(f"\n回归模型:")
    print(f"  MAE: ${reg_result['metrics']['mae']:.2f} (基线: ${reg_result['metrics']['naive_mae']:.2f})")
    print(f"  改进: {reg_result['metrics']['improvement']:.1f}%")
    print(f"\n二分类模型:")
    print(f"  Accuracy: {bin_result['metrics']['accuracy']*100:.1f}%")
    print(f"  AUC: {bin_result['metrics']['auc']:.4f}")
    print(f"\n多分类模型:")
    print(f"  Accuracy: {multi_result['metrics']['accuracy']*100:.1f}%")
    print(f"  Macro F1: {multi_result['metrics']['f1_macro']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Delta prediction models')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input feature CSV')
    parser.add_argument('--output-dir', '-o', type=str, required=True,
                        help='Output directory for models')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    args = parser.parse_args()

    train_all_models(
        input_path=args.input,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio
    )


if __name__ == "__main__":
    main()
